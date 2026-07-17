# 机器学习模型推理加速完整手册

> 面试复习 + 实战 Playbook | 基于 voice_safety P2.5 项目实战整理
>
> 涵盖：加速分级 · 图优化原理 · 推理框架全景 · 量化技术 · LLM 特有优化 · 硬件视角 · 面试题清单

---

## 目录

- [Part 1. 全景：为什么推理需要单独加速](#part-1-全景为什么推理需要单独加速)
- [Part 2. 加速的五个层级（金字塔模型）](#part-2-加速的五个层级金字塔模型)
- [Part 3. 图优化完整分类（L1-L6）](#part-3-图优化完整分类l1-l6)
- [Part 4. 主流推理框架矩阵](#part-4-主流推理框架矩阵)
- [Part 5. 模型类型 × 推理方案选型](#part-5-模型类型--推理方案选型)
- [Part 6. 量化技术家族](#part-6-量化技术家族)
- [Part 7. LLM 推理专属优化](#part-7-llm-推理专属优化)
- [Part 8. 前沿探索（2025-2026）](#part-8-前沿探索2025-2026)
- [Part 9. 硬件视角](#part-9-硬件视角)
- [Part 10. 多卡并行推理](#part-10-多卡并行推理)
- [Part 11. 性能分析实战](#part-11-性能分析实战)
- [Part 12. 常见坑与踩雷](#part-12-常见坑与踩雷)
- [Part 13. 面试高频问题清单](#part-13-面试高频问题清单)
- [Part 14. 实战案例：voice_safety 全流程](#part-14-实战案例voice_safety-全流程)
- [附录 A. 术语表](#附录-a-术语表)
- [附录 B. 参考资料](#附录-b-参考资料)

---

## Part 1. 全景：为什么推理需要单独加速

### 1.1 训练 vs 推理的特性差异

| 维度 | 训练 (Training) | 推理 (Inference) |
|---|---|---|
| 计算模式 | Forward + Backward | Forward only |
| Batch 大小 | 大 (32-1024)，追求吞吐 | 小 (1-32)，追求延迟 |
| 精度需求 | FP32/BF16（数值稳定） | FP16/INT8 皆可（可容忍精度损失） |
| 内存分配 | 动态（梯度、优化器状态） | 静态（可预分配） |
| 图结构 | 允许运行时改变（PyTorch eager） | 固定（利于编译） |
| 目标指标 | Loss ↓ + 收敛速度 | **P99 latency + QPS + $/req** |
| 部署形态 | 集群训练几天 | 生产服务 7×24 小时 |

**核心观察**：推理场景下，模型图是**静态的**、Batch 是**小的**、精度是**可损失的**——这些特性正是编译优化的沃土。

### 1.2 推理性能三大指标

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│    Latency (延迟)  ← 单请求端到端时间               │
│         ↑                                           │
│    Throughput (吞吐)  ← 单位时间处理请求数 (QPS)    │
│         ↑                                           │
│    Cost ($ / 1M req)  ← 每百万请求成本              │
│                                                     │
└─────────────────────────────────────────────────────┘

三者矛盾统一:
  · 提 latency 手段 → 降 batch → QPS 下降
  · 提 QPS 手段 → 增 batch → latency 上升
  · 降 cost → 用小 GPU/量化 → latency 或精度受损
```

**大部分推理优化 = 在这三角里挪重心。**

### 1.3 生产瓶颈来源（真实分布）

从我们 voice_safety 项目实测 + 业界经验总结：

```
30%  GPU 计算 (kernel efficiency)
25%  内存/PCIe 数据搬运 (H2D/D2H)
20%  Kernel launch overhead
15%  Framework overhead (Python GIL / Go CGO)
10%  网络/序列化 (gRPC / protobuf)
```

**含义**：单纯堆 GPU 算力（换 A30 → H100）只能解 30% 的问题；剩下 70% 得靠软件栈。

---

## Part 2. 加速的五个层级（金字塔模型）

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│  Level 5: 服务化融合 (vLLM/TGI/SGLang)                    │  ← LLM 专属
│  ─────────────────────────────                            │
│  Level 4: 算法/模型 (蒸馏/剪枝/稀疏)                      │  ← 需重训
│  ─────────────────────────────                            │
│  Level 3: 精度/量化 (FP16/INT8/FP8/INT4)                  │  ← 精度权衡
│  ─────────────────────────────                            │
│  Level 2: 编译器 (TRT/TVM/XLA/MLIR)                       │  ← 编译期优化
│  ─────────────────────────────                            │
│  Level 1: Runtime (ONNX Runtime/LibTorch)                 │  ← 运行时优化
│  ─────────────────────────────                            │
│  Level 0: 硬件 (GPU/TensorCore/HBM/NVLink)                │  ← 硬件基础
│                                                           │
└───────────────────────────────────────────────────────────┘

自底向上, ROI 递减但工作量递增
```

### Level 0: 硬件层

**关键组件**：

| 组件 | 作用 | 性能指标 |
|---|---|---|
| CUDA Cores | 通用并行计算 | FP32 TFLOPS |
| **Tensor Cores** | 矩阵乘专用加速单元 | FP16/INT8/FP8 TFLOPS (是 CUDA 的 8-16x) |
| **HBM (High Bandwidth Memory)** | GPU 显存 | 带宽 (GB/s) 是关键 |
| **NVLink** | 多卡互联 | 900 GB/s (H100)，远快过 PCIe 128 GB/s |
| **L2 Cache** | 片上缓存 | 命中率决定 memory-bound 场景性能 |

**主流 GPU 关键规格**：

| GPU | 架构 | FP16 TensorCore | HBM 带宽 | 显存 | Compute Cap |
|---|---|---:|---:|---:|---:|
| **T4** | Turing | 65 TFLOPS | 320 GB/s | 16GB | SM_75 |
| **A10** | Ampere | 125 TFLOPS | 600 GB/s | 24GB | SM_86 |
| **A30** (我们用的) | Ampere | 165 TFLOPS | 933 GB/s | 24GB | SM_80 |
| **A100** | Ampere | 312 TFLOPS | 2039 GB/s | 40/80GB | SM_80 |
| **L40S** | Ada Lovelace | 366 TFLOPS | 864 GB/s | 48GB | SM_89 |
| **H100** | Hopper | 989 TFLOPS + FP8 | 3350 GB/s | 80GB | SM_90 |
| **H200** | Hopper | 989 TFLOPS + FP8 | 4800 GB/s | 141GB | SM_90 |
| **B200** | Blackwell | 2250 TFLOPS + FP4 | 8000 GB/s | 192GB | SM_100 |

**看数据的关键**：
- **训练** → 看 FP16 TFLOPS（计算密集）
- **推理** → **看 HBM 带宽**（大部分推理是 memory-bound！）
- **LLM decode** → **只看带宽**（每 token 只 1 次 matmul，全靠带宽扫权重）

### Level 1: Runtime 层

**演进路径**：

```
Framework Native (PyTorch eager)
       ↓ [导出静态图]
TorchScript / TF SavedModel
       ↓ [跨框架统一 IR]
ONNX (Open Neural Network Exchange)
       ↓ [ONNX Runtime 加载]
Execution Provider 选择：
   ├─ CPUExecutionProvider     (通用 CPU, SIMD)
   ├─ CUDAExecutionProvider    (通用 NVIDIA GPU, 走 cuDNN)
   ├─ TensorRTExecutionProvider (深度编译 → Level 2)
   ├─ CoreMLExecutionProvider  (Apple Neural Engine)
   ├─ OpenVINOExecutionProvider (Intel CPU/iGPU/VPU)
   ├─ ROCmExecutionProvider    (AMD GPU)
   └─ WebGPUExecutionProvider  (浏览器)
```

**ONNX Runtime 做的事**：
- 加载 ONNX 图
- 做基础图优化（L1-L2 图重写、简单融合）
- 分派到 Execution Provider 执行
- 提供跨框架、跨语言、跨硬件的统一接口

**优点**：一份模型，多硬件跑，Python/C++/Go 语言都有 binding。

### Level 2: 编译器层

**主流推理编译器**：

| 编译器 | 归属 | 目标硬件 | 特点 |
|---|---|---|---|
| **TensorRT** | NVIDIA | NVIDIA GPU 独占 | 手写 kernel 库最全，推理性能天花板 |
| **TensorRT-LLM** | NVIDIA | NVIDIA GPU | TRT 的 LLM 分支，含 continuous batching |
| **TVM** | Apache | GPU/CPU/Edge/多加速卡 | 学术界主流，Auto-schedule 强大 |
| **XLA** | Google | TPU/GPU/CPU | TPU 唯一路径，JAX 默认后端 |
| **MLIR / IREE** | LLVM Foundation | 通用 | 下一代 IR，未来趋势 |
| **Torch Inductor** | Meta | GPU/CPU | PyTorch 2.0 内置，`torch.compile` 后端 |
| **Torch-TRT** | NVIDIA + Meta | NVIDIA GPU | PyTorch 到 TRT 的桥梁 |
| **ONNX-MLIR** | IBM | 多 | ONNX → MLIR → LLVM |

**编译器 vs Runtime 的关键区别**：

```
Runtime (ORT):                       Compiler (TRT):
   动态解释图, 每层调 kernel              静态编译, 生成一份二进制
   优化 = 通用图重写                       优化 = 深度融合 + kernel autotune
   加速 2-3x                              加速 5-10x
   即插即用, 无需硬件绑定                  产物强绑定 GPU + 版本
```

### Level 3-5: 精度 / 算法 / 服务化

详见 [Part 6](#part-6-量化技术家族) / [Part 7](#part-7-llm-推理专属优化)。

---

## Part 3. 图优化完整分类（L1-L6）

**图优化 = 编译期对计算图做的所有变换**。TensorRT / TVM / XLA 都会做这些，只是覆盖深度不同。

### L1: 图重写 (Graph Rewriting) —— 消除冗余

```python
# 例 1: 常量折叠
x + 0                → x
x * 1                → x
Reshape(Reshape(x))  → Reshape(x)   # 折叠嵌套

# 例 2: 死码消除
y = f(x)
z = g(x)   # z 未被后续使用 → 删除 g
return y

# 例 3: 代数化简
(x + y) - y          → x
Concat([x])          → x            # 单元素 concat
Transpose(Transpose(x, [1,0]), [1,0])  → x  # 双次转置相消
```

**收益**：不做主要计算加速，但简化图利于后续步骤。

### L2: 算子融合 (Operator Fusion) —— 核心收益

**垂直融合（串行合并）**：

```
不融合:                          融合后:
┌─────┐                          ┌─────────────────┐
│Conv │                          │ Conv+BN+ReLU    │
└──┬──┘                          │ (single kernel) │
   ↓ 写显存                      └─────────────────┘
┌─────┐                          输入 → 寄存器 → 输出
│ BN  │
└──┬──┘                          省了 2 次显存往返!
   ↓ 写显存                      (通常内存带宽是瓶颈)
┌─────┐
│ReLU │
└─────┘
```

**水平融合（并行合并）**：

```
Attention 的 Q, K, V 三个 projection:

不融合:                       融合后:
  Q = MatMul(x, W_q)           X = concat(W_q, W_k, W_v)  # 拼权重
  K = MatMul(x, W_k)           Y = MatMul(x, X)           # 1 次 matmul
  V = MatMul(x, W_v)           Q, K, V = split(Y)
  (3 个 kernel launch)         (1 个 kernel launch)
```

**跨层融合（Element-wise 链接）**：
```
Add + Mul + Sigmoid + Cast → 单个融合 kernel（GPU 上跑一次）
```

**Voice_safety 案例**：TRT 8.6.1 尝试融合 `LayerNorm + Shuffle + Transpose + Shuffle` 时 crash（Issue #3490），TRT 10 修复。这类融合失败是编译器 bug 高发区。

### L3: 内存布局 (Memory Layout Optimization)

```
数据格式选择:
  NCHW  (PyTorch 默认, cuDNN 常用)
    Batch × Channel × Height × Width
  NHWC  (TensorFlow 默认, TensorCore 高效)
    Batch × Height × Width × Channel
  NC/32HW32  (INT8 TensorCore 专用格式)

编译器会插入 Format 转换节点, 选整体最优.
```

**In-place operation**：
```python
# 常规: ReLU 输出新 buffer
y = ReLU(x)   # 新 alloc

# In-place: 直接覆盖 x
ReLU_inplace(x)   # 无新 alloc, 但 x 值被销毁 (需检查后续无使用)
```

**Buffer 生命周期分析 + 复用**：

```
Layer 1: 分配 buf_A, 输出到 buf_A
Layer 2: 用 buf_A → 输出到 buf_B, buf_A 生命周期结束
Layer 3: 用 buf_B → 输出到 ?
         → 可以复用 buf_A (已死), 而不是新 alloc buf_C
```

TRT 静态规划整个 buffer 图，实现零运行时 malloc。

### L4: 精度调度 (Precision Selection)

**混合精度决策**：

```
FP32 保留 (数值敏感层):          FP16/BF16 主体:
  - LayerNorm 的 mean/var         - Conv2d
  - Softmax                        - MatMul (Attention Q@K, Attn@V)
  - Loss (仅训练)                  - Element-wise (ReLU, GeLU)
                                    
Compiler 自动插入 cast:
  fp16_data → cast_fp32 → fp32_op → cast_fp16 → fp16_data
```

**详细精度维度见 [Part 6](#part-6-量化技术家族)。**

### L5: Kernel Autotune —— 最耗时的一步

**做什么**：对每个算子，从多种实现里选最快的。

```
Conv2d 有多少种实现？
  · Winograd (适合 3x3 filter, 小 batch)
  · Implicit GEMM (通用, 大多数场景)
  · Direct Convolution (适合 depthwise / 1x1)
  · FFT-based (适合大 kernel size)

GEMM 有多少种？
  · cuBLAS (NVIDIA 官方)
  · cuBLASLt (heuristic 更好, 支持复杂 fusion)
  · CUTLASS (可定制 shape 和 精度)
  · CUTE (最新, 更 flexible)
  · 手写 kernel (针对特定 shape 定制)

Attention 更多:
  · FlashAttention 2 / 3
  · xFormers memory-efficient
  · cuDNN's fused MHA
  · Paged Attention (vLLM)
```

**Tune 过程**：
```
对每个算子 op:
  for tactic in candidate_kernels(op):
    for trial in range(3):
      t = measure_kernel(tactic, input_shape)
    latency[tactic] = median(trials)
  best = argmin(latency)
```

**这就是 TRT engine build 慢的根源**——每层几十种 tactic 各跑 3 次。voice_safety 24 层 × ~40 tactics × 3 trials = ~2880 次 kernel launch + timing。

**workspace 8GB 就是装这些 tactic 的临时 buffer** 空间。

### L6: Runtime 优化 (Deployment)

```
CUDA Graph capture:
  录制 24 层的 kernel launch 序列成一个 graph
  运行时一次性 launch, 消除每层 launch overhead (~10μs × 层数)
  
Multi-stream 并行:
  独立子图 → 不同 CUDA stream → SM 并行调度
  
Memory pool pre-allocation:
  启动时一次性 alloc, 运行时零 malloc
```

### 图优化收益拆解（voice_safety 实测）

| 优化项 | 加速倍数 |
|---|---:|
| ONNX + CUDA EP (基线) | 1x (~60ms) |
| + L1 图重写 | ~1.1x |
| + L2 算子融合 | ~1.5x |
| + L3 内存布局 | ~1.15x |
| + L4 FP16 混合精度 | ~1.5x |
| + L5 Kernel autotune | ~1.3x |
| + L6 CUDA graph / stream | ~1.1x |
| **综合 (TRT FP16 编译后)** | **~4x (~15ms)** |

---

## Part 4. 主流推理框架矩阵

### 4.1 全框架对比大表

| 框架 | 定位 | 目标模型 | 硬件 | 特色 | 上手难度 |
|---|---|---|---|---|---|
| **PyTorch (eager)** | Framework | 全部 | CPU/GPU | 动态图，开发友好 | ⭐ |
| **TorchScript** | Framework JIT | 静态图能力的 PyTorch 模型 | CPU/GPU | PyTorch 内置，无需转换 | ⭐⭐ |
| **torch.compile** | JIT compiler | PyTorch 2.0+ | GPU/CPU | Torch Inductor 后端 | ⭐⭐ |
| **ONNX Runtime** | Runtime | 通用（跨框架） | 多硬件 | 一份模型多硬件跑 | ⭐⭐ |
| **TensorRT** | Compiler | 通用（编译到 NVIDIA） | NVIDIA GPU | 推理性能天花板 | ⭐⭐⭐ |
| **Torch-TRT** | Bridge | PyTorch 模型 | NVIDIA GPU | PyTorch 直通 TRT | ⭐⭐⭐ |
| **TVM** | Compiler | 通用 | 多硬件（含 edge） | 学术界主流，Auto-schedule | ⭐⭐⭐⭐ |
| **XLA** | Compiler | JAX / TF | TPU / GPU | TPU 唯一路径 | ⭐⭐⭐ |
| **OpenVINO** | Runtime + Compiler | 通用 | Intel CPU/iGPU/NPU | Intel 硬件独占 | ⭐⭐ |
| **CoreML** | Framework | 通用 | Apple 硬件 | iOS/macOS 唯一 | ⭐⭐ |
| **TensorFlow Lite** | Runtime | TF 模型 | Mobile CPU/GPU/NPU | Mobile 端主流 | ⭐⭐ |
| **NCNN / MNN / TNN** | Mobile 推理框架 | ONNX/TF | Mobile CPU/GPU | 腾讯/阿里等国产 mobile | ⭐⭐ |
| **Triton Inference Server** | Serving | 全部 | 多硬件 | NVIDIA 出的 serving 层 | ⭐⭐⭐ |
| **vLLM** | LLM Serving | Decoder-only LLM | NVIDIA GPU | PagedAttention 首创 | ⭐⭐⭐ |
| **TGI** | LLM Serving | LLM (HF 生态) | NVIDIA GPU | HuggingFace 官方 | ⭐⭐⭐ |
| **TensorRT-LLM** | LLM Compiler + Serving | LLM | NVIDIA GPU | NVIDIA 出的 LLM 专版 TRT | ⭐⭐⭐⭐ |
| **SGLang** | LLM Serving | LLM | NVIDIA GPU | RadixAttention，前沿 | ⭐⭐⭐ |
| **LMDeploy** | LLM Serving | LLM | NVIDIA GPU | 商汤开源，性能强 | ⭐⭐⭐ |
| **DeepSpeed-Inference** | LLM Serving | LLM | 多 GPU | Microsoft，多卡并行强 | ⭐⭐⭐⭐ |
| **MLC-LLM** | Compiler | LLM | 多硬件（含手机） | TVM 生态，跨设备 | ⭐⭐⭐⭐ |

### 4.2 场景选型 Decision Tree

```
问题: 你的场景是什么?
│
├─ 只跑 CNN/ViT/BERT 等 encoder 模型 (分类/检索/理解)
│    → NVIDIA GPU 生产? → TensorRT (最优)
│    → Intel CPU? → OpenVINO
│    → 需要跨硬件? → ONNX Runtime + 多个 EP
│    → 需要 mobile? → NCNN / TFLite / CoreML
│
├─ 跑 encoder-decoder (T5, Whisper, BART)
│    → 生产用 TensorRT (支持) 或 TensorRT-LLM (含 continuous batching)
│    → 研究用 HuggingFace + Flash Attention
│
├─ 跑 decoder-only LLM (GPT/LLaMA/Qwen/DeepSeek)
│    → 追求极致 QPS + 稳定 → TensorRT-LLM
│    → 灵活性 + 上手快 → vLLM
│    → HF 生态无缝 → TGI
│    → 前沿功能 (RadixAttention 等) → SGLang
│    → 多卡大模型 → DeepSpeed-Inference
│
├─ 跑 diffusion (SD, DALL-E, Sora)
│    → Diffusers 库 + xFormers + FlashAttention
│    → 极致优化 → TensorRT diffusion (NVIDIA 有专门 pipeline)
│    → Mobile → Diffusers ONNX + CoreML
│
├─ 研究/实验阶段
│    → PyTorch eager + torch.compile
│
└─ 学术 paper (跨硬件对比)
     → TVM (有 Auto-schedule 论文)
```

---

## Part 5. 模型类型 × 推理方案选型

**为什么模型类型决定方案** —— 因为不同类型模型的**动态性和瓶颈完全不同**。

### 5.1 Encoder-only（BERT / ViT / voice_safety）

**特点**：一次 forward pass，无迭代，输出固定 shape。

**瓶颈**：静态计算（Conv/GEMM/Attention）+ 内存搬运。

**推荐方案**：**TensorRT / OpenVINO / TVM**（静态编译天花板）。

**不适合**：vLLM / TGI（它们的 continuous batching / KV cache 优化用不上）。

### 5.2 Decoder-only（GPT / LLaMA / Qwen / Claude 底层）

**特点**：
- Prefill 阶段：一次性处理 prompt（compute-bound）
- Decode 阶段：每次生成 1 token，KV cache 增长（memory-bound）
- 输出长度不定，可能提前终止

**瓶颈**：
- Prefill：Attention 是 O(N²)
- Decode：**HBM 带宽**（每个 token 要扫一遍权重）
- KV cache 显存占用（长上下文爆炸）

**推荐方案**：**vLLM / TGI / TensorRT-LLM / SGLang**。

### 5.3 Encoder-Decoder（T5 / BART / Whisper 语音识别）

**特点**：Encoder 一次 forward + Decoder 自回归。

**瓶颈**：Encoder 部分是静态，Decoder 部分同 5.2。

**推荐方案**：
- 通用：TensorRT-LLM（支持 encoder-decoder）
- Whisper 专用：Faster-Whisper（CTranslate2 后端）

### 5.4 Diffusion（Stable Diffusion / DALL-E）

**特点**：迭代去噪（50 步 x U-Net forward）。

**瓶颈**：U-Net 计算 + Attention（长度较短）。

**推荐方案**：
- 生产：TensorRT （SD 专用 pipeline）
- 研究：Diffusers + xFormers

### 5.5 MoE（Mixture of Experts，如 Mixtral / DeepSeek）

**特点**：每 token 只激活部分专家网络，稀疏计算。

**瓶颈**：**Expert 之间的路由 + 通信**（多卡场景是 all-to-all）。

**推荐方案**：**DeepSpeed-Inference / TensorRT-LLM MoE / SGLang**。

---

## Part 6. 量化技术家族

**核心问题**：训练时 FP32/BF16，推理时能不能用更低精度？

### 6.1 精度演进

```
FP32 (32-bit float)   ← 训练标准, 推理浪费
   ↓
BF16 (16-bit brain float, Google) ← 保 FP32 动态范围, 精度减
   ↓
FP16 (16-bit IEEE)   ← A100 TensorCore 加速 2x, 但动态范围小
   ↓
FP8 (E4M3 / E5M2, H100+)  ← Hopper 引入, 计算 2x + 显存减半
   ↓
INT8 (integer quantization) ← 需要 calibration, 精度损失需评估
   ↓
INT4 (extreme quantization) ← LLM weight-only, 显存关键
   ↓
INT2 / 1-bit (BitNet)  ← 前沿探索, 精度大打折扣
```

### 6.2 硬件对精度的支持矩阵

| 精度 | T4 | A30/A100 | L40S | H100 | 说明 |
|---|:---:|:---:|:---:|:---:|---|
| FP32 | ✅ | ✅ | ✅ | ✅ | 训练默认 |
| BF16 | ❌ | ✅ | ✅ | ✅ | Ampere 起支持 |
| FP16 | ✅ | ✅ | ✅ | ✅ | 通用 |
| TF32 | ❌ | ✅ | ✅ | ✅ | Ampere 加速 FP32 计算 |
| INT8 | ✅ | ✅ | ✅ | ✅ | 全通用 |
| **FP8 (E4M3/E5M2)** | ❌ | ❌ | ❌ | ✅ | **Hopper 独占** |
| INT4 (WMMA) | ❌ | ⚠️ 有限 | ⚠️ | ✅ | 部分支持 |
| **FP4** | ❌ | ❌ | ❌ | ❌ | **Blackwell (B200) 引入** |

**看一眼就懂**：H100 是当前推理的甜蜜点（FP8 + 显存带宽 3.35TB/s），Blackwell (B200) 是下一代的答案（FP4 + 8TB/s 带宽）。

### 6.3 量化算法分类

#### 按训练参与度

```
PTQ (Post-Training Quantization) - 训练完再量化, 无需重训
   ├─ Static PTQ  - 需要 calibration dataset 找 activation range
   └─ Dynamic PTQ - 运行时统计 range (慢, 但无 calibration)

QAT (Quantization-Aware Training) - 训练时模拟量化, 精度最好
   → 需要额外 fine-tuning 阶段
```

#### 按量化对象

```
Weight-only Quantization   - 只压缩权重 (LLM 主流, 显存关键)
   → GPTQ, AWQ, GGUF (llama.cpp)

Activation Quantization    - 压缩激活值
   → SmoothQuant (LLM 常用)

Full Quantization          - 权重 + 激活 都量化
   → INT8 推理主流方案
```

### 6.4 主流量化算法

| 算法 | 类型 | 精度 | 主要用途 | 特点 |
|---|---|---|---|---|
| **INT8 PTQ (TRT/ORT)** | 全量化 PTQ | INT8 | CNN/BERT 推理 | 需 calibration set，成熟 |
| **QAT** | 全量化 QAT | INT8/INT4 | 精度苛刻场景 | 需重训 |
| **SmoothQuant** | 激活量化 | INT8 | LLM 推理 | 迁移激活异常值到权重 |
| **GPTQ** | Weight-only PTQ | INT4 | LLM | 逐层最小化重构误差 |
| **AWQ** (Activation-aware) | Weight-only PTQ | INT4 | LLM | 保护重要通道 |
| **GGUF** (llama.cpp) | Weight-only | Q2/Q4/Q5/Q6/Q8 | LLM 本地部署 | CPU 友好，量化格式丰富 |
| **FP8 (E4M3)** | 全精度降低 | FP8 | LLM 训练+推理 | H100 硬件加速，精度好 |
| **BitNet 1.58** | 极致权重量化 | Ternary (-1/0/1) | 前沿 LLM | 显存节省 10x+ |

### 6.5 精度损失评估流程

```
1. Baseline 建立 (FP32 或 FP16)
   → 记录关键指标 (accuracy, F1, BLEU, cosine similarity)

2. 量化目标选择 (INT8 / FP8 / INT4)

3. Calibration (PTQ 需要)
   → 用 100-1000 条代表性数据 collect activation range

4. 量化推理
   → 跑同样的 test set

5. 精度对比
   → 掉点 < 0.5%? → Accept
   → 掉点 > 1%? → 试 mixed precision (关键层保 FP16)
   → 精度不可接受? → QAT (重训) 或降级到 FP16

6. 加速验证 (光精度过关不够, 得实测有加速)
   → TRT_FP16: ~2x
   → TRT_INT8: ~4x
   → TRT_FP8 (H100): ~2x + 显存半
```

---

## Part 7. LLM 推理专属优化

**核心观察**：LLM 推理与传统模型完全不同——**decode 阶段是 memory-bound + 迭代动态**。

### 7.1 LLM 推理的两阶段

```
┌────────────────────────────────────────────────────────────┐
│  Prefill 阶段 (一次性)                                     │
│  ─────────────                                             │
│  输入: prompt (N tokens)                                   │
│  计算: 24-96 layers × (attention O(N²) + FFN O(N))         │
│  瓶颈: COMPUTE-BOUND (大 batch matmul, GPU 打满)           │
│  时长: ~100ms - 数秒 (取决于 prompt 长度)                  │
│                                                            │
│  产物: 每层的 KV cache (为后续 decode 复用)                │
├────────────────────────────────────────────────────────────┤
│  Decode 阶段 (自回归迭代, 生成 output)                     │
│  ────────────                                              │
│  每步:                                                     │
│    输入: 上一步生成的 1 token                              │
│    计算: 24-96 layers × (attention O(1) + FFN O(1))        │
│    瓶颈: MEMORY-BOUND (每 token 全权重扫一遍 HBM)          │
│    时长: 每 token ~10-50ms                                 │
│                                                            │
│  总耗时: 500 tokens × 20ms = 10s (端到端)                  │
└────────────────────────────────────────────────────────────┘
```

**关键数字**：Decode 阶段每 token 要扫一遍整个模型的权重（几十 GB），完全被 HBM 带宽 bound。所以 **LLM 推理的核心指标是 HBM 带宽利用率**。

### 7.2 KV Cache —— 显存杀手 + 优化核心

**为什么需要 KV cache**：
```
Attention 公式: Attn(Q, K, V) = softmax(Q @ K^T / √d) @ V

Decode 时:
  第 t 步生成 token → Q_t (1 个新的 query)
  K/V 需要包含所有历史 tokens: K_1..t, V_1..t

  如果不缓存, 每步重算 K_1..t, V_1..t → O(t²) 累计
  缓存后: 每步只算 K_t, V_t 追加 → O(t) 累计
```

**KV cache 大小估算**：
```
每 token 每 layer 的 KV = 2 * num_heads * head_dim * dtype_bytes

Llama-2 70B (80 层, 64 heads, head_dim=128, FP16):
  每 token KV = 2 × 64 × 128 × 2 × 80 = 2.6 MB
  4K 上下文 = 4096 × 2.6 MB = 10 GB (!! 超过 A100 显存一半)
  32K 上下文 = 32768 × 2.6 MB = 80 GB (!! 超过 H100 显存)
```

**这就是为什么 LLM 长上下文这么费显存**。

### 7.3 PagedAttention (vLLM 核心创新)

**问题**：KV cache 是变长的（生成还没结束不知道多长），如果按最大长度预分配显存，浪费 60-80%。

**vLLM 的解**：模仿 OS 虚拟内存分页管理。

```
传统 KV cache 分配:
  Request A (max_len=2048): 分配 2048 × 2.6MB = 5.3GB
  实际用 500 tokens → 浪费 1.5 × 500 × 2.6MB = 4GB

PagedAttention:
  Physical memory: 分成 4KB pages (KV block)
  Logical KV cache: 用 page table 映射
  
  Request A 生成时按需 alloc page:
    生成第 1-16 token → page 5 (1 KV block = 16 tokens)
    生成第 17-32 token → page 12
    ...
  
  显存利用率: 96%+
  同时可服务 request 数: 4-5x
```

**副作用**：需要专门写 Paged Attention kernel（不能直接用 cuDNN），vLLM 团队手写了 CUDA。

### 7.4 Continuous Batching (In-flight Batching)

**问题**：LLM 请求长度差异极大（有的 10 tokens 有的 1000），传统 batch 会等最长的完成才能开始下一批。

**传统 static batching**：
```
Batch [A(len=500), B(len=100), C(len=50)]
  → 处理 500 步, 但 B 在第 100 步就完成了(浪费), C 在第 50 步(浪费更多)
  → GPU 利用率 <30%
```

**Continuous batching**：
```
时间 t=50: A(50%), B(50%), C(100% ✅)
             → 立即将 C 移出 batch
             → 从 queue 里插入 D (刚到达的新请求)
             → batch 变成 [A, B, D]
             
时间 t=100: A(20%), B(100% ✅), D(50%)
             → 移出 B, 插入 E
             → batch 变成 [A, D, E]
```

关键：**请求进出 batch 的时机不需要等 batch 完成**。

**收益**：GPU 利用率从 30% → 80%+，QPS 提升 2-5x。

### 7.5 Speculative Decoding (投机解码)

**问题**：decode 每步只生成 1 token，GPU 大部分 SM 空闲。

**核心思路**：用一个**小模型**先"猜"多个 token，用**大模型**一次性验证。

```
Step 1: Draft model (7B) 猜 5 个 token: [A, B, C, D, E]
Step 2: Large model (70B) 一次前向:
         检查 [A, B, C, D, E] 里哪些猜对了
         → 假设前 3 个对 (A, B, C), D 不对 → 采纳 A, B, C
         → 用 large model 的输出替代 D (X), 丢弃 E
         → 净生成 4 tokens

命中率高 (~70%) 时:
  1 次 large model forward = 生成 3-4 tokens
  相当于加速 3-4x
```

**变体**：Medusa（多头 draft）、EAGLE（更强 draft）、Lookahead Decoding（并行 branch）。

### 7.6 Chunked Prefill / Split Prefill

**问题**：Prefill (compute-bound) 和 decode (memory-bound) 混合时，一个长 prompt 会阻塞所有 decode 请求。

**解**：把长 prompt 切成 chunks，与 decode 请求混合调度，让 GPU 同时跑两种负载（compute + memory 交叉利用）。

### 7.7 Multi-LoRA Serving

**场景**：一个 base model 上跑多个 LoRA adapter（微调）。

**天真方案**：每个 LoRA 一个 model instance → 显存 × N。

**Multi-LoRA**：base model 只存一份，LoRA weights (~10MB each) 动态切换，一份服务多个 fine-tuned 用户。

**支持框架**：vLLM, TGI, Punica。

---

## Part 8. 前沿探索（2025-2026）

> **时间戳**：2026-07-15 快照。本章记录 2024 年以来已进入生产的 LLM 推理关键技术。所有数字均来自论文原文、官方 blog 或框架 release notes。

### 8.1 P/D 分离（Prefill/Decode Disaggregation）

#### 原理

**Prefill vs Decode 的算术强度差异**（Part 7.1 结论）：

```
Prefill:  N tokens 一次输入
  Attention: O(N²) FLOPs, N² 大矩阵乘
  FFN:       O(N)  FLOPs
  Arithmetic Intensity ≈ 200 FLOPs/byte
  → Compute-bound, TensorCore 打满

Decode:   1 token 一次输入
  Attention: O(1) FLOPs (与 KV cache 长度成正比的一维向量点积)
  FFN:       O(1) FLOPs
  Arithmetic Intensity ≈ 2 FLOPs/byte
  → Memory-bound, HBM 带宽 bound
```

**Collocated 方案的干扰模型**：

```
时间线 (同一 GPU 上):
  t=0     Batch: [A_prefill(N=2048), B_decode, C_decode]
  t=0     prefill A 占满 SM
  t=100ms A 完成 → B/C 才能推进一步
          期间 B/C 的 TPOT 全部 = 100ms (被 A 阻塞)
          实际 B/C decode 一步只需 ~15ms
```

**分离架构**：

```
                    ┌───────────────┐
                    │  Prefill Pool │
Request ──Router─→  │  · 独立 TP/PP │
                    │  · compute 优化│
                    └──────┬────────┘
                           │ 传 KV cache
                           ▼
                    ┌───────────────┐
                    │  Decode Pool  │
                    │  · 独立 TP/PP │
                    │  · 带宽优化   │
                    └──────┬────────┘
                           │
                           ▼
                         Response
```

#### DistServe 的 TP/PP 选型公式（M/D/1 排队模型）

单实例平均 TTFT（服务时间 D，请求率 R）：

```
单卡:        Avg_TTFT = D + R·D² / (2·(1−R·D))

2-way inter-op (PP):
             Avg_TTFT = D + R·D² / (4·(2−R·D))

2-way intra-op (TP), K ∈ (1, 2) 为加速系数:
             Avg_TTFT = D/K + R·D² / (2K·(K−R·D))
```

决策规则：

| 条件 | 偏向 |
|---|---|
| 低到达率 R（执行时间占比高） | **TP** |
| 高到达率 R（排队时间占比高） | **PP** |
| 紧 TTFT SLO | **TP** |
| 弱互联（K 小） | **PP**（TP 收益打折） |

**Prefill vs Decode 各自最优**（DistServe 实测，OPT-175B on ShareGPT）：

| 阶段 | 最优 (inter_op, intra_op) | 原因 |
|---|---|---|
| Prefill | (3, 3) | 中等 batch，TP 减 TTFT |
| Decode | (3, 4) | Batch 大，需要更大 TP 减 TPOT |

#### KV Cache 传输的量级估算

```
KV_size = 2 · num_heads · head_dim · seq_len · num_layers · dtype_bytes

OPT-66B, 512 tokens: 1.13 GB
Llama-70B, 4K tokens: 10.6 GB
DeepSeek-V3 (MLA), 4K tokens: ~0.7 GB
```

跨节点传输耗时：

| 链路 | 峰值 | 10 GB 传输 |
|---|---:|---:|
| NVLink (H100 8卡) | 900 GB/s | 11 ms |
| InfiniBand NDR 400G | 50 GB/s | 200 ms |
| 25 GbE 以太网 | 3.1 GB/s | 3200 ms |

#### Layer-wise 传输 overlap 计算（Mooncake 实现）

```
Prefill 侧, 逐 layer 处理:
  for layer in 0..L-1:
      wait(async_load[layer])         # 等 prefix cache 就位
      trigger(async_load[layer+1])    # 预取下一层
      run_attention(layer)            # GPU 计算
      trigger(async_store[layer])     # 异步传出本层 KV
  wait(all_pending_stores)

Wall time = max(load_time, prefill_time)   # 而非串行相加
```

DistServe 论文实测：OPT-175B on ShareGPT，KV 传输占端到端 < 0.1%，>95% 请求传输 < 30ms。

#### Mooncake 存储层次

```
GPU HBM       80 GB    900 GB/s      当前推理 batch 的 KV
CPU DRAM     500 GB     20 GB/s      prefix cache（近期热点会话）
SSD (NVMe)   10 TB      3 GB/s       长会话历史

KV block 粒度: 512 tokens
Hash tag:      block 内容 + 全部前缀（用于 dedup + 复用）
Eviction:      LRU（论文实测最优）
```

**Conductor 调度算法**（选 prefill 节点）：

```
for candidate in prefill_instances:
    prefix_match_len = longest_prefix(request, candidate.cache)
    T_queue = estimate_queue_time(candidate)
    T_prefill = estimate_prefill_time(len(request) - prefix_match_len)
    T_TTFT_est = T_queue + T_prefill

    # 如果远端命中的前缀比本地多（超过阈值），考虑跨节点拉 prefix
    if best_remote_match - local_match > kvcache_balancing_threshold:
        T_TTFT_est += T_transfer

    if T_TTFT_est > TTFT_SLO:
        continue  # 不满足 SLO 直接跳过

select instance with min(T_TTFT_est)
if none selected: return HTTP 429
```

#### 数据

| 系统 | 论文 | 数据集 / 模型 | 报告数字 |
|---|---|---|---|
| **DistServe** | OSDI'24 | ShareGPT / OPT-175B | 请求率 **4.48x**；SLO 可紧缩 **10.2x** |
| **Splitwise** | ISCA'24 | Azure trace | 同资源吞吐 **2.35x**；同吞吐成本 **−20%**（对应 1.4x throughput/$） |
| **Mooncake** | FAST'25 | Kimi 生产 trace | 长上下文吞吐 **+525%**；实际处理请求 **+75%** |
| **Mooncake Kimi K2 (1T)** | 官方 blog | 128× H200 | Prefill **224k tokens/s**；Decode **288k tokens/s** |

#### 工程支持

| 框架 | 版本 | Connector | 传输协议 |
|---|---|---|---|
| **vLLM** | ≥ 0.6.0 | `NixlConnector` / `MooncakeConnector` / `MooncakeStoreConnector` | NCCL P2P / RDMA / Mooncake Transfer Engine |
| **SGLang** | 主干 | HiCache（layer-wise 传输） | Mooncake Transfer Engine |
| **TensorRT-LLM** | Dynamo 框架 | NVIDIA 官方 | NCCL |
| **LMDeploy** | v0.12+ | DLSlime / Mooncake protocol | RDMA |
| **DeepSeek 开源栈** | V3 起 | 自研 | 内部 |

vLLM 配置示例：

```yaml
# vllm serve --kv-transfer-config <file>
kv_connector: MooncakeConnector
kv_role: kv_producer          # prefill 节点角色
kv_rank: 0
kv_parallel_size: 2
```

---

### 8.2 MTP（Multi-Token Prediction）

DeepSeek-V3 引入，模型自带的推测解码机制。

#### 架构（DeepSeek-V3 §2.2）

```
主模型 (61 层 MoE Transformer)
输入序列 [t₁, ..., tₙ] → 每 token 输出 hidden state h_i

MTP Module k (k = 1..D):
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │   前一 depth 表征 h_i^(k-1)  ─── RMSNorm ──┐           │
  │                                            ├── concat ─┤
  │   未来 token embed e(t_{i+k}) ── RMSNorm ──┘           │
  │       (embedding 层与主模型共享)                       │
  │                                              │         │
  │                                              ▼         │
  │                                     Linear M_k         │
  │                                     (d × 2d)           │
  │                                              │         │
  │                                              ▼         │
  │                                     Transformer 块     │
  │                                              │         │
  │                                              ▼         │
  │                                     h_i^(k)            │
  │                                              │         │
  │                                              ▼         │
  │                                     共享 output head   │
  │                                              │         │
  │                                              ▼         │
  │                                     p_{i+k+1}          │
  │                                                        │
  └────────────────────────────────────────────────────────┘

关键约束: h_i^(k) 依赖 h_i^(k-1)  → sequential, 保完整因果链
         (与 Meta 2024 论文的 D 个 parallel head 结构不同)
```

#### 训练目标

```
每个 depth 的 loss (T = 序列长度):
   L_MTP^k = −(1/T) · Σ_{i=2+k..T+1} log P_i^k[t_i]      # cross-entropy

总 MTP loss (D 个 depth 平均, λ 加权):
   L_MTP = (λ/D) · Σ_{k=1..D} L_MTP^k

总训练 loss:
   L_total = L_main + L_MTP
```

DeepSeek-V3 官方设置（论文 §2.2 / §5.4.3）：
- 训练时 **D = 1**（单 MTP module）
- λ 具体数值论文未明确公开

#### 推理路径

**Mode A：discard MTP**
```
主模型独立 forward, 与常规 decode 相同, 无额外开销
```

**Mode B：MTP 作为 speculative decoder**
```
Step 1: 主模型 forward, 得 h_i, 采 token t_{i+1}
Step 2: MTP module k=1, 用 (h_i, e(t_{i+1})) 前向 → 猜 t_{i+2}
Step 3: 主模型 forward [t_{i+1}, t_{i+2}] (一次前向覆盖 2 个位置)
Step 4: 位置 i+2 处主模型输出 vs MTP 猜测 t_{i+2}
        · 一致 → 接受, 净生成 2 tokens
        · 不一致 → 丢弃 MTP 猜测, 保留主模型输出, 净生成 1 token
```

#### 数据

| 来源 | 场景 | 数字 |
|---|---|---|
| DeepSeek-V3 §5.4.3 | 跨话题第 2 token 接受率 | **85%-90%** |
| DeepSeek-V3 官方 | 端到端 Tokens/s | **1.8x** |
| SGLang blog (2025-07) | DeepSeek V3 + MTP | 输出吞吐 **+60%**，质量无损 |
| vLLM 社区 | DeepSeek-R1, k=1 | 接受率 81-82.3%；QPS=1 时 **1.63x**；QPS>8 时加速衰减 |

#### 工程支持

| 框架 | 版本 | 配置 |
|---|---|---|
| **SGLang** | ≥ v0.4.5 | `--speculative-algorithm EAGLE --speculative-draft-model <deepseek-model>` |
| **vLLM** | ≥ 0.7 | `--speculative-config '{"model": "...", "num_speculative_tokens": 1}'` |
| **TensorRT-LLM** | 主干 | Speculative Sampling MTP variant |

**限制**：只能用于**训练阶段启用 MTP** 的模型（DeepSeek-V3 / R1 权重发布已含 MTP heads）。Llama / Qwen 等要用需重训或走 EAGLE。

---

### 8.3 EAGLE（feature-level speculative decoding）

#### 原理

**Feature 定义**（EAGLE-1）：

```
Target LLM forward path:
  T_{1:j} ─→ Embedding ─→ E_{1:j} ─→ ... ─→ f_j ─→ LM_Head ─→ p_{j+1} ─→ t_{j+1}
                                            ↑
                                            └── EAGLE 说的 "feature":
                                                second-to-top hidden state
```

**关键洞察**：直接预测 token 命中率低（token 是从分布采样的，本质随机）；改在 feature 层做自回归——feature 是**确定的向量**。

#### Autoregression Head 架构（EAGLE-1）

```
Input:
  feature seq    F_{1:i} = (f_1, ..., f_i)         shape (bs, i, d)
  token seq (前移一位) T_{2:i+1}                    shape (bs, i)

Step 1: token → embedding (用 target 的 embedding 层, frozen)
        → E_{2:i+1}                                 shape (bs, i, d)

Step 2: concat feature 与 embedding on hidden dim
        → shape (bs, i, 2d)

Step 3: FC 层 (2d → d)                              # 唯一大参数
        → shape (bs, i, d)

Step 4: 单个 Transformer decoder layer
        → 预测 f̂_{i+1}                              shape (bs, 1, d)

Step 5: 用 target 的 LM Head (frozen)
        → p̂_{i+2} = Softmax(LM_Head(f̂_{i+1}))
        → 采样 t̂_{i+2}

Step 6: 把 (f̂_{i+1}, t̂_{i+2}) 追加到输入, 继续 autoregress
```

#### 可训练参数量

| Target LLM | Trainable params | 占比 |
|---|---:|---:|
| 7B | 0.24B | 3.4% |
| 13B | 0.37B | 2.8% |
| 33B | 0.56B | 1.7% |
| 70B | 0.99B | 1.4% |
| Mixtral 8×7B | 0.28B | — |

#### Tree Attention

```
不是线性 draft n 个 token, 而是 tree-structured (fanout k=4 at root):

              Root (last real token)
              /   |   |    \
        cand_1  c_2  c_3  cand_4      (top-4 by prob)
        / | \
    gc_1a gc_1b gc_1c                 (每个 c 也有 fanout)
        ...

Verify 阶段:
  target LLM 一次 forward + tree mask attention
  → 每 tree node 的接受概率
  → Multi-Round Speculative Sampling (Algorithm 1) 递归验证:
      · 若某候选被拒 → 用调整后分布 norm(max(0, p - p̂)) 试下一 sibling
      · 若 k 个 sibling 全拒 → 从调整后分布采样
  → 落到接受最深的一支

实测: depth m 的 tree 通过 m 次 draft forward 生成 >m tokens
     示例: 10-token tree 只需 3 次 forward
```

#### 训练 loss（EAGLE-1）

```
L_reg = SmoothL1(f_{i+1}, DraftModel(T_{2:i+1}, F_{1:i}))    # 特征回归

# 分类 loss 走 frozen LM Head, distillation-style:
p_{i+2}   = Softmax(LM_Head(f_{i+1}))         # target 真实 feature 过 head
p̂_{i+2}  = Softmax(LM_Head(f̂_{i+1}))         # draft 预测 feature 过 head
L_cls = CrossEntropy(p_{i+2}, p̂_{i+2})

L_total = L_reg + w_cls · L_cls,   w_cls = 0.1  # (回归 loss 数量级小 10 倍)
```

**训练数据增强**：训练时对输入 feature 加均匀噪声 U(−0.1, 0.1)，缓解 autoregression 推理阶段的误差累积。

**训练成本**：
- 7B/13B/33B：单节点 RTX 3090，1-2 天
- 70B：4× A100 40G，1-2 天
- 数据：ShareGPT ~68k 对话

#### 三代主要差异

| 版本 | 时间 | 与前代差异 | 论文数字 |
|---|---|---|---|
| **EAGLE-1** | 2024-01, arXiv 2401.15077 | 上述 baseline，固定 tree | vs vanilla 3.0x；vs Medusa 1.6x (13B) |
| **EAGLE-2** | 2024-06, EMNLP'24 | **动态 draft tree**：按上下文调整 depth/width，不更新 draft 参数 | vs vanilla ~3.5x |
| **EAGLE-3** | 2025-03, arXiv 2503.01840 | **抛弃 feature prediction**，改直接 token prediction + multi-layer feature fusion via "training-time test" | up to **6.5x** vs vanilla；SGLang batch=64 throughput 1.38x；1.4x vs EAGLE-2 |

**EAGLE-3 关键动机**：EAGLE-1/2 的 draft 命中率随训练数据增加**饱和**（论文观察）。原因是 feature-level 自回归限制了 draft 能学到的信息。EAGLE-3 让 draft 直接预测 token distribution，配合**多层 target feature 融合**（不只 second-to-top），突破饱和。

#### 工程支持

| 框架 | 版本 | 配置示例 |
|---|---|---|
| **vLLM** | ≥ 0.7 (Speculators v0.3.0) | 见下 |
| **SGLang** | ≥ v0.4 | SpecForge 训练工具链（2025-07 开源） |
| **TensorRT-LLM** | 主干 | Speculative Sampling API |
| **AWS Bedrock** | 2026-Q1 | P-EAGLE（parallel drafting） |

vLLM 配置：

```bash
vllm serve meta-llama/Llama-3-8B \
  --speculative-config '{
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "num_speculative_tokens": 5,
    "draft_tensor_parallel_size": 1
  }'
```

---

### 8.4 FlashInfer（LLM 推理 attention kernel 库）

MLSys'25 Best Paper (arXiv 2501.01005)。

#### 定位

FlashAttention 是**单个 kernel**（长上下文 attention 的 IO 优化）。LLM serving 涌现的 attention 变体已经十几种：paged / radix / sliding-window / MLA / cross / speculative tree / chunked prefill / logits-cap 等。手写维护成本不可控。

FlashInfer 的方案：**统一 kernel 库 + JIT 生成器**，类比 CUTLASS 之于 GEMM。

#### 机制 1：Block Sparse Row (BSR) 统一 KV 布局

```
BSR 表示 (Block Compressed Sparse Row):
  indptr  = [0, 2, 3, 5, ...]      # 每 row 的 block 起点
  indices = [0, 3, 1, 2, 4, ...]   # 每 block 的列索引
  data    = 顺序拼接的 dense KV blocks

Tile size (B_r, B_c):
  · 传统 FA2: 要求 (128, 128) 倍数
  · FlashInfer: 任意 (B_r, B_c), 包括 (16, 1) / (1, 16) 这种细粒度

Composable Formats (多请求共享 prefix):
  prefix 部分:  (3, 1) 大块 → 多 query 复用 shared memory
  独立后缀:     (1, 1) 小块 → 精细化
  实现: 只调整 indices/indptr 数组, 无数据搬移
```

一个 BSR 表达可覆盖：Paged Attention (vLLM)、Radix Tree (SGLang)、Tree Attention (speculative)、KV importance mask。

#### 机制 2：JIT 生成 attention kernel

Attention 表达式模板：

```
Output = f_epilogue( scan( f_logits( f_q(Q) · f_k(K) ) ) · f_v(V) )
```

用户可注入的 hook（inspired by FlexAttention, 扩展了 Q/K/V transform）：

| Functor | 用途 |
|---|---|
| `QueryTransform` / `KeyTransform` / `ValueTransform` | 融合 RoPE / normalization / MLA 投影 |
| `LogitsTransform` | soft-cap (Gemma-2 / Grok)、ALiBi |
| `LogitsMask` | causal、sliding window、custom mask |
| `OutputTransform` | 输出后处理 |
| Softmax on/off | 支持 FlashSigmoid 等无 softmax 变种 |

生成流程：

```
用户提交 CUDA 代码字符串
       ↓
nvrtc / torch.utils.cpp_extension JIT 编译
       ↓
注册为 custom op (PyTorch / DLPack)
```

#### 机制 3：Load-balanced Scheduler（灵感来自 Stream-K）

```
Algorithm 1:
  1. tile cost 定义: cost(l_q, l_kv) = α · l_q + β · l_kv
  2. KV chunk 上限 L_kv = 总工作量 / CTA_count
  3. 每 query tile 的 KV 按 L_kv 切成 chunks
  4. chunks 按 cost 降序排序
  5. 贪心分配: 最小堆维护各 CTA 累积 cost
              每次将最大 chunk 给累积 cost 最小的 CTA

Plan / Run 分离 (Inspector-Executor):
  plan()  在 CPU 侧, 每 step 执行 1 次, 跨所有 layer 复用
  run()   persistent kernel, grid size 固定 → CUDAGraph 兼容
```

**Attention 与 contraction（合并 split-KV partial output）融合进一个 persistent kernel**。

#### 支持的 kernel 类型

- Attention 变种：dense/sparse prefill、decode、append、shared-prefix、tree attention (speculative)、sliding window、custom mask、logits soft-cap、FlashSigmoid、MLA、fused RoPE+attention
- Contraction kernel：split-KV partial output 合并
- Sampling kernel：见 8.5

（GEMM **不在** FlashInfer 主库。）

#### 数据（论文 Table，vs Triton backend）

| 场景 | FlashInfer 相对 |
|---|---|
| Decode inter-token latency (ITL) | ↓ **29-69%** |
| Long-context latency | ↓ **28-30%** |
| Parallel generation | ↑ **13-17%** |

#### 工程支持

| 框架 | 版本 | 用法 |
|---|---|---|
| **vLLM** | ≥ 0.5 | `VLLM_ATTENTION_BACKEND=FLASHINFER` |
| **SGLang** | 主干 | 默认 backend |
| **MLC-Engine** | ≥ 2025 | 默认 |
| **NVIDIA** | 2025-11 | 官方发布优化过的 FlashInfer LLM serving kernels |
| **ROCm (AMD)** | 2025-10 | 移植版本 |

Python API（decode kernel 示例）：

```python
import flashinfer

# Paged KV cache 上的 decode attention
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
wrapper.plan(
    indptr, indices, last_page_len,
    num_qo_heads, num_kv_heads, head_dim,
    page_size=16, data_type=torch.float16
)
o = wrapper.run(q, paged_kv_cache)
```

---

### 8.5 Sampling 算子重构（FlashInfer sorting-free）

FlashInfer v0.2.3，2025-03 引入。

#### 传统 sort-based sampling

vLLM v0 / PyTorch 的 pipeline：

```python
def top_k_top_p_sampling(logits, top_k, top_p):
    sorted_logits, sorted_idx = logits.sort(descending=True)   # O(V log V)  ← 主开销
    sorted_logits[..., top_k:] = -float('inf')                 # top-k mask
    probs = softmax(sorted_logits)
    cum_probs = probs.cumsum(dim=-1)
    top_p_mask = cum_probs > top_p
    sorted_logits[top_p_mask] = -float('inf')                  # top-p mask
    sampled_sorted_idx = multinomial(softmax(sorted_logits))
    return sorted_idx.gather(-1, sampled_sorted_idx)           # scatter back
```

耗时随 vocab_size 增长：

| 模型 | V | Batch=64 sampling (H100) | 占 decode step |
|---|---:|---:|---:|
| Llama-2 | 32K | ~0.2 ms | ~5% |
| Llama-3 | 128K | ~0.8 ms | ~15% |
| Qwen-3 | 152K | ~1.0 ms | ~20% |

（decode 单步 ~5-15 ms 参考）

#### Dual Pivot Rejection Sampling

数学基础：

```
目标: 从截断分布 p_filtered / Z 采样
  (filter = top-k 或 top-p 掉的部分设为 0, Z 是归一化常数)

朴素 rejection: 采样后若 token 在 filtered 集就 reject 重来
  问题: 接受轮数无上界 → 尾延迟不可控
```

Dual Pivot 算法（简化伪代码）：

```
Init: low ← 0, high ← max(p_i)

loop:
    u ← uniform(0, 1)
    j ← inverse_transform_sample(u, valid_range=(low, ∞))
    p_j ← probability of token j

    pivot_1 ← p_j
    pivot_2 ← (pivot_1 + high) / 2

    if j is in filter_set (top-k / top-p):
        return j                        # accept
    else:
        # 用 pivot_1 或 pivot_2 收缩 (low, high)
        # 保证每轮至少减半
        (low, high) ← shrink_range(...)

复杂度: O(log(1/ε)) worst case, ε = 浮点最小可表示值
       → 尾延迟可预测
```

**Correctness**：论文形式化证明输出概率恰为 p_j / Z（与直接 filter + categorical 数学等价）。

**Kernel 实现**：单个 fused CUDA kernel，用 CUB primitive：
- `BlockReduce`（求 max、sum）
- `BlockScan`（cumsum）
- `AdjacentDifference`

**Early stopping**：累积概率超过随机数 u 就 exit，不用扫全 vocab。

#### 数据（论文，vLLM 1×H100）

- Sampling 时间 **↓ >50%**（across Llama-3 / Qwen / Mistral 三个模型）
- 单 kernel 融合 top-k / top-p / temperature，取代原来 4 步 pipeline

#### 工程支持

| 框架 | 状态 | API |
|---|---|---|
| **FlashInfer** | 主实现 | `flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, uniform_samples, top_k, top_p)` |
| **vLLM v1** | 默认（走 FlashInfer） | — |
| **SGLang** | 默认 | — |
| **MLC-LLM** | 默认 | — |

同一算法也用于 speculative decoding 的 chain / tree verification。

---

### 8.6 MLA（Multi-head Latent Attention）

DeepSeek-V2/V3 提出。KV cache 从多头张量压缩到 low-rank latent。

#### 原理

标准 MHA / GQA 的 KV cache：

```
KV_size_per_token = 2 · num_heads · head_dim · dtype_bytes

Llama-2 70B (GQA, num_kv_heads = 8, head_dim = 128, FP16):
  每 token = 2 × 8 × 128 × 2 = 4 KB
  32K 上下文 = 128 MB / request
```

MLA 引入 latent 维度 d_c << num_heads · head_dim：

```
标准 MHA:
  K = X · W_K              shape (seq, num_heads · head_dim)
  V = X · W_V              shape (seq, num_heads · head_dim)
  存: K, V

MLA:
  c^KV = X · W_DKV         shape (seq, d_c),  d_c 常取 512
  K, V 推理时按需展开:
     K = c^KV · W_UK       (推理时才算)
     V = c^KV · W_UV
  存: c^KV (低秩表示)

KV cache 占用对比 (DeepSeek-V2 vs Llama-2):
  Llama-2 (MHA):  2 · num_heads · head_dim = 4096 元素/token/layer
  DeepSeek-V2 (MLA): d_c = 512 元素/token/layer
  → 减 ~93% (论文 §3.2 报告 4x-93% depending on config)
```

**推理性能保持**：MLA 通过**矩阵吸收**技巧把 W_UK 融进 attention 的其他 matmul（Q · W_UK^T 提前算完存回 W_Q'），实际推理 kernel 与 GQA 相当。

#### 数据

| 模型 | KV cache per token (FP16) | 长上下文可行性 |
|---|---:|---|
| Llama-2 70B (MHA) | ~2.6 MB (基线) | 4K 已很吃力 |
| Llama-2 70B (GQA-8) | ~326 KB | 32K OK |
| DeepSeek-V2 (MLA) | ~70 KB | 128K 舒适 |

#### 工程支持

| 框架 | MLA 支持 |
|---|---|
| **vLLM** | ≥ 0.6，DeepSeek 模型自动走 MLA path |
| **SGLang** | 主干 |
| **TensorRT-LLM** | DeepSeek 分支 |
| **FlashInfer** | 原生 MLA decode kernel（block-sparse KV 表达） |

**限制**：MLA 是**模型架构级设计**，Llama / Qwen 等已有模型不能直接切换（需重训）。

---

### 8.7 RadixAttention（SGLang）

跨请求 prefix 复用，用 radix tree 组织 KV cache。

#### 场景

```
Request A: "你是助手\n用户: 什么是 Python\n助手: Python 是..."
Request B: "你是助手\n用户: 什么是 Java\n助手: Java 是..."

共享 prefix: "你是助手\n用户: 什么是 " (30 tokens)
  · 传统: 每请求独立 prefill 全 30 tokens (重复计算)
  · RadixAttention: 前 30 tokens 命中 → 直接引用已有 KV block
                    只需 prefill 差异后缀
```

#### 数据结构

```
KV cache 组织为 radix tree:

  root
   ├─ [tok₀ … tok₁₀]  → block_5
   │   ├─ [tok₁₁, tok₁₂]  → block_9
   │   │   └─ ...
   │   └─ [tok₁₁', tok₁₃']  → block_11
   └─ [tok₀', ...]  → block_3

匹配算法 (request 到达):
  1. token 序列按 KV block size (常 16 tokens/block) 分块
  2. 从 root 沿 tree 向下匹配, 找最长共享 prefix
  3. 命中的 block 直接引用其物理 KV
  4. 只 prefill 未命中的后缀

Eviction (LRU):
  block reference count 归零 → 可回收
  但保留一段时间以待未来请求命中
```

#### 数据（SGLang 论文）

| 场景 | 加速 |
|---|---|
| Multi-turn chat | **2-5x** |
| Few-shot prompting（相同 system prompt） | **2-4x** |
| Tree-of-Thought / Agent 场景 | **1.5-3x** |

#### 工程支持

| 框架 | 支持 |
|---|---|
| **SGLang** | 原生（首发） |
| **vLLM** | ≥ 0.6，`--enable-prefix-caching` |
| **TensorRT-LLM** | KV cache reuse |
| **LMDeploy** | ✅ |

---

### 8.8 FP4 与 Blackwell

B200 / GB200 引入 native FP4 TensorCore（Hopper 只有 FP8）。

#### 精度表示

```
FP4 主流两种编码:
  E2M1: 1 sign + 2 exponent + 1 mantissa
        表示 16 个值, 动态范围 ~[0.5, 6.0]
  E1M2: 1 sign + 1 exponent + 2 mantissa
        动态范围窄, 精度略高

Blackwell 硬件方案:
  E2M1 TensorCore
  + Micro-scaling (MXFP4, OCP 标准):
      每 32 元素共享一个 exponent (E8M0)
      提升有效动态范围, 缓解 FP4 表示能力不足
```

#### 数据

| Precision | Peak TensorCore (B200) | Llama-70B 显存占用 |
|---|---:|---:|
| FP16 | 2250 TFLOPS | 140 GB |
| FP8 (E4M3) | 4500 TFLOPS | 70 GB |
| **FP4 (E2M1 / MXFP4)** | **9000 TFLOPS** | **35 GB** |

**精度对齐**：FP4 单纯 PTQ 掉点 ~5-15%（论文报告，取决于任务），实用需 QAT 或结合 SmoothQuant / AWQ / GPTQ。

#### 工程支持

| 框架 | FP4 状态 |
|---|---|
| **TensorRT-LLM** | ≥ v0.14 alpha 支持 MXFP4 |
| **vLLM** | 主干实验（2026-Q1） |
| **NVIDIA NeMo** | 官方 FP4 QAT recipe |
| **llama.cpp** | Q4 系列（非严格 MXFP4，但概念接近） |

---

### 8.9 前沿技术落地 checklist

按改动范围排序：

```
仅换库调用 (0 架构改动):
  □ FlashInfer attention backend      (Part 8.4)
  □ FlashInfer sampling               (Part 8.5)
  □ Prefix caching                    (Part 8.7)

配置开关:
  □ Chunked prefill                   (Part 7.6)
  □ Continuous batching               (Part 7.4)
  □ 量化到 FP8 / INT4 (AWQ/GPTQ)      (Part 6)

架构改造:
  □ P/D 分离                          (Part 8.1)
  □ Speculative decoding:
     - DeepSeek 生态 → MTP            (Part 8.2)
     - 其他模型     → EAGLE-3         (Part 8.3)
  □ MLA (需换模型或重训)              (Part 8.6)

硬件:
  □ H100 → H200: 显存 80→141 GB, 带宽 3.35→4.8 TB/s
  □ H100 → B200: FP4 支持, 带宽 →8 TB/s   (Part 8.8)
```

---

## Part 9. 硬件视角

### 9.1 Roofline Model —— 判断计算 vs 内存瓶颈

```
Performance (TFLOPS)
     ↑
     │        ← Compute-bound region
     │        算力上限 (peak TFLOPS)
     │────────────────────────
     │       ╱
     │      ╱  ← 斜率 = memory bandwidth
     │     ╱
     │    ╱   ← Memory-bound region
     │   ╱
     │  ╱
     └───────────────────────→
        Arithmetic Intensity (FLOPs/Byte)

判断:
  Intensity 高 (Conv, large GEMM) → Compute-bound → 换更快 TensorCore
  Intensity 低 (LayerNorm, Softmax, LLM decode) → Memory-bound → 换更宽 HBM
```

**Voice_safety Transformer 的算术强度**：
- Attention Q@K: ~200 FLOPs/byte（compute-bound）
- LayerNorm: ~2 FLOPs/byte（memory-bound）
- FFN GEMM: ~300 FLOPs/byte（compute-bound）
- 整体：中等，混合 bound

### 9.2 Compute-bound vs Memory-bound 检测

```bash
# 用 nsys / Nsight profiling
nsys profile --stats=true python inference.py

# 看关键指标
tensor_core_utilization: >70% → compute-bound
tensor_core_utilization: <30% → memory-bound
memory_throughput / peak_bandwidth: >70% → memory-bound
```

**优化方向依此分岔**：
- Compute-bound → 升 TensorCore (FP16/FP8/INT8)、kernel fusion
- Memory-bound → 量化权重（INT4）、FlashAttention（省 IO）

### 9.3 常见 GPU 生产选型

| 场景 | 推荐 GPU | 理由 |
|---|---|---|
| **Encoder 推理 (BERT, ViT)** | A10 / L4 | 中等算力足够，成本低 |
| **Encoder 高吞吐** | A30 / A100 | 大 HBM，batch 大 |
| **小 LLM (7B) 推理** | L40S / A100 40G | 显存够放 + 带宽好 |
| **中 LLM (13-30B) 推理** | A100 80G / H100 | 显存 + 带宽 |
| **大 LLM (70B+) 推理** | H100 x 2-4 (NVLink) | 显存 + 高速互联 |
| **超大 LLM (400B+) 推理** | H100 x 8 / H200 / B200 | 多卡张量并行 |
| **Diffusion 推理** | L40S / A100 | 需要中大显存 |
| **Edge / 移动** | Jetson Orin | 端侧 |

---

## Part 10. 多卡并行推理

**核心问题**：一张卡装不下的模型，或者要更高吞吐。

### 10.1 三种并行策略

```
Data Parallelism (DP):
  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
  │model │  │model │  │model │  │model │  ← 模型完整复制到每卡
  │ full │  │ full │  │ full │  │ full │
  └──────┘  └──────┘  └──────┘  └──────┘
   batch1    batch2    batch3    batch4   ← 每卡跑不同 batch
   
  收益: 吞吐 x N
  代价: 每卡都得能放下整个模型
  用途: 小模型高吞吐

Tensor Parallelism (TP):
  ┌──────────────────────────────────┐
  │            Model (整体)          │
  ├──────┬──────┬──────┬─────────────┤
  │ shard│ shard│ shard│ shard       │  ← 每层参数切成 4 份
  │  1/4 │  2/4 │  3/4 │  4/4        │
  └──────┴──────┴──────┴─────────────┘
   GPU0   GPU1   GPU2   GPU3
   ← 每次 forward 需要 all-reduce 通信 (NVLink 关键)
   
  收益: 显存 / N, 延迟略降
  代价: 通信开销 (NVLink 强需要)
  用途: 大 LLM 单请求延迟

Pipeline Parallelism (PP):
  Layer 1-6:  GPU 0
  Layer 7-12: GPU 1
  Layer 13-18: GPU 2  
  Layer 19-24: GPU 3
  
  请求流水线: request-A 到 GPU 0 → GPU 1 → GPU 2 → GPU 3
             同时 request-B 已经在 GPU 0 (流水线并行)
             
  收益: 显存 / N
  代价: 首个请求延迟 x N, 流水线气泡
  用途: 极大模型 (单张卡装不下)
```

### 10.2 组合策略（3D Parallelism）

生产大模型（如 405B）通常混合：
```
TP=8 (单机 8 卡内)  x  PP=4 (跨机)  x  DP=2 (副本)  = 64 GPUs
```

### 10.3 通信关键

| 并行方式 | 通信量 | 硬件要求 |
|---|---|---|
| DP | AllReduce (仅训练) | PCIe 就够 |
| **TP** | **每层 AllReduce** | **NVLink 必需**（PCIe 会拖后腿） |
| PP | 层间 point-to-point | 慢通信可容忍 |
| EP (MoE) | AllToAll | NVLink / IB |

---

## Part 11. 性能分析实战

### 11.1 Profiling 工具矩阵

| 工具 | 出品方 | 适用 | 特色 |
|---|---|---|---|
| **nsys** (Nsight Systems) | NVIDIA | 系统级 timeline | 看 CPU/GPU 交互全景 |
| **nvprof / ncu** (Nsight Compute) | NVIDIA | Kernel 级细节 | 看每个 kernel 的 SM/Cache 利用 |
| **trtexec** | NVIDIA | TensorRT 专用 | 单个 engine 的 tuning 结果 |
| **PyTorch Profiler** | Meta | PyTorch 代码 | 分层耗时 |
| **py-spy** | Community | Python | 抓 Python 端 stack |
| **perf** | Linux | 通用 | 系统调用/CPU |
| **nvidia-smi dmon** | NVIDIA | GPU util 采样 | 生产监控友好 |

### 11.2 定位瓶颈的"四问法"

拿到一个慢的推理服务，按序问：

```
Q1: GPU util 高吗？
  → nvidia-smi dmon 看 sm 利用
  → <30% → 前端瓶颈 (framework/serving/network)
  → >70% → 后端瓶颈 (kernel/compute)

Q2: Memory bandwidth 用满了吗？
  → nsys / ncu 看 memory throughput / peak
  → >70% → memory-bound, 该量化 / 减模型大小
  → <30% → compute-bound, 该升 TensorCore

Q3: Kernel launch overhead 大吗？
  → 看 CPU stack 有多少时间在 launch kernel
  → 每个 kernel <100μs 但数量多 → 用 CUDA graph 融合

Q4: Kernel 本身效率高吗？
  → ncu 看 Tensor Core utilization / achieved occupancy
  → 差 → 换 kernel (FlashAttention 替 vanilla attn)
```

### 11.3 一个真实案例（voice_safety）

```
观察 (Part 13 详见):
  QPS 145, GPU util avg 37%, P50 657ms
  
Q1 GPU util: 37% <70% → 前端瓶颈
Q2 Memory: 未测但估算 <50% → 不是 memory bound
Q3 Launch overhead: 每 batch 24 requests, launch overhead 可控
Q4 Kernel 效率: batch=24 attention 计算/IO 混合

结论: 主要瓶颈在 host prep (flat memcpy 23MB) + PCIe H2D
优化方向: pinned memory + CUDA graph capture
```

---

## Part 12. 常见坑与踩雷

### 12.1 编译期坑

| 现象 | 原因 | 解 |
|---|---|---|
| TRT build 极慢（几十分钟） | Dynamic shape profile 太宽 + workspace 不够 | 缩窄 profile 范围 / 扩 workspace |
| "Could not find implementation for ForeignNode" | TRT 编译器 bug 或 workspace OOM | 先扩 workspace，仍失败查 TRT changelog |
| ONNX 导出精度不一致 | opset 版本不同 / operator behavior 差异 | pin opset 版本，跑 numeric 对比 |
| 从 PyTorch 导出的 ONNX 有冗余节点 | trace 时产生的 dead ops | 用 onnx-simplifier 预处理 |

### 12.2 运行时坑

| 现象 | 原因 | 解 |
|---|---|---|
| 首次推理巨慢（warmup） | Kernel JIT / TRT engine build / CUDA context init | 部署时预推理 3-5 次 warmup |
| Dynamic shape 输入 shape 变化时慢 | TRT engine 需要重新做 shape inference | 减少 shape 变化，用 padding 或 bucketing |
| INT8 精度掉点严重 | Calibration set 不代表实际分布 | 用真实生产流量做 calibration |
| 长跑内存增长 | Buffer 未复用 / Session 泄漏 | 用 Ort::MemoryInfo 明确 buffer 归属 |
| GPU util 100% 但 QPS 上不去 | Kernel efficiency 低 (occupancy 差) | ncu profiling，可能是 kernel 选错 |
| 多请求延迟不稳 | Cross-batching 攒批时机变动 | 固定 batch collect window 分析 |

### 12.3 部署坑

| 现象 | 原因 | 解 |
|---|---|---|
| Pod 起来但 is_tensorrt=false | ORT 版本没含 TRT provider | 检查 tarball 名（-gpu-cuda12-）|
| Engine 加载失败 "different device" | 硬件 SM 版本不匹配 | 编译时对应 GPU 上编 |
| CUDA driver / runtime 版本冲突 | Forward compat 未启用 | 装 cuda-compat 包 |
| Kernel launch 报 "out of memory" | 显存碎片 | 重启 pod / 用 memory pool |
| 编译期 CUDA OOM | Workspace 超出显存 | 缩小 workspace |

### 12.4 精度坑

| 现象 | 原因 | 解 |
|---|---|---|
| FP32 训练 → FP16 推理精度掉 | 权重范围超 FP16 表示 | 找 outlier 层保 FP32 |
| INT8 校准后精度掉 5%+ | Calibration set 太小 / 分布偏 | 增大 calibration set / 用 percentile |
| FP8 精度损失 | E4M3 vs E5M2 选错 | Weight 用 E4M3, gradient 用 E5M2 |
| 大 batch 精度略低 | Numerical drift | 对精度敏感层保 FP32 |

---

## Part 13. 面试高频问题清单

### 13.1 基础概念

**Q1: 为什么推理需要单独优化，不能直接用训练框架？**
> 训练框架优先灵活性（动态图、梯度）→ 推理场景静态图 + 精度可损失 + 大量 batching 优化空间。生产推理性能 vs eager 差 5-10x。

**Q2: ONNX 是什么？为什么需要它？**
> Open Neural Network Exchange，跨框架的模型 IR。作用：
> 1. PyTorch 训练的模型可以直接跑在 TF Serving / Caffe / 手机上
> 2. 统一到 ONNX 后，可以选不同 Execution Provider (CUDA/TRT/OpenVINO/CoreML)
> 3. 编译器（TRT/TVM）只需要支持 ONNX 就能覆盖所有框架

**Q3: TensorRT 是 Runtime 还是 Compiler？两者区别是什么？**
> Compiler。区别：
> - Runtime (ORT)：动态解释图，每层调 kernel，通用但慢
> - Compiler (TRT)：AOT 编译成硬件二进制，融合 + autotune 深度优化，快 5x 但硬件绑定

**Q4: TRT engine 能不能跨机器 / 跨 GPU 型号复用？**
> 不能跨 GPU compute capability (SM_XX) / TRT major version / CUDA major version。Minor 版本通常可兼容。

### 13.2 图优化

**Q5: 算子融合能带来什么收益？举个例子。**
> 减少显存 IO。Conv+BN+ReLU 不融合每层都要写显存再读，融合后中间结果保留在寄存器/共享内存。10x 内存 IO 减少通常意味着 30-50% 加速（因为大部分 layer 是 memory-bound）。

**Q6: 为什么 attention 的 Q/K/V projection 可以水平融合？**
> Q、K、V 三个 matmul 有共享的 input（同一个 x），可以 concat 权重矩阵变成一个大 GEMM，一次算出 [Q; K; V] 再 split。减少 kernel launch 从 3 次到 1 次。

**Q7: FlashAttention 相比 vanilla attention 快在哪？**
> Vanilla: attention matrix (N × N) 会写入 HBM 再读出，O(N²) HBM IO。
> FlashAttention: tile 分块 + softmax 数值稳定 + 中间结果只在 SRAM，O(N) HBM IO。
> 收益对长上下文（N=2048+）巨大。

**Q8: CUDA graph 是什么，什么时候用？**
> 录制一次 kernel launch 序列成一个 graph，运行时一次性 launch 整个 graph（而不是逐层）。
> 适用：kernel 数量多、每个 kernel 小（<100μs）、shape 固定的场景。
> 不适用：dynamic shape 场景（每种 shape 需要单独 capture）。

### 13.3 量化

**Q9: PTQ 和 QAT 区别？各自适用场景？**
> - PTQ：训练完直接量化，需 calibration set。掉点 0.5-2%，工作量小。
> - QAT：训练时模拟量化，fine-tuning。掉点 <0.5%，需重训。
> 首选 PTQ，精度不达标才 QAT。

**Q10: INT8 量化会掉多少精度？**
> Encoder 模型（CNN/BERT）：0.3-1% top-1。
> LLM：weight-only INT4 (GPTQ/AWQ) 通常掉 <1 point on MMLU。
> 全 INT8 (activation) 对 LLM 挑战大，SmoothQuant 用迁移技巧才能做到。

**Q11: FP8 相比 INT8 优势？**
> 有指数位，动态范围广，activation 不需要 calibration。H100 硬件加速。缺点：只有 Hopper+ 支持。

**Q12: LLM 为什么主流用 weight-only 而不是全 INT8？**
> LLM 是 memory-bound（decode 阶段全靠 HBM 带宽扫权重）。Weight-only INT4 把 70B 模型从 140GB 压到 35GB，直接提升带宽利用率 4x。而 activation 量化收益小、精度风险大。

### 13.4 LLM 特有

**Q13: KV cache 是什么？为什么它导致 LLM 长上下文这么费显存？**
> Attention 每步需要历史所有 tokens 的 K 和 V。缓存下来避免重算。
> 大小 = 2 × num_heads × head_dim × dtype × num_layers × seq_len。
> Llama-2 70B 32K 上下文 KV cache = 80GB，超过单张 H100 显存。

**Q14: PagedAttention 解决什么问题？原理？**
> 问题：变长 KV cache 按 max_len 预分配浪费 60%+ 显存。
> 原理：模仿 OS 虚拟内存分页，KV cache 分 pages（每 page 16 tokens），动态按需分配。
> 收益：显存利用率 96%+，并发请求数 4-5x。

**Q15: Continuous batching 和 static batching 区别？**
> Static: 一批请求等最长的完成才能处理下一批，GPU 利用率 <30%。
> Continuous: 请求可以在任意 step 加入/退出 batch，GPU 利用率 80%+，QPS 2-5x。
> vLLM/TGI/TRT-LLM 都支持。

**Q16: Speculative decoding 原理？为什么能加速？**
> 用小 draft model 猜 k 个 token，大 model 一次前向验证。命中率 70% 时，1 次大 model forward 生成 3-4 tokens。
> 本质：把 decode 从 memory-bound（每 token 扫全权重）变成部分 compute-bound（一次前向利用更多 GPU）。

**Q17: 为什么 encoder 模型不适合 vLLM？**
> vLLM 的核心优化（PagedAttention / continuous batching / speculative decoding）全部针对 decoder 的迭代过程。encoder 一次 forward 就完事，这些优化无用武之地。static compilation (TRT) 就是最优。

### 13.5 硬件/系统

**Q18: 什么是 memory-bound vs compute-bound？如何判断？**
> Compute-bound：算力打满，纯 GEMM 大 batch。看 Tensor Core util。
> Memory-bound：带宽打满，激活函数、LayerNorm、LLM decode。看 HBM throughput。
> 工具：ncu (Nsight Compute)，看 "Compute (SM) Throughput" 和 "Memory Throughput"。

**Q19: Tensor Cores 是什么？和 CUDA cores 区别？**
> Tensor Cores：矩阵乘专用加速单元，一次算一个 4×4 或 16×16 matmul。FP16/FP8 是 CUDA cores 的 8-16x 吞吐。
> CUDA Cores：通用并行计算单元，做逐元素运算和特殊 kernel。
> 推理场景 GEMM 全走 TensorCore，element-wise 走 CUDA cores。

**Q20: NVLink 和 PCIe 差别？什么场景需要 NVLink？**
> PCIe: 32-128 GB/s，标准 CPU-GPU 通道。
> NVLink: 900 GB/s (H100)，GPU-GPU 直连。
> **Tensor Parallelism 必须 NVLink**（每 layer AllReduce），PCIe 会成为瓶颈。
> DP / PP 用 PCIe 也 OK。

### 13.6 系统设计

**Q21: 设计一个 LLM 推理服务，用哪个框架？**
> - 追求极致 QPS + NVIDIA GPU 稳定 → TensorRT-LLM
> - 灵活性 + HF 生态 → vLLM 或 TGI
> - 前沿功能 (RadixAttention) → SGLang
> - 多卡大模型 → DeepSpeed-Inference / TRT-LLM

**Q22: 单卡 QPS 满足不了业务怎么办？**
> 优先横向扩容（N 副本 × 单卡 QPS）。工程收益线性，无技术风险。
> 深度优化（量化、蒸馏）只有在扩容不经济时才做（比如推理占 10% 的成本时不划算，占 50%+ 时值得做）。

**Q23: 生产上 latency 波动大，怎么定位？**
> - GC / JIT 暂停：warmup 后 profile
> - Batch 攒批不稳：固定 batch collect window
> - Kernel launch queue 堆积：CUDA graph capture
> - CPU 干扰：pin CPU 核，减少上下文切换

**Q24: TRT engine cache 应该放哪里？**
> - Pod 本地 /tmp（首启慢，重启复用，简单）
> - PVC (persistent volume claim)（跨 pod 复用）
> - 上传到 blobstore + 启动时下载（工业化，可跨 CI/CD）
> 我们 voice_safety 走的是本地 /tmp 方案。

### 13.7 实战场景

**Q25: 你的 voice_safety 项目最难的技术挑战是什么？**
> TRT 8.6 的 dynamic shape + Transformer LayerNorm 融合编译器 bug（Issue #3490）。
> 报错 "Could not find implementation for ForeignNode"，一度误判为 workspace OOM，扩到 8GB 后仍 crash。
> 深挖 NVIDIA issue 后发现是 TRT 编译器已知 bug（zerollzeng 官方确认），只有 TRT 10 修复。
> 最终升级 base image 到 nvcr.io/nvidia/tensorrt:24.12-py3 (TRT 10.7)，2:31 build 成功，20s 音频原生 dynamic 推理。

**Q26: 单卡 QPS 达到 145 后遇到瓶颈，怎么分析的？**
> 观察：GPU util 只 37% avg，加并发 QPS 不涨（继续加只涨 latency）。
> 分析：batch=24 flush 时间 165ms，其中 GPU 计算只占 60%，剩下是 host prep (23MB memcpy) + PCIe H2D + D2H。
> 结论：架构天花板到了，突破点在 IO 优化（pinned memory、CUDA graph）或量化。

---

## Part 14. 实战案例：voice_safety 全流程

### 14.1 项目背景

- **业务**：短视频音频娇喘 / 性内容检测
- **模型**：Roblox voice-safety-classifier-v3
- **架构**：24-layer Transformer, ~305M params
- **输入**：16kHz mono PCM，2-30s 音频
- **输出**：sexual_score float (二分类)
- **目标 QPS**：单卡 100+，业务 QPS 由副本数 × 单卡 QPS 决定

### 14.2 P2.5 阶段 7 轮迭代（浓缩版）

| Round | 关键动作 | 结果 |
|---|---|---|
| 1 | ORT 1.13.1 → 1.16.3 + base image 升级 | is_tensorrt=false，dlopen libcudart.so.11.0 失败 |
| 2 | ORT 1.16.3 → 1.17.1 -cuda12- 裸包 | is_cuda=true 但 trt_err="TensorRT provider is not enabled" |
| 3 | 1.17.1 → 1.17.3 **-gpu-cuda12-** 完整包 | is_tensorrt=true ✅，15s 音频 GT 精度对齐 |
| 4 | 尝试 profile T dynamic [32000, 480000] | TRT build 报 Error 10 "Could not find implementation" |
| 5 | Workaround: static T=240000 + engine.go zero-pad | 15s 音频跑通，但 20s 音频兜底、短音频精度稀释风险 |
| 6 | 扩 workspace 1GB → 8GB 试恢复 dynamic | 依然 crash 7 次，证伪 workspace OOM 假设 |
| 7 | 深挖发现 Issue #3490，升 base image 到 24.12-py3 (TRT 10.7) | ✅ 2:31 build 成功，dynamic T 原生跑通，20s 音频推理成功 |

### 14.3 单卡 145 QPS 的构成拆解

```
业务请求 10s 音频, 单卡稳态 QPS = 145
每请求端到端 P50 = 657ms (含排队 + 处理)

分层拆解:
┌──────────────────────────────────────────────────┐
│ 客户端 gRPC 序列化                    ~5ms       │
│ 网络 (pod 内) + phoenix 框架接收       ~5ms       │
│ audio decode (WAV → PCM)               ~15ms      │
│ engine.Predict 排队 (sem chan)          ~450ms    │  ← 主要延迟
│ batch_coordinator 攒批 (12ms window)    ~12ms     │
│ flat memcpy 23MB                        ~5ms      │
│ H2D transfer via PCIe                   ~10ms     │
│ TRT batch=24 inference                  ~120ms    │  ← GPU 时间
│ D2H transfer                            ~1ms      │
│ Response 打包 + 返回                    ~5ms      │
├──────────────────────────────────────────────────┤
│ Total                                  ~630ms     │
└──────────────────────────────────────────────────┘

主要时间在 sem chan 排队 (450ms):
  in-flight requests = QPS × latency = 145 × 0.657 = 95
  max_concurrency = 256, GPU 上限 batch=32
  等 batch 空闲 slot 平均要等 ~450ms
```

### 14.4 下一步优化路径（按 ROI）

见 [Part 12 优化空间分析](#part-12-常见坑与踩雷) 相关章节。

---

## 附录 A. 术语表

| 术语 | 全称 | 含义 |
|---|---|---|
| **AOT** | Ahead-of-Time compilation | 预编译（vs JIT） |
| **JIT** | Just-in-Time compilation | 即时编译 |
| **EP** | Execution Provider | ONNX Runtime 的后端插件 |
| **HBM** | High Bandwidth Memory | GPU 显存类型（HBM2e/HBM3） |
| **IR** | Intermediate Representation | 中间表示（ONNX/HLO/MLIR） |
| **KV cache** | Key-Value cache | LLM decode 复用的历史 K/V |
| **MHA** | Multi-Head Attention | 多头注意力 |
| **MoE** | Mixture of Experts | 混合专家模型（Mixtral/DeepSeek） |
| **NVLink** | NVIDIA GPU 高速互联 | 900 GB/s（H100） |
| **PTQ** | Post-Training Quantization | 训练后量化 |
| **QAT** | Quantization-Aware Training | 量化感知训练 |
| **QPS** | Queries per Second | 请求每秒 |
| **SM** | Streaming Multiprocessor | GPU 计算单元 |
| **SASS** | Streaming ASSembler | NVIDIA GPU 汇编 |
| **PTX** | Parallel Thread eXecution | NVIDIA 虚拟 ISA |
| **TFLOPS** | Tera FLoating-point OPerations per Second | 万亿次浮点运算/秒 |
| **TP / PP / DP / EP** | Tensor/Pipeline/Data/Expert Parallelism | 4 种并行策略 |
| **CGO** | C-Go binding | Go 调 C 库 |
| **ABI** | Application Binary Interface | 二进制接口 |
| **cuBLAS** | CUDA Basic Linear Algebra Subprograms | NVIDIA 线性代数库 |
| **cuDNN** | CUDA Deep Neural Network library | NVIDIA 深度学习原语库 |
| **CUTLASS** | CUDA Templates for Linear Algebra Subroutines | NVIDIA 可定制 kernel 模板库 |

---

## 附录 B. 参考资料

### 官方文档
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [PyTorch 2.0 & torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)

### 关键论文
- **FlashAttention** (Tri Dao, 2022) - 长上下文 attention 优化
- **PagedAttention / vLLM** (UC Berkeley, 2023) - LLM serving 革命
- **GPTQ / AWQ** - Weight-only 量化算法
- **SmoothQuant** (Song Han, 2022) - LLM 激活量化
- **Speculative Decoding** (Google, 2022) - 投机解码
- **Roofline Model** (Berkeley, 2008) - 性能瓶颈分析框架

### 深度教程
- [Efficient Deep Learning Book](https://efficientdlbook.com/) - MIT/Song Han
- [NVIDIA GTC Talks](https://www.nvidia.com/gtc/) - 每年最前沿 
- [Character.AI Blog](https://research.character.ai/) - LLM 推理优化实战

### 开源代码值得读
- **vLLM**: `vllm/attention/backends/flash_attn.py` - Paged Attention 实现
- **FlashAttention**: `flash_attn/flash_attn_interface.py` - 手写 CUDA
- **TensorRT-LLM**: `tensorrt_llm/models/` - LLM 优化模板
- **llama.cpp**: 极致 CPU 优化 + 量化

### 面试相关
- 大厂 ML infra 面试：[LeetCode ML System Design](https://leetcode.com/)
- 系统设计：[Alex Xu - ML System Design](https://bytebytego.com/)

---

## 学习路径建议

**入门（1-2 周）**：
1. PyTorch → ONNX 导出实操
2. ONNX Runtime 跑推理，理解 EP 概念
3. TensorRT 编译一个简单模型（如 ResNet）

**进阶（1-2 月）**：
4. 精读 TRT 官方 Developer Guide，理解图优化每一层
5. 用 nsys / ncu profile 一个真实模型，找出瓶颈
6. 实操 INT8 量化（TRT/ORT），做精度回归
7. 读 FlashAttention paper + 源码

**LLM 方向**：
8. 部署 Llama-2-7B 用 vLLM，理解 continuous batching
9. 读 PagedAttention paper，理解显存管理
10. 试用 TensorRT-LLM，对比 vLLM 性能
11. GPTQ / AWQ 量化 Llama，做精度对比

**深度方向**：
12. TVM 或 MLIR 学习一个编译器 IR
13. 手写 CUDA kernel（矩阵乘 / attention）
14. 读 CUTLASS 源码

---

**文档版本**: v1.2 (2026-07-15)
**变更**:
- v1.2: Part 8 前沿探索重写，改为「原理（图/伪代码/公式）→ 数据 → 工程支持」结构，补充 DistServe M/D/1 公式、EAGLE Autoregression Head 参数量、FlashInfer BSR / JIT / Scheduler 三大机制、Dual Pivot Rejection Sampling 伪代码、MLA 数学表达等技术细节
- v1.1: 新增 Part 8 前沿探索（2025-2026）
**作者**: chenglitao (based on voice_safety P2.5 project)
**License**: 内部资料，欢迎完善

> 有问题或补充，请提交 PR 或联系作者。祝面试顺利 🚀
