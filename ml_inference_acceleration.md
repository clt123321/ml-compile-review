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

> **文档时间戳**：2026-07-15 快照。本章追踪 2024 年以来 LLM 推理领域最重要的**架构级 / 算法级 / 算子级**演进，与 Part 7 的成熟优化互补。
>
> **阅读姿势**：这些技术**都还在快速迭代**，具体数字（加速倍数、接受率）以论文和框架 release notes 为准，本章给出的是 2026-07 之前的公开数据。生产选型时**优先看框架是否已集成**（vLLM / SGLang / TRT-LLM），而非纸面数字。

### 8.1 全景：LLM 推理这一年半发生了什么

以 2023 年底 vLLM 的 PagedAttention 为分水岭，LLM 推理从"单机单卡优化"跨入**系统架构级重构**阶段。过去 18 个月三条主线：

```
主线 1: 架构级 —— P/D 分离 (Disaggregated Serving)
  DistServe (2024-01, OSDI'24)
    → Splitwise (2024-04, Microsoft ISCA'24)
    → Mooncake (2024-06, Moonshot Kimi 生产)
    → vLLM / SGLang / TRT-LLM / LMDeploy 全面支持 (2025)

主线 2: 算法级 —— 推测解码进化 (外挂 draft → 模型内生)
  EAGLE-1 (2024-01, feature-level speculative)
    → EAGLE-2 (2024-06, 动态 draft tree)
    → MTP (2024-12, DeepSeek-V3 模型自带 MTP heads)
    → EAGLE-3 (2025-03, 直接 token 预测 + multi-layer fusion)

主线 3: 算子级 —— Kernel 库范式
  FlashAttention 3 (2024-07, H100 FP8 支持)
    → FlashInfer (2025-01, MLSys'25 Best Paper, block-sparse + JIT)
    → Sampling 算子重构 (2025-03, sorting-free)
```

**核心观察**：Part 7 讲的是"用现有工具优化"，Part 8 讲的是"业界重新定义了工具本身"。

**技术选型转折点**：
- 2023：单机 vLLM 就够用
- 2025-2026：**生产大模型必须考虑 P/D 分离 + 推测解码 + FlashInfer kernel**，否则单卡吞吐落后同行 3-5x

### 8.2 P/D 分离（Prefill/Decode Disaggregation）

#### 8.2.1 问题背景

Part 7.1 讲过 LLM 推理分 prefill (compute-bound) 和 decode (memory-bound) 两阶段。**传统方案两阶段跑在同一张 GPU 上**，产生两个问题：

```
问题 1: Prefill / Decode 互相干扰
  当 GPU 正在跑长 prompt 的 prefill (占满 SM):
     所有 decode 请求被阻塞
     P99 latency 抖动严重

问题 2: 资源配比无法独立优化
  Prefill 需要: 高 FLOPS (compute-bound)
  Decode 需要: 高 HBM 带宽 (memory-bound)
  同一张卡兼顾, 两者都不最优
```

#### 8.2.2 核心思想

**用不同 GPU 集群分别处理 prefill 和 decode**，通过网络传输 KV cache 衔接两阶段。

```
传统 collocated:                    P/D 分离:
┌────────────────┐                  ┌──────────┐  ┌──────────┐
│   GPU 0        │                  │  Prefill │  │  Decode  │
│ ┌────────────┐ │                  │  Pool    │──│  Pool    │
│ │ prefill req│ │                  │          │KV│          │
│ │ decode req │ │                  │  A100    │→ │  H100    │
│ └────────────┘ │                  │  (计算)  │  │  (带宽)  │
└────────────────┘                  └──────────┘  └──────────┘
    互相干扰                           独立扩缩容 + 硬件异构可行
```

#### 8.2.3 三代代表系统对比

| 系统 | 时间 | 出品 | 关键创新 | 数字 |
|---|---|---|---|---|
| **DistServe** | 2024-01<br>OSDI'24 | 北大 + UCSD | 学术首发，两阶段独立 TP/PP 策略 + 拓扑感知放置 | Goodput **7.4x**<br>SLO 可紧缩 **12.6x** |
| **Splitwise** | 2024-04<br>ISCA'24 | Microsoft | 异构 GPU（prefill 高算力 + decode 低成本） | 同资源吞吐 **2.35x**<br>或 20% 成本降 **1.4x** |
| **Mooncake** | 2024-06<br>arXiv | Moonshot AI<br>（Kimi 生产） | **KV cache 中心化架构**：CPU DRAM + SSD 组池化 + 早期拒绝 | 长上下文吞吐 **+525%**<br>多处理 75% 请求 |

**Mooncake 是当前生产标杆**（2025-2026 Kimi 亿级 DAU 在线跑），其核心创新是把 KV cache 提升为**一级公民**——不再是 GPU 内的临时 buffer，而是可以跨 GPU / CPU DRAM / SSD 分层存储、跨请求复用的持久化资源。

#### 8.2.4 KV Cache 传输是关键工程点

P/D 分离后，KV cache 必须从 prefill 节点跨 GPU 传到 decode 节点，这是新的瓶颈来源：

```
数量级估算 (Llama-70B, 4K prompt):
  KV cache size = 4096 × 2.6 MB = 10 GB

NVLink 900 GB/s:  10 GB / 900 = ~11 ms   (同机内, OK)
IB 400 Gbps:      10 GB / 50 = ~200 ms   (跨机, 灾难)
以太网 25 Gbps:   10 GB / 3.1 = ~3.2 s   (完全不可用)

→ P/D 分离几乎必须 NVLink / IB, 或就近同机部署 prefill+decode 节点
→ Mooncake 的 KV cache 池化实际上是"预加载到 SSD / DRAM 减少重复传输"
```

#### 8.2.5 生产落地状态（2026-07 快照）

| 框架 | P/D 分离支持 | 备注 |
|---|---|---|
| **vLLM** | ✅ (NixlConnector, 2025) | 生产可用 |
| **SGLang** | ✅ | 深度整合 |
| **TensorRT-LLM** | ✅ (Dynamo, NVIDIA 官方) | NVIDIA 主推 |
| **LMDeploy** | ✅ v0.12 (DLSlime, Mooncake protocol) | 商汤 |
| **DeepSeek 开源栈** | ✅ (自研) | DeepSeek-V3 生产 |

**结论**：P/D 分离已从"论文技术"跨过"框架实验"进入"生产标配"，2026 年做 LLM serving 不上 P/D 就是落后。

### 8.3 MTP（Multi-Token Prediction）—— 模型自带的推测解码

#### 8.3.1 问题背景

Part 7.5 的 speculative decoding 依赖**外挂的 draft model**（额外训一个小模型）。三个痛点：
1. 需要额外训练 + 部署 draft model
2. Draft 与 target 词汇 / 风格漂移，命中率不稳
3. 长上下文时 draft 也吃显存

**MTP 的解法**：让 target 模型**自己**输出多个未来 token 的预测——训练时 densify 监督信号，推理时直接白嫖 speculative decoding。

#### 8.3.2 时间线：两条线交汇

```
2024-04: Meta "Better & Faster LLMs via Multi-token Prediction" (Gloeckle et al.)
         → D 个独立 heads 并行预测 next-1, next-2, ..., next-D
         → 训练数据效率提升 + 推理可用于 speculative
         → 未在大规模生产模型使用

2024-12: DeepSeek-V3 Technical Report
         → 改进版 MTP: D 个 sequential 模块 (非独立并行)
         → 保完整因果链: 每个 depth k 用前面 k-1 的表征
         → 生产开源, 掀起 MTP 复兴

2025+:   SGLang / vLLM / TRT-LLM 陆续加 MTP 支持
```

#### 8.3.3 DeepSeek-V3 MTP 架构（重点）

```
输入序列: [t_1, t_2, t_3, ..., t_n]
主模型输出: h_i (第 i 位置的最终 hidden state)

MTP Module k (k=1..D, D 通常 = 4):
  ┌────────────────────────────────────────────────────┐
  │ 输入 1: 前一 depth 的表征 h_i^(k-1) [已 RMSNorm]    │
  │ 输入 2: 未来第 k 个 token 的 embedding [已 RMSNorm] │
  │        (embedding 层与主模型共享)                   │
  │                                                    │
  │   concat → 线性投影 M_k (d × 2d) → Transformer 块  │
  │   → 输出 h_i^(k)                                   │
  │   → 通过共享的 output head 得到 t_{i+k} 的预测     │
  └────────────────────────────────────────────────────┘

关键:
  1. embedding 和 output head 全部共享 (省参数)
  2. 深度间是 sequential 的 (保因果链)
  3. 与 Meta 版本 (parallel heads) 不同, 生成质量更好
```

#### 8.3.4 训练收益

| 收益 | 说明 |
|---|---|
| **监督信号 densify** | 每个位置贡献 D 个 loss，训练数据效率提升 |
| **表征质量提升** | 强制模型 pre-plan 未来 tokens 的表征 |
| **无需推理开销** | 训练用，推理阶段 MTP 模块可选 discard |

#### 8.3.5 推理收益（作为 speculative decoder）

DeepSeek-V3 官方数据：
- **第二 token 接受率**：**85%-90%**（跨话题稳定）
- **端到端 TPS 提升**：**1.8x**（Tokens Per Second）

vLLM 社区实测（DeepSeek-R1，k=1）：
- 接受率 81-82.3%
- QPS=1 时 **1.63x** 加速
- 高 QPS (>8) 下加速衰减（batch 摊薄了 speculative 的 GPU 空闲时间）

SGLang 官方博客（2025-07）：
- MTP 端到端 **+60% output throughput**（DeepSeek V3，无质量损失）

#### 8.3.6 MTP vs EAGLE：如何选

| 维度 | MTP | EAGLE-3 |
|---|---|---|
| **训练** | 与主模型联合训练 | Draft 独立训练 |
| **权重** | MTP heads 是模型自带参数（DeepSeek-V3 权重发布已含） | 需单独训 draft 权重 |
| **接受率** | 85-90% (DeepSeek-V3 报告) | 类似或略高（EAGLE-3 报告 up to 6.5x） |
| **通用性** | 只能用在**训练时加过 MTP** 的模型 | 可为**任何**主模型训 draft |
| **生产成熟度** | DeepSeek 生态原生，其他模型需重训 | SGLang / vLLM 集成，多模型通用 |

**选型建议**：跑 DeepSeek 模型时 **MTP 无脑用**（权重免费送）；跑 Llama / Qwen 等其他模型时 **EAGLE-3 训 draft**。

### 8.4 EAGLE 系列演进（通用推测解码 SOTA）

EAGLE 是当前**通用推测解码 SOTA**（2026），三代演进逻辑清晰：

#### 8.4.1 三代对比

| 版本 | 时间 | 论文 | 核心思路 | 加速比 |
|---|---|---|---|---|
| **EAGLE-1** | 2024-01 | arXiv 2401.15077 | Draft 在 **feature level**（second-to-top-layer）自回归 + tree attention | 3x vs vanilla<br>1.6x vs Medusa (13B) |
| **EAGLE-2** | 2024-06 | ICML'24 | 动态 draft tree（depth / width 按上下文调整） | 3.5x vs vanilla |
| **EAGLE-3** | 2025-03 | arXiv 2503.01840 | **抛弃 feature prediction, 直接 token prediction** + multi-layer feature fusion via "training-time test" | **6.5x** peak<br>1.4x vs EAGLE-2 |

#### 8.4.2 每代关键突破的直观理解

```
EAGLE-1 vs Medusa:
   Medusa 用 K 个独立 heads 并行猜, 无因果依赖 → 命中率 ~50-65%
   EAGLE-1 在 feature 层做自回归猜 → 命中率 ~70-80%
   多几个点意味着 "接受长度" 从 2.5 涨到 4.0, 直接 1.5x

EAGLE-2:
   静态 draft tree 有的分支永远不用 (context 无关的浪费)
   动态调整 tree 结构 → 高价值分支用更多 budget
   +15% acceptance length

EAGLE-3:
   意外发现: EAGLE-1/2 的 feature prediction 有 "训练数据饱和" 问题
   → 抛弃 feature, 让 draft 直接预测 token
   + 用多层 (不只 second-to-top) feature 融合
   + "training-time test" 缩小 training / inference gap
   → 数据 scaling 恢复, 6.5x 天花板
```

#### 8.4.3 生产落地

- **SGLang**：EAGLE-3 官方合作，SpecForge 训练框架 2025-07 开源
- **vLLM**：Speculators v0.3.0 (2025-12) 官方将 EAGLE-3 列为当前 SOTA
- **AWS**：P-EAGLE（Parallel EAGLE）优化版本，AWS Blog 2026
- **TensorRT-LLM**：Speculative Sampling 章节支持 EAGLE

**结论**：非 DeepSeek 生态（Llama、Qwen、Mistral 等）想上推测解码，**EAGLE-3 是默认选择**。

### 8.5 FlashInfer：Attention Kernel 库范式（MLSys'25 Best Paper）

#### 8.5.1 定位

FlashAttention（Tri Dao, 2022）解决了长上下文 attention 的 IO 瓶颈。但**FlashAttention 只是一个 kernel**。到 2024-2025，LLM serving 涌现出海量 attention 变体：

```
仅 attention 就有 ~10 种变体:
  · Prefill attention (长 sequence, batch=1)
  · Decode attention (短 query, KV cache 长)
  · Paged attention (KV cache 分页, vLLM 用)
  · Radix attention (prefix cache, SGLang 用)
  · MLA - Multi-head Latent Attention (DeepSeek 用)
  · Grouped Query Attention (Llama-2 70B 用)
  · Sliding window (Mistral 用)
  · Cross attention (encoder-decoder)
  · Speculative verify attention (tree structure)
  · Chunked prefill attention

每种都要手写高性能 CUDA kernel? → 组合爆炸, 维护地狱
```

**FlashInfer 的解**：一个**统一的 kernel 库 + 生成器**，用 JIT 编译按需生成 kernel。类比 CUTLASS 之于 GEMM，FlashInfer 之于 attention。

#### 8.5.2 三大核心技术

```
1. Block-sparse KV Cache 布局
   传统: KV cache 密集连续 → 稀疏场景 (如 sliding window) 浪费带宽
   FlashInfer: block-sparse + composable formats
   → 一份代码支持多种 KV 组织方式 (dense/paged/radix/tree)

2. JIT-compiled Attention Template
   用户提供 attention variant 的参数 (mask, scaling, softmax variant...)
   FlashInfer JIT 生成 fused CUDA kernel
   → 加新变体不用改库, 用户空间描述即可

3. Load-Balanced Scheduler
   不同请求 seq_len 差异大 → 负载不均, GPU SM 空闲
   FlashInfer 动态调度确保每个 SM 满载
   + 保持 CUDAGraph 兼容 (静态签名)
```

#### 8.5.3 关键数字（vs 主流 baseline）

| 场景 | FlashInfer 相对 SOTA baseline |
|---|---|
| **Inter-token latency** (decode) | **↓ 29-69%** |
| **Long-context latency** | ↓ 28-30% |
| **Parallel generation** | ↑ 13-17% |

#### 8.5.4 生产整合（2026-07 快照）

- **vLLM**：默认 attention backend 之一
- **SGLang**：核心依赖
- **MLC-Engine**：整合
- **NVIDIA**：2025-11 官方发布优化过的 FlashInfer LLM serving kernels
- **ROCm**：2025-10 AMD 移植版本（跨硬件）
- **MLSys 2026**：NVIDIA 主办 FlashInfer AI Kernel Generation Contest

**结论**：**FlashInfer 已成为 LLM 推理 attention kernel 的事实标准**，直接用第三方 kernel（cuDNN MHA / xFormers）已经过时。做 LLM serving 时看框架有没有集成 FlashInfer，是判断技术栈新鲜度的重要指标。

### 8.6 Sampling 算子重构（2025-03 FlashInfer Sorting-Free）

#### 8.6.1 被忽视的瓶颈

大部分人认为 sampling（从 logits 选下一个 token）是"最便宜的一步"。**错**——现代 LLM 词汇表 vocab_size 涨到 128K+（Llama-3 128K, Qwen 152K），传统 sampling 已是 decode 阶段的显著 overhead。

```
传统 PyTorch / vLLM v0 sampling:
  1. logits.sort()               O(V log V), V=vocab_size
  2. gather + mask (top-k)       O(V)
  3. softmax + cumsum + mask (top-p)  O(V)
  4. scatter to invert sort      O(V)
  → 4+ 次 kernel launch, 显存带宽被 sort 吃掉

V=128K, batch=64 时:
  sampling 占 decode step 20-30% 时间 (!!)
```

#### 8.6.2 核心创新：Dual Pivot Rejection Sampling

FlashInfer 2025-03 blog "Sorting-Free GPU Kernels for LLM Sampling"：

```
朴素 rejection sampling 的问题:
  接受轮数无上界 → 尾延迟不可控

Dual Pivot Rejection Sampling (FlashInfer v0.2.3):
  每轮用 2 个 pivot 判断:
    pivot_1 = 当前 sampled 概率
    pivot_2 = (pivot_1 + high) / 2
  三种情况:
    - 接受当前 sample
    - 用 pivot_1 收缩范围
    - 用 pivot_2 收缩范围
  → 每轮范围至少减半 → O(log(1/ε)) worst case
  → 尾延迟可预测, 生产稳定
```

#### 8.6.3 数字（vLLM 1×H100）

- **Sampling 时间 ↓ >50%** across three tested models
- 单 kernel 融合 top-k / top-p / temperature，取代原 4 步 pipeline

#### 8.6.4 落地

- **FlashInfer** 内建，可直接调用 `flashinfer.sampling.top_k_top_p_sampling_from_probs`
- **vLLM v1** 已集成
- **SGLang / MLC-LLM** 集成

**结论**：这是过去两年"最容易被 overlook 但收益最直接"的优化之一——只需换库调用，不动模型，decode QPS 立涨。

### 8.7 其他值得关注的方向（2026-07 速览）

以下技术每个都可以独立成章，本节仅提供**是什么、为什么重要、代表实现**，深挖留作后续独立文档。

#### 8.7.1 MLA（Multi-head Latent Attention）—— DeepSeek V2/V3 首创

**问题**：GQA 只是把 KV heads 数量减少（多 Q 共享 KV），但 head_dim 还是原尺寸。KV cache 大小仍然线性于 num_heads。

**MLA 思路**：把 KV cache 压缩到一个 low-rank latent 空间（rank << head_dim），在计算 attention 时才展开。

**收益**：DeepSeek-V2 KV cache 相比 Llama-2 70B 小 **93%**，长上下文可行性大幅提升。

**代价**：需要 MLA-aware 的 kernel（FlashInfer / vLLM 都已支持）。

#### 8.7.2 RadixAttention（SGLang）—— Prefix Cache 跨请求共享

**问题**：多轮对话、few-shot prompting、code completion 场景下，**同一段 prefix 被反复计算**。

**解**：用 Radix tree 管理所有请求的 KV cache 前缀，命中就复用。SGLang 首创（2024）。

**数字**：多轮对话场景 **2-5x** 吞吐提升。

**生产**：SGLang 核心，vLLM 也加了 prefix caching。

#### 8.7.3 Chunked Prefill —— 从技术到标配

Part 7.6 提过。**已成 2026 生产标配**（vLLM / SGLang / TRT-LLM 默认开启）。价值在于 P/D 分离前的过渡方案 + P/D 分离场景下调度 prefill chunk。

#### 8.7.4 FP4 / Blackwell —— 硬件新精度维度

**H100 引入 FP8**（Hopper, 2022），**B200 引入 FP4**（Blackwell, 2024-2025）。FP4 可让 Llama-70B 塞进 40GB 显存（原 140GB FP16），推理成本再降 4x。

**软件生态**：TensorRT-LLM / vLLM / SGLang 正在完善 FP4 支持（2026 全年主线）。

**注意**：FP4 精度损失比 FP8 大，需 QAT 或 SmoothQuant / AWQ 等激活迁移技术协助。

#### 8.7.5 长上下文优化 —— 从 4K 到 1M

| 技术 | 核心思路 | 时间 |
|---|---|---|
| **YaRN** | RoPE 频率插值扩展 | 2023 |
| **StreamingLLM** | Attention sink + sliding window，无限流式 | 2023-09 |
| **RingAttention** | 跨 GPU 分块 attention | 2023-10 |
| **Landmark Attention** | 关键 token 特殊标记 | 2023 |
| **Gemini 1.5 / Claude 3** | 生产 1M+ 上下文 | 2024-2025 |

**趋势**：长上下文推理已从"研究话题"变成"生产要求"（RAG / 代码 / 论文助手），系统层配套（KV cache 池化、chunked prefill、P/D 分离）都是为此服务。

#### 8.7.6 Encoder 侧的前沿（低热度但真实）

Encoder-only 模型（BERT / ViT / voice_safety）领域相对 LLM 热度低，但也有：
- **BEiT-3 / EVA** 统一 vision + text encoder
- **Flash-Linear-Attention** 线性注意力生产落地
- **RWKV / Mamba** 状态空间模型作为 encoder 替代（长序列友好）

**结论**：Encoder 模型 2025-2026 主要动向在**架构侧**（Transformer 替代品），推理编译栈变化不大，TRT / OpenVINO 依然是首选。

### 8.8 前沿选型指南：ROI 提示

**Q: 生产 LLM serving 想升级，先做什么？**

按 ROI 排序（回本快 → 慢）：

```
Tier 1 (立即可做, 换库调用即可):
  ├─ FlashInfer 集成 (attention kernel)          → decode latency ↓ 30%+
  ├─ FlashInfer sampling                          → sampling latency ↓ 50%+
  └─ Prefix caching (vLLM/SGLang 已内建)          → 多轮对话 2-5x

Tier 2 (需要工程量, 但架构可控):
  ├─ Chunked prefill 打开                        → P99 稳定性
  ├─ Continuous batching (基础功能, 未开必开)     → GPU util 30% → 80%
  └─ 量化到 FP8/INT4 (AWQ/GPTQ)                  → 显存 2-4x

Tier 3 (深度架构改动):
  ├─ P/D 分离 (需要 IB 网络 + Mooncake 协议)     → 生产吞吐 2-5x
  ├─ 推测解码 (MTP if DeepSeek else EAGLE-3)     → decode 1.4-6.5x
  └─ MLA (需要模型侧改造, 主要走 DeepSeek)        → KV cache 90%+ 减

Tier 4 (换硬件):
  └─ H100 → H200 → B200                          → 显存 + 带宽双升
```

**Q: 学习优先级？（面试 / 晋升场景）**

必知（能讲原理 + 生产数字）：
1. P/D 分离（Mooncake 架构一定要能画图）
2. EAGLE-3 / MTP 至少一个
3. FlashInfer 定位（不需要读 kernel 源码）

加分：

4. Sampling 算子重构（体现"细节意识"）
5. MLA / RadixAttention（体现"跟进最新"）

**Q: 会不会过时？**

- **P/D 分离**：架构级重构，未来 3 年主线，不会过时。
- **推测解码**：Model-inherent 方向（MTP）会挤压外挂 draft（EAGLE），但整体投机思路成熟稳定。
- **FlashInfer**：作为 kernel library 有生态壁垒，短期不会被替代。
- **FP4 / B200**：硬件驱动，看 NVIDIA 节奏（明年迭代继续）。

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

**文档版本**: v1.1 (2026-07-15)
**变更**: 新增 Part 8 前沿探索（2025-2026），追踪 P/D 分离、MTP、EAGLE-3、FlashInfer、Sampling 重构等 18 个月内业界重大进展
**作者**: chenglitao (based on voice_safety P2.5 project)
**License**: 内部资料，欢迎完善

> 有问题或补充，请提交 PR 或联系作者。祝面试顺利 🚀
