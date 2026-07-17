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
- [Part 8. 硬件视角](#part-8-硬件视角)
- [Part 9. 多卡并行推理](#part-9-多卡并行推理)
- [Part 10. 性能分析实战](#part-10-性能分析实战)
- [Part 11. 常见坑与踩雷](#part-11-常见坑与踩雷)
- [Part 12. 面试高频问题清单](#part-12-面试高频问题清单)
- [Part 13. 实战案例：voice_safety 全流程](#part-13-实战案例voice_safety-全流程)
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

## Part 8. 硬件视角

### 8.1 Roofline Model —— 判断计算 vs 内存瓶颈

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

### 8.2 Compute-bound vs Memory-bound 检测

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

### 8.3 常见 GPU 生产选型

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

## Part 9. 多卡并行推理

**核心问题**：一张卡装不下的模型，或者要更高吞吐。

### 9.1 三种并行策略

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

### 9.2 组合策略（3D Parallelism）

生产大模型（如 405B）通常混合：
```
TP=8 (单机 8 卡内)  x  PP=4 (跨机)  x  DP=2 (副本)  = 64 GPUs
```

### 9.3 通信关键

| 并行方式 | 通信量 | 硬件要求 |
|---|---|---|
| DP | AllReduce (仅训练) | PCIe 就够 |
| **TP** | **每层 AllReduce** | **NVLink 必需**（PCIe 会拖后腿） |
| PP | 层间 point-to-point | 慢通信可容忍 |
| EP (MoE) | AllToAll | NVLink / IB |

---

## Part 10. 性能分析实战

### 10.1 Profiling 工具矩阵

| 工具 | 出品方 | 适用 | 特色 |
|---|---|---|---|
| **nsys** (Nsight Systems) | NVIDIA | 系统级 timeline | 看 CPU/GPU 交互全景 |
| **nvprof / ncu** (Nsight Compute) | NVIDIA | Kernel 级细节 | 看每个 kernel 的 SM/Cache 利用 |
| **trtexec** | NVIDIA | TensorRT 专用 | 单个 engine 的 tuning 结果 |
| **PyTorch Profiler** | Meta | PyTorch 代码 | 分层耗时 |
| **py-spy** | Community | Python | 抓 Python 端 stack |
| **perf** | Linux | 通用 | 系统调用/CPU |
| **nvidia-smi dmon** | NVIDIA | GPU util 采样 | 生产监控友好 |

### 10.2 定位瓶颈的"四问法"

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

### 10.3 一个真实案例（voice_safety）

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

## Part 11. 常见坑与踩雷

### 11.1 编译期坑

| 现象 | 原因 | 解 |
|---|---|---|
| TRT build 极慢（几十分钟） | Dynamic shape profile 太宽 + workspace 不够 | 缩窄 profile 范围 / 扩 workspace |
| "Could not find implementation for ForeignNode" | TRT 编译器 bug 或 workspace OOM | 先扩 workspace，仍失败查 TRT changelog |
| ONNX 导出精度不一致 | opset 版本不同 / operator behavior 差异 | pin opset 版本，跑 numeric 对比 |
| 从 PyTorch 导出的 ONNX 有冗余节点 | trace 时产生的 dead ops | 用 onnx-simplifier 预处理 |

### 11.2 运行时坑

| 现象 | 原因 | 解 |
|---|---|---|
| 首次推理巨慢（warmup） | Kernel JIT / TRT engine build / CUDA context init | 部署时预推理 3-5 次 warmup |
| Dynamic shape 输入 shape 变化时慢 | TRT engine 需要重新做 shape inference | 减少 shape 变化，用 padding 或 bucketing |
| INT8 精度掉点严重 | Calibration set 不代表实际分布 | 用真实生产流量做 calibration |
| 长跑内存增长 | Buffer 未复用 / Session 泄漏 | 用 Ort::MemoryInfo 明确 buffer 归属 |
| GPU util 100% 但 QPS 上不去 | Kernel efficiency 低 (occupancy 差) | ncu profiling，可能是 kernel 选错 |
| 多请求延迟不稳 | Cross-batching 攒批时机变动 | 固定 batch collect window 分析 |

### 11.3 部署坑

| 现象 | 原因 | 解 |
|---|---|---|
| Pod 起来但 is_tensorrt=false | ORT 版本没含 TRT provider | 检查 tarball 名（-gpu-cuda12-）|
| Engine 加载失败 "different device" | 硬件 SM 版本不匹配 | 编译时对应 GPU 上编 |
| CUDA driver / runtime 版本冲突 | Forward compat 未启用 | 装 cuda-compat 包 |
| Kernel launch 报 "out of memory" | 显存碎片 | 重启 pod / 用 memory pool |
| 编译期 CUDA OOM | Workspace 超出显存 | 缩小 workspace |

### 11.4 精度坑

| 现象 | 原因 | 解 |
|---|---|---|
| FP32 训练 → FP16 推理精度掉 | 权重范围超 FP16 表示 | 找 outlier 层保 FP32 |
| INT8 校准后精度掉 5%+ | Calibration set 太小 / 分布偏 | 增大 calibration set / 用 percentile |
| FP8 精度损失 | E4M3 vs E5M2 选错 | Weight 用 E4M3, gradient 用 E5M2 |
| 大 batch 精度略低 | Numerical drift | 对精度敏感层保 FP32 |

---

## Part 12. 面试高频问题清单

### 12.1 基础概念

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

### 12.2 图优化

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

### 12.3 量化

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

### 12.4 LLM 特有

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

### 12.5 硬件/系统

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

### 12.6 系统设计

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

### 12.7 实战场景

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

## Part 13. 实战案例：voice_safety 全流程

### 13.1 项目背景

- **业务**：短视频音频娇喘 / 性内容检测
- **模型**：Roblox voice-safety-classifier-v3
- **架构**：24-layer Transformer, ~305M params
- **输入**：16kHz mono PCM，2-30s 音频
- **输出**：sexual_score float (二分类)
- **目标 QPS**：单卡 100+，业务 QPS 由副本数 × 单卡 QPS 决定

### 13.2 P2.5 阶段 7 轮迭代（浓缩版）

| Round | 关键动作 | 结果 |
|---|---|---|
| 1 | ORT 1.13.1 → 1.16.3 + base image 升级 | is_tensorrt=false，dlopen libcudart.so.11.0 失败 |
| 2 | ORT 1.16.3 → 1.17.1 -cuda12- 裸包 | is_cuda=true 但 trt_err="TensorRT provider is not enabled" |
| 3 | 1.17.1 → 1.17.3 **-gpu-cuda12-** 完整包 | is_tensorrt=true ✅，15s 音频 GT 精度对齐 |
| 4 | 尝试 profile T dynamic [32000, 480000] | TRT build 报 Error 10 "Could not find implementation" |
| 5 | Workaround: static T=240000 + engine.go zero-pad | 15s 音频跑通，但 20s 音频兜底、短音频精度稀释风险 |
| 6 | 扩 workspace 1GB → 8GB 试恢复 dynamic | 依然 crash 7 次，证伪 workspace OOM 假设 |
| 7 | 深挖发现 Issue #3490，升 base image 到 24.12-py3 (TRT 10.7) | ✅ 2:31 build 成功，dynamic T 原生跑通，20s 音频推理成功 |

### 13.3 单卡 145 QPS 的构成拆解

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

### 13.4 下一步优化路径（按 ROI）

见 [Part 11 优化空间分析](#part-11-常见坑与踩雷) 相关章节。

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

**文档版本**: v1.0 (2026-07-15)
**作者**: chenglitao (based on voice_safety P2.5 project)
**License**: 内部资料，欢迎完善

> 有问题或补充，请提交 PR 或联系作者。祝面试顺利 🚀
