# ml-compile-review

> 机器学习编译 (ML Compilation) 与推理加速的个人学习/复习/面试笔记合集
>
> 从 voice_safety P2.5 生产项目实战整理而来, 之后逐步扩充相关主题。

## 为什么建这个仓

- 机器学习编译 (MLC/ML compilation) 是最近几年跨 "深度学习框架 × 编译器 × 硬件" 三个领域的融合方向, 涉及 IR 设计、图优化、算子融合、内核生成、量化、硬件后端等大量概念, 需要**系统化整理**才能形成体系。
- 生产项目里踩过的坑 (TRT 8.6 → 10.7 动态 shape bug、CUDA/cuDNN/ORT 版本矩阵、量化精度对齐等) **值得沉淀**, 便于面试复述 + 后续同类项目复用。
- 相较于零散的知乎/博客文章, 本仓追求**互相印证**: 每个理论点尽量绑定实战 case (数据/日志/代码路径), 反之每个 case 都上溯到理论根因。

## 当前收录

| 文档 | 内容概要 | 行数 |
|---|---|---|
| [`ml_inference_acceleration.md`](./ml_inference_acceleration.md) | 推理加速完整手册: 加速金字塔 5 层 · 图优化 L1-L6 · 6 大推理框架矩阵 · 量化家族 · LLM 专属优化 · **前沿探索 (P/D 分离 · MTP · EAGLE · FlashInfer · Sampling · MLA · RadixAttention · FP4)** · 硬件视角 · 多卡并行 · 性能分析 · 面试题清单 · voice_safety 实战 | 2066 |

### `ml_inference_acceleration.md` 章节速览

1. **全景**: 训练 vs 推理为何要单独加速, 推理场景的核心指标 (延迟/吞吐/成本三角)
2. **加速五个层级**: 硬件 → 算子 → 图 → 框架 → 服务化融合 (金字塔模型)
3. **图优化 L1-L6**: 常量折叠 → 算子融合 → 布局优化 → 内存重用 → 精度混合 → 并行调度
4. **主流推理框架矩阵**: TensorRT / ONNX Runtime / TorchScript / XLA / TVM / vLLM / TGI 横向对比
5. **模型类型 × 推理方案选型**: encoder-only / decoder-only / encoder-decoder 各自的加速路径
6. **量化技术家族**: PTQ / QAT / SmoothQuant / GPTQ / AWQ / KV-Cache 量化
7. **LLM 推理专属优化**: KV-Cache / PagedAttention / Continuous Batching / Speculative Decoding / MoE 稀疏化
8. **前沿探索 (2025-2026)** 🆕: 每技术按「原理 → 数据 → 工程支持」组织
   - **8.1 P/D 分离**: DistServe M/D/1 排队公式 · Mooncake 三级存储 · Layer-wise KV 传输 overlap · 4 大生产系统对比
   - **8.2 MTP**: DeepSeek-V3 MTP 架构图 + loss 公式 · 85-90% 接受率 · 与 EAGLE 对比
   - **8.3 EAGLE**: feature-level 自回归定义 · Autoregression Head 参数量表 · Tree Attention · SmoothL1 + CE loss · 三代差异
   - **8.4 FlashInfer**: BSR 布局 · JIT template functor · Load-balanced scheduler (Stream-K 灵感)
   - **8.5 Sampling 重构**: 传统 sort-based O(V log V) 瓶颈 · Dual Pivot Rejection Sampling 伪代码
   - **8.6 MLA**: latent 维度数学表达 · KV cache 压缩 93%
   - **8.7 RadixAttention**: 跨请求 prefix radix tree
   - **8.8 FP4 / Blackwell**: MXFP4 微缩放 · 精度对齐挑战
9. **硬件视角**: NVIDIA GPU 代际差异 / TPU / IPU / 国产芯片 (Habana/寒武纪/华为昇腾)
10. **多卡并行推理**: TP / PP / EP / DP 组合策略
11. **性能分析实战**: Nsight / nvprof / TensorBoard / PyTorch Profiler 使用要点
12. **常见坑与踩雷**: 版本矩阵地狱 / TRT dynamic shape bug / 精度对齐陷阱
13. **面试高频问题清单**: 60+ 题, 从概念到实战
14. **实战案例**: voice_safety P2.5 从 ORT 1.13.1 CUDA EP → ORT 1.20 TRT 10.7 dynamic shape 的完整旅程

## 未来补充计划

- [ ] TVM / MLIR / IREE 三家开源 MLC 编译器深入对比
- [ ] Triton (OpenAI) 与 CUDA/CUTLASS 的关系
- [x] ~~Speculative Decoding 主流实现深挖~~ → 已在 Part 8 追加 EAGLE-1/2/3 演进 + MTP + Medusa 对比
- [ ] Quantization-Aware Training 完整 pipeline (以 LLM 为例)
- [ ] 服务化框架深挖 (vLLM PagedAttention 源码走读 / TGI 架构)
- [ ] Mooncake KV cache 池化协议深度拆解 (P/D 分离下 KV 传输的工程细节)
- [ ] 国产 AI 芯片编译栈调研 (华为 CANN / 寒武纪 Neuware / 昇思 MindSpore Lite)

## 使用方式

- 面试前顺一遍 §13 面试题, 遇到不熟就跳回对应 Part 补
- 生产项目遇到具体优化, 从 §5 选型开始, 落到 §3 图优化找具体技术
- 版本冲突/踩坑时, 优先看 §12 有无对应条目, 再看 §14 实战案例
- **关注业界前沿** (2025-2026 SOTA 追新): 直接看 §8, 每次论文/框架发布后同步更新

## 版本历史

- **v1.2** (2026-07-15): Part 8 前沿探索重写。每技术改为「原理（图/伪代码/公式）→ 数据 → 工程支持」结构，补充 DistServe M/D/1 公式、EAGLE 参数量表、FlashInfer BSR/JIT/Scheduler 三大机制、Dual Pivot Rejection Sampling 伪代码、MLA 数学表达等技术细节
- **v1.1** (2026-07-15): 新增 Part 8 前沿探索 —— P/D 分离、MTP、EAGLE-3、FlashInfer、Sampling 重构
- **v1.0** (2026-07-15): 初始版本，Part 1-13 完整手册（现 Part 1-7 + Part 9-14）

## License

个人学习笔记, MIT.
