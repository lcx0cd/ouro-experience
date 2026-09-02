# Ouro-1.4B 完整速度基准数据（Apple M4, 16GB, MPS）

设备：Apple M4 (10-core GPU) / 16GB 统一内存 / torch 2.13 / transformers 4.57.6 / BF16

## 1. 步数 × prompt 长度 → tok/s

**方法**：同一模型，steps ∈ {1,2,3,4} × prompt ∈ {短,中,长}，greedy decode 60 tokens。

| steps | 短 prompt (93 tok) | 中 prompt (370 tok) | 长 prompt (936 tok) |
|-------|-------------------|-------------------|-------------------|
| 1     | **19.43**          | 16.55             | 9.19 (仅 22 tok 后停) |
| 2     | 12.34              | 10.45             | 7.72 |
| 3     | 8.47               | 7.13              | 4.56 |
| 4     | 6.52               | 5.18              | 2.94 |

**关键观察**：
- 每加 1 步，速度约减半（符合"权重读取次数线性增加"的理论）
- **prompt 长度的影响几乎和步数一样大**：936 token 长 prompt 即使 steps=1 也只有 9.2 tok/s
- steps=1 + 长 prompt 时只生成了 22 token 就停（EOS 提前），说明长 context 下模型行为异常

## 2. 理论极限分析

Ouro 每生成 1 token 要把 2.87GB 权重读 4 遍（steps=4）= 11.5GB/token。
M4 内存带宽 ~120GB/s → 理论极限 ~10 tok/s。实测 6.5 tok/s = 65% 效率。
**这是架构物理约束，不是工程问题——LoopLM 用 4× 算力换 3-4× 参数效率。**

## 3. 长序列 KV cache 膨胀

| max_new | 速度 | 说明 |
|---------|------|------|
| 60  | 5.73 tok/s | 基线 |
| 200 | 3.26 tok/s | 掉 43% |
| 500 | 3.27 tok/s | 稳定在低档 |

Ouro 的 KV cache 是 96 槽（4 steps × 24 层），是普通模型的 4 倍。
论文 Table 14 验证"last-step cache 共享"可在 decode 阶段省 4× 内存且质量几乎不掉，但需改 modeling 代码。

## 4. 自适应退出实测（base + Thinking 模型）

发布的两个模型（base 和 Thinking）的 early-exit gate 前两步 λ 值极低：

| 模型 | λ(step0) | λ(step1) | λ(step2) | λ(step3) |
|------|---------|---------|---------|---------|
| base     | 0.027 | 0.075 | 0.305 | 0.500 |
| Thinking | 0.026 | 0.083 | 0.363 | 0.508 |

| threshold | base 平均步数 | Thinking 平均步数 |
|-----------|-------------|-----------------|
| 0.5  | 3.69 | 3.40 |
| 0.75 | 3.97 | 3.97 |
| 0.9+ | 4.00 | 4.00 |

**结论**：自适应退出对发布模型无效——论文 3.4 节的 specialized adaptive exit training 未包含在发布模型中。

## 5. MLX 路径速览

- 自写 MLX 移植（`ouro_mlx.py`）：模型正确，int8 量化质量正常，但 eager 模式 0.8-0.9 tok/s（Python 调度瓶颈）
- `mx.compile` 需要函数式 cache 重构，且 Ouro 的 96 槽 cache 与 MLX 编译不友好
- mlx-community 4bit 版本基于修复前模型（2025-11），装 mlx-lm 会强升 transformers 破坏兼容
- 结论：**MLX 路线在 M4 上不值得继续投入**，除非社区出现针对 looped 架构的编译优化