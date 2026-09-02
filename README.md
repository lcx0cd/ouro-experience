# Ouro-1.4B 本地部署实录：踩坑 → 修正 → 结论

> 在 Apple M4 (16GB, MPS) 上部署 ByteDance Ouro-1.4B（Looped Language Model）的完整实验报告。
> 包括：环境搭建、速度基准、自适应退出验证、MLX 移植尝试、显存污染发现、提示词格式陷阱、10 题对比实测。

---

## 结论概要

**Ouro-1.4B 在 Apple M4 上的中文复杂推理任务中，不具备替代云端大模型（如 DeepSeek）的能力。** 原因：
1. **中文能力弱**：训练数据中中文仅占 ~2%，输出中英混杂、重复循环
2. **推理深度浅**：停留在"复述问题+泛泛讨论"，无法输出论文要求的"可核验推论"
3. **输出不稳定**：3/10 题退化到重复循环
4. **速度受架构限制**：4 步循环 × 2.87GB 权重 = 11.5GB/token 读取，M4 理论极限 ~10 tok/s

但这不代表实验没有价值——我们发现了几个**对任何本地模型部署都有用的教训**。

---

## 关键发现（对任何人都实用）

### 1. 显存污染 🐛 → 脚本必须清理 MPS 缓存

**现象**：在 torch-MPS 上连续多次 `model.generate()` 后，速度从 10.7 tok/s 逐次掉到 1.3 tok/s，同时输出质量退化（重复循环、数字爆炸）。

**原因**：Ouro 的 UniversalTransformerCache 在 MPS 上连续 generate 时可能有显存累积，导致越来越慢直到 swap。

**修复**：每次 generate 之间必须：
```python
del out
gc.collect()
torch.mps.empty_cache()
```

**验证**：加上清理后，10 题速度稳定在 10.2-12.0 tok/s，不再退化。

### 2. 提示词格式陷阱 🪤 → 格式比内容更敏感

**现象**：用 `"事实：……问题：……"` 格式时，模型退化到输出"可能的可能的可能的"或"2008年 2月 2008年 2月"等重复内容。用自然提问格式时，同一题输出正常。

**原因**：Ouro-1.4B base 在训练数据中可能见过大量 `"事实：……问题：……"` 格式的多选题/判断题，触发了硬编码的输出模式。

**结论**：格式化的 prompt 前缀（如 `"事实："`、`"问题："`、`"A. B. C. D."`）会触发小模型在训练数据中见过的特定模式，**用小模型时必须用自然语言格式**。

### 3. 自适应退出对发布模型无效

**现象**：论文宣称 adaptive exit 可在平均 2.5 步达到 66% 准确率（vs 4 步 67.35%）。但实测 base 和 Thinking 模型的前两步 λ 值极低（0.03/0.08），threshold=0.5 时平均仍需 3.69 步。

**原因**：论文 3.4 节的 `specialized adaptive exit training` 没有包含在发布模型中——那是消融实验的训练，不是成品。

### 4. KV cache 4× 膨胀 → 长序列掉速 43%

60 tok 生成：5.73 tok/s → 500 tok 生成：3.27 tok/s（掉速 43%）。论文 Table 14 验证了"last-step cache 共享"可恢复，但需要改 modeling 代码。

### 5. MLX 社区 4bit 版本的陷阱

`mlx-community/Ouro-1.4B-4bit` 基于 **2025-11-09** 的官方模型转换（早于官方 2026-01-18 的 KV cache fix），且用旧版 mlx-lm 0.28.4。装 mlx-lm 0.31.3 会强升 transformers 到 5.x，破坏 Ouro 兼容性。

---

## 速度基准（M4, 10-core GPU, 16GB, torch-MPS BF16）

| steps | 短 prompt (93 tok) | 中 prompt (370 tok) | 长 prompt (936 tok) |
|-------|-------------------|-------------------|-------------------|
| 1     | 19.43 tok/s       | 16.55 tok/s       | 9.19 tok/s (22 tok 后 EOS) |
| **2** | **12.34 tok/s**   | **10.45 tok/s**   | 7.72 tok/s        |
| 3     | 8.47 tok/s        | 7.13 tok/s        | 4.56 tok/s        |
| 4     | 6.52 tok/s        | 5.18 tok/s        | 2.94 tok/s        |

**推荐默认档**：steps=2，短 prompt 下 12.3 tok/s，质量与 steps=4 差距很小（论文 Table 7: MMLU ~60.9% vs 67.35%）。

---

## 10 题对比实测

从"Kiyosaki 房产叙事地图"研究文档中精切了 10 个子议题，同一 prompt 分别喂给 Ouro-1.4B（本地 steps=2）和 DeepSeek v4-flash（云端）。

**结论**：Ouro 在 10 题上均未达到 DeepSeek 的质量。详见 `results/10-question-comparison.md`。

---

## 官方生态核查 + 我们的判断（2026-09）

项目页 `ouro-llm.github.io` 声称 "vLLM and SGLang integration is ready"。复核后的准确状态：

- **vLLM**：`registry.py` 已注册 `OuroForCausalLM`（v0.26.0），但无独立 `ouro.py`——transformers 后端通用加载。模型卡确认"vLLM 不支持 adaptive exit，总是跑满 4 步"。属"能跑"层面，无 loop 专门优化。
- **SGLang**：同通用加载，无 loop 专门实现。
- **无 GGUF/llama.cpp**（架构不支持）、**无 AWQ/GPTQ**（HF 仅列 MLX 4bit）、**无 MLX 编译优化**（mlx-engine Issue #277 无 PR）。

**判断（已确认并发表）**：发布 11 个月、下载 2 万次，官方支持存在但都停留在"能跑"的通用加载层面——**没有出现过针对 loop 架构的推理优化工作**（无 GGUF、无 MLX 编译优化、无专门推理框架、论文 3.4 节的 specialized adaptive exit training 也未发布进权重）。这个架构方向的推理优化基本是空白。

**对我们结论的影响**：这解释了为什么 M4 上没有现成的提速路径——不是我们没找到，而是这个方向本身缺乏社区投入。

---

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv ouro-venv
source ouro-venv/bin/activate

# 2. 安装依赖（注意：必须 transformers<4.56 或 >=4.56 但需 ouro-cache-fix）
pip install torch transformers==4.57.6 accelerate

# 3. 下载模型
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ByteDance/Ouro-1.4B

# 4. 跑一个测试
python scripts/test_ouro.py "The future of AI is" --steps 2 --max-new 60
```

---

## 文件结构

```
ouro-experience/
├── README.md                    # 本文件
├── scripts/
│   ├── test_ouro.py             # 基础推理脚本（含显存清理）
│   └── bench_ouro.py            # 速度基准（含 token 长度影响）
├── configs/
│   ├── steps-2-config.json      # steps=2 推荐配置
│   └── steps-4-config.json      # 论文默认配置
├── results/
│   ├── benchmark-data.md        # 完整基准数据
│   ├── gate-distribution.md     # 自适应退出验证数据
│   └── 10-question-comparison.md  # 10 题对比详情
└── findings/
    ├── pollution-bug.md         # 显存污染 + 解决方案
    └── prompt-format.md         # 提示词格式影响
```

---

## 许可证

MIT