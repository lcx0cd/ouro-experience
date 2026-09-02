# 自适应退出（Adaptive Exit）验证数据

## 背景

论文 arXiv:2510.25741 宣称通过 early-exit gate 可在平均 2.5 步达到 66% 准确率（vs 4 步 67.35%），
实现"简单输入提前退出、复杂输入多算几步"的自适应计算。

## 验证方法

加载官方发布模型（base / Thinking），对研究 prompt 做 forward，收集每步的 gate 输出，
计算 `λ_i = sigmoid(gate_i)` 和退出分布。

## 结果

### gate 的 λ 值（每步平均）

| 模型 | λ(step0) | λ(step1) | λ(step2) | λ(step3) |
|------|---------|---------|---------|---------|
| Ouro-1.4B base | 0.027 | 0.075 | 0.305 | 0.500 |
| Ouro-1.4B-Thinking | 0.026 | 0.083 | 0.363 | 0.508 |

前两步 λ 极低（0.03/0.08），说明模型几乎不会在前两步退出。

### 不同 threshold 下的平均步数

| threshold | base 平均步数 | Thinking 平均步数 | 速度增益 |
|-----------|-------------|-----------------|---------|
| 0.5  | 3.69 | 3.40 | ~8% |
| 0.75 | 3.97 | 3.97 | ~1% |
| 0.9  | 4.00 | 4.00 | 0% |
| 1.0  | 4.00 | 4.00 | 0% |

## 结论

**自适应退出对官方发布的模型无效。** 无论 base 还是 Thinking，前两步的退出概率都极低，
即使把 threshold 降到 0.5，平均仍需 3.4-3.7 步，只省 8% 计算量。

## 原因

论文 3.4 节的 "specialized adaptive exit training"（显式教 gate 根据任务损失做退出决策）
是**消融实验的训练过程**，没有包含在 HF 发布的两个模型中。发布模型只有 entropy-regularized
目标训练出的 gate，它学到的是"深度分布尽量均匀"，而不是"根据难度退出"。

## 验证代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
cfg.total_ut_steps = 4
model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# 直接调 model.model 获取 gate_list
_, _, gate_list = model.model(input_ids=inp)
lambdas = [torch.sigmoid(g.squeeze(-1)) for g in gate_list]  # 每步的 λ
```