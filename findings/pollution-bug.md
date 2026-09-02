# 显存污染 Bug：torch-MPS 连续 generate 速度退化

## 现象

在 torch-MPS 上对同一个模型**连续多次**调用 `model.generate()`，速度和输出质量逐次退化：

| 次数 | 速度 | 输出 |
|------|------|------|
| 第 1 次 | 10.3 tok/s | 正常 |
| 第 2 次 | 4.1 tok/s | 开始异常 |
| 第 3 次 | 1.9 tok/s | 重复循环 |

在连续跑 10 个问题时，速度从 Q1 的 10.7 tok/s 一路掉到 Q2-Q10 的 1.3-2.0 tok/s，同时输出开始大量重复（"可能的可能的可能的"、"20%的20%的20%"、"2008年 2月 2008年 2月"）。

## 根因

Ouro 使用 `UniversalTransformerCache`（4 UT steps × 24 层 = 96 槽 KV cache）。在 Apple MPS 上连续 generate 时，KV cache 和中间张量没有被正确释放，显存逐渐累积直到触发 swap。**速度退化和输出退化是同一个根因**——显存压力下计算开始出错。

## 修复

每次 generate 之后必须手动清理：

```python
import gc
import torch

with torch.no_grad():
    out = model.generate(input_ids=inp, max_new_tokens=120, do_sample=False)

del out
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

## 验证

加上清理后，同一题连跑 4 次速度稳定在 11.7-11.8 tok/s，输出完全一致（确定性的 greedy decode）：

```
第1次: 80 tok in 7.4s = 10.8 tok/s | 嗯，我现在要解决这个问题...
第2次: 80 tok in 6.8s = 11.7 tok/s | 嗯，我现在要解决这个问题...
第3次: 80 tok in 6.8s = 11.7 tok/s | 嗯，我现在要解决这个问题...
第4次: 80 tok in 6.8s = 11.8 tok/s | 嗯，我现在要解决这个问题...
```

## 教训

1. **任何在 torch-MPS 上做批量评测的脚本，都必须在每次 generate 后清理显存**，否则第 N 个结果不可信。
2. 如果看到"速度逐次变慢 + 输出开始重复"同时出现，先怀疑显存污染，不要急着下"模型不行"的结论。
3. 这个 bug 让我们差点误判 Ouro 的能力——第一批 10 题测试全部作废，修正后重跑。

## 检测方法

在每次 generate 前后打印 MPS 内存占用：

```python
if torch.backends.mps.is_available():
    print(f"MPS allocated: {torch.mps.current_allocated_memory()/1e9:.2f} GB")
```
