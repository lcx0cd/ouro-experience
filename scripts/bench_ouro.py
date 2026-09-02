#!/usr/bin/env python3
"""Ouro-1.4B speed benchmark: find the right "档位" for your M4 Mac.
Measures:  steps × prompt_length → tok/s, prefill_time, gen_time.
Loads the model once per `steps` value (4 loads total), then runs all prompt sizes.

Usage:
  ouro-venv/bin/python bench_ouro.py
"""
import time, json, torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

MODEL_DIR = "/Volumes/high-speed/kuangre/models/ouro-1.4B"

# ── test prompts ───────────────────────────────────────────────────────────
SHORT = "你是一个宏观经济与房产叙事研究员。给定两个线索：Kiyosaki的$1.2B债务和深圳高评高贷案，请输出一个可核验推论。"

MEDIUM = """你是一个宏观经济与房产叙事研究员。给定以下两个线索：
1. Kiyosaki的$1.2B债务：NYPost报道他背了12亿债务，但实际是1500套公寓的投资组合债务，个人仅承担3000-6000万。这个数字是他自己主动吆喝的，在播客上说"So, I'm a billion two in debt"。
2. 深圳高评高贷案：2026年5月深圳700中介被抓，评估公司把500万的房子抬到650万，银行按650万放贷455万，多出的30万覆盖首付还倒拿装修款。

请输出：(a) 1-2个可核验推论；(b) 每个推论的核验方法。不要复述已知事实。"""

LONG = """你是一个宏观经济与房产叙事研究员。以下是两个研究线索：

研究线索一：Kiyosaki债务叙事。NYPost 2026-09-01报道"Kiyosaki is $1.2 billion in debt"。实际细节：这是他与合伙人共同持有的1500套公寓单元的投资组合债务，非个人欠款。个人承担约3000-6000万。数字是他自己在"Get Rich Education"播客主动说的。前妻Kim Kiyosaki澄清"严格说我们确实背着这些债，但债挂在资产上"。NYPost正文其实写了澄清，标题用负债框架制造点击。本质：媒体标题党，但Kiyosaki本人参与合谋——主动制造+利用这个数字作为注意力杠杆。

研究线索二：高评高贷/评估价操纵。中国侧（2026-05深圳700中介被抓）：评估公司把500万房子评估到650万，银行按650万放贷455万，多出30万覆盖首付，客户零首付上车还倒拿装修款。本质是虚增合同价/评估价从银行套取超过实际购房成本的资金，涉及阴阳合同。美国侧同构：2025全年美国房主提取2050亿房屋净值，2026 Q1单季470亿，其中60%是cash-out refinance，平均每笔提94000。Fannie Mae已对房产转LLC后60-180天cash-out refinance发出反欺诈警报。机制同构：评估价上涨→重新估值→提取超额贷款→当免税收入花掉。

请输出 (a) 1-2个可核验推论；(b) 每个推论的核验方法或数据源；(c) 与认知基座定理（表达vs结算/铡刀论/压缩函数所有权）的闭环一句话。不要复述已有事实。"""

SIZES = [
    ("short",  SHORT,  "~80 chars"),
    ("medium", MEDIUM, "~400 chars"),
    ("long",   LONG,   "~1200 chars"),
]
STEPS = [1, 2, 3, 4]

def build_model(steps):
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg.total_ut_steps = steps
    cfg.early_exit_threshold = 1.0
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return tok, model

def run_one(model, tok, prompt, max_new=60):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    inp_len = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=False, temperature=None, top_p=None,
        )
    t1 = time.time()
    gen_tokens = outputs[0][inp_len:].shape[0]
    dt = t1 - t0
    return {
        "prompt_tokens": inp_len,
        "gen_tokens": gen_tokens,
        "time_s": round(dt, 2),
        "tok_s": round(gen_tokens / max(dt, 1e-9), 2),
    }

def main():
    dev = "MPS" if torch.backends.mps.is_available() else "CPU"
    print("=" * 64)
    print("Ouro-1.4B Speed Benchmark  (device=%s)" % dev)
    print("=" * 64)
    hdr = f"{'steps':<6}{'size':<9}{'prompt_tok':<12}{'gen_tok':<9}{'time_s':<9}{'tok_s':<8}"
    print(hdr)
    print("-" * 64)

    results = []
    for s in STEPS:
        print(f"\n# steps={s}: loading model once...", flush=True)
        tok, model = build_model(s)
        for name, prompt, desc in SIZES:
            r = run_one(model, tok, prompt, max_new=60)
            r.update(steps=s, prompt_size=name, prompt_desc=desc)
            results.append(r)
            print(f"{r['steps']:<6}{r['prompt_size']:<9}{r['prompt_tokens']:<12}{r['gen_tokens']:<9}{r['time_s']:<9}{r['tok_s']:<8}  tok/s", flush=True)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n" + "=" * 64)
    print("JSON:")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # recommendation: short prompt baseline per steps
    print("\n=== 推荐档位（短 prompt 基线）===")
    for s in STEPS:
        row = [r for r in results if r["steps"] == s and r["prompt_size"] == "short"]
        if row:
            print(f"  steps={s}:  {row[0]['tok_s']} tok/s   (prompt {row[0]['prompt_tokens']} tok)")

if __name__ == "__main__":
    main()