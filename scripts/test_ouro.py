#!/usr/bin/env python3
"""Ouro-1.4B local inference smoke test.
Run from the ouro-venv:
  ouro-venv/bin/python test_ouro.py "your prompt"
"""
import sys, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/Volumes/high-speed/kuangre/models/ouro-1.4B"

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple GPU)"
    return torch.device("cpu"), "CPU"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="The future of AI is")
    ap.add_argument("--steps", type=int, default=None, help="override total_ut_steps")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--no-device-map", action="store_true")
    args = ap.parse_args()

    device, device_name = pick_device()
    print(f"[*] Device: {device_name} ({device})")
    print(f"[*] Loading tokenizer + model...", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.no_device_map:
        kwargs["device_map"] = None
    else:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **kwargs)
    model.eval()
    t1 = time.time()
    print(f"[*] Model loaded in {t1-t0:.1f}s")

    if args.steps is not None:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print(f"[*] Default total_ut_steps={cfg.total_ut_steps}, early_exit_threshold={cfg.early_exit_threshold}")
        print(f"[*] Overriding total_ut_steps -> {args.steps}")

    print(f"[*] Prompt: {args.prompt!r}")
    inputs = tok(args.prompt, return_tensors="pt").to(model.device)

    t2 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    t3 = time.time()

    gen = outputs[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    dt = t3 - t2
    print("\n" + "=" * 70)
    print("OUTPUT:")
    print("=" * 70)
    print(text)
    print("=" * 70)
    print(f"[*] Generated {len(gen)} tokens in {dt:.1f}s ({len(gen)/max(dt,1e-9):.2f} tok/s)")
    print(f"[*] Peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else f"[*] RAM estimate: {len(gen)} tokens")

if __name__ == "__main__":
    main()
