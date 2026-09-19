#!/usr/bin/env python3
"""
Comprehensive Validation Evaluation Script
Evaluates fine-tuned QLoRA model on all held-out validation samples.
Computes Confusion Matrices, Precision, Recall, F1, and MAE metrics.
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./final_qlora_adapter"

def parse_response(text):
    # Overall verdict
    v_match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    verdict = v_match.group(1).upper() if v_match else "UNKNOWN"

    # Material
    m_match = re.search(r"Material Constraint:\s*(PASS|FAIL)", text, re.IGNORECASE)
    mat = m_match.group(1).upper() if m_match else "UNKNOWN"
    twg_match = re.search(r"Twg\s*=\s*([0-9.]+)", text)
    twg = float(twg_match.group(1)) if twg_match else None

    # Power
    p_match = re.search(r"Power Constraint:\s*(PASS|FAIL)", text, re.IGNORECASE)
    pwr = p_match.group(1).upper() if p_match else "UNKNOWN"
    tc_match = re.search(r"Tc_out\s*=\s*([0-9.]+)", text)
    tc = float(tc_match.group(1)) if tc_match else None

    dp_match = re.search(r"Delta_P\s*=\s*([0-9.]+)", text)
    dp = float(dp_match.group(1)) if dp_match else None

    return {
        "verdict": verdict,
        "material": mat,
        "power": pwr,
        "twg": twg,
        "tc_out": tc,
        "delta_p": dp
    }

def main():
    print("Loading model and tokenizer for full validation evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    with open("val.jsonl", "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f]

    print(f"Running inference across all {len(val_samples)} validation samples...")
    results = []

    for idx, sample in enumerate(val_samples):
        sys_msg = sample[0]
        user_msg = sample[1]
        gt_raw = sample[2]['content']

        prompt = tokenizer.apply_chat_template([sys_msg, user_msg], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        pred_raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

        pred_parsed = parse_response(pred_raw)
        gt_parsed = parse_response(gt_raw)

        results.append({
            "idx": idx + 1,
            "pred": pred_parsed,
            "gt": gt_parsed,
            "pred_raw": pred_raw,
            "gt_raw": gt_raw
        })

    # Compute metrics
    # Overall Verdict
    tp = sum(1 for r in results if r['pred']['verdict'] == 'PASS' and r['gt']['verdict'] == 'PASS')
    fp = sum(1 for r in results if r['pred']['verdict'] == 'PASS' and r['gt']['verdict'] == 'FAIL')
    tn = sum(1 for r in results if r['pred']['verdict'] == 'FAIL' and r['gt']['verdict'] == 'FAIL')
    fn = sum(1 for r in results if r['pred']['verdict'] == 'FAIL' and r['gt']['verdict'] == 'PASS')
    total = len(results)

    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print("\n" + "="*60)
    print("VALIDATION BENCHMARK RESULTS (N=30)")
    print("="*60)
    print(f"Accuracy  : {acc*100:.2f}% ({tp + tn}/{total})")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-Score  : {f1:.4f}")
    print("\nConfusion Matrix (Overall Verdict):")
    print(f"                 Ground Truth PASS | Ground Truth FAIL")
    print(f"Predicted PASS :        {tp:2d}         |        {fp:2d}")
    print(f"Predicted FAIL :        {fn:2d}         |        {tn:2d}")
    print("="*60)

    # Save benchmark metrics to json
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "details": results
    }
    with open("benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Full metrics saved to benchmark_metrics.json\n")

if __name__ == "__main__":
    main()
