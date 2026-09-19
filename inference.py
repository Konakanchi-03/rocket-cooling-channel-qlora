#!/usr/bin/env python3
"""
Inference CLI for Rocket Engine Cooling Channel Pre-Screening Model
Fine-tuned Qwen2.5-7B-Instruct with QLoRA
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./final_qlora_adapter"

SYSTEM_PROMPT = (
    "You are an expert aerospace propulsion thermal analyst. "
    "Your task is to act as a fast pre-screening filter for rocket engine cooling channel geometries, "
    "predicting pass/fail status against design constraints (material wall temperature limit <= 700.0 K, "
    "and expander power cycle outlet temperature limit >= 210.0 K) before CFD simulation."
)

def load_inference_model(base_model_id=BASE_MODEL_ID, adapter_path=ADAPTER_PATH):
    print(f"[*] Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    print(f"[*] Loading 4-bit base model: {base_model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"[*] Loading LoRA adapter weights from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[+] Model ready for inference!\n")
    return model, tokenizer

def predict_design(model, tokenizer, tw_inner, a_min_nominal, b_rib, ar_chamber, ar_throat, ar_exit):
    user_content = (
        f"Evaluate the following cooling channel geometry design candidate:\n"
        f"- Inner wall thickness (tw_inner): {tw_inner:.4f} mm\n"
        f"- Nominal throat channel width (a_min_nominal): {a_min_nominal:.4f} mm\n"
        f"- Rib width (b_rib): {b_rib:.4f} mm\n"
        f"- Chamber aspect ratio (AR_chamber): {ar_chamber:.4f}\n"
        f"- Throat aspect ratio (AR_throat): {ar_throat:.4f}\n"
        f"- Exit aspect ratio (AR_exit): {ar_exit:.4f}\n\n"
        f"Determine whether this design satisfies the thermal and power cycle constraints."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="Predict rocket cooling channel thermal/power pass-fail.")
    parser.add_argument("--tw_inner", type=float, default=1.20, help="Inner wall thickness in mm (0.5 to 3.0)")
    parser.add_argument("--a_min", type=float, default=1.50, help="Nominal throat channel width in mm (0.8 to 3.0)")
    parser.add_argument("--b_rib", type=float, default=2.00, help="Rib width in mm (0.8 to 3.0)")
    parser.add_argument("--ar_chamber", type=float, default=1.00, help="Chamber aspect ratio (0.5 to 1.5)")
    parser.add_argument("--ar_throat", type=float, default=5.00, help="Throat aspect ratio (2.0 to 8.0)")
    parser.add_argument("--ar_exit", type=float, default=0.65, help="Exit aspect ratio (0.3 to 1.0)")
    args = parser.parse_args()

    model, tokenizer = load_inference_model()

    print("=" * 60)
    print("INPUT CANDIDATE GEOMETRY:")
    print(f"  tw_inner:     {args.tw_inner:.4f} mm")
    print(f"  a_min_throat: {args.a_min:.4f} mm")
    print(f"  b_rib:        {args.b_rib:.4f} mm")
    print(f"  AR_chamber:   {args.ar_chamber:.4f}")
    print(f"  AR_throat:    {args.ar_throat:.4f}")
    print(f"  AR_exit:      {args.ar_exit:.4f}")
    print("=" * 60)

    verdict = predict_design(
        model, tokenizer,
        args.tw_inner, args.a_min, args.b_rib,
        args.ar_chamber, args.ar_throat, args.ar_exit
    )

    print("\nMODEL PRE-SCREENING RESULT:")
    print(verdict)
    print("=" * 60)

if __name__ == "__main__":
    main()
