import os
import sys
import time
import json
import threading
import subprocess
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# -------------------------------------------------------------
# 1. Background GPU Monitor Thread
# -------------------------------------------------------------
stop_monitor = False

def monitor_gpu(log_file="nvidia_smi.log", interval=5):
    """Logs nvidia-smi every `interval` seconds to `log_file`."""
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== GPU Monitoring Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write("timestamp, memory.used [MiB], memory.total [MiB], utilization.gpu [%], temperature.gpu\n")
    while not stop_monitor:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(res.stdout)
        except Exception:
            pass
        time.sleep(interval)

# -------------------------------------------------------------
# 2. Main Training Function
# -------------------------------------------------------------
def main():
    global stop_monitor
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    print("[Monitor] GPU monitoring started -> logging to nvidia_smi.log (updates every 5s)")

    print("\n" + "="*70)
    print("PHASE 3: ENVIRONMENT & QLORA SETUP")
    print("="*70)

    # VRAM Budget Estimation
    print("\n[VRAM Budget Estimation on 16GB RTX 5060 Ti]")
    print("  - Base model (Qwen2.5-7B in NF4 4-bit) : ~5.18 GB")
    print("  - LoRA Adapter Weights (rank 16, all-linear) : ~0.15 GB")
    print("  - Paged 8-bit AdamW Optimizer states         : ~0.35 GB")
    print("  - Forward Activations (batch_size=2, max_len=512, grad_ckpt) : ~1.50 GB")
    print("  - Expected Peak VRAM Usage                   : ~7.2 - 7.5 GB")
    print("  - Available VRAM Headroom on 16GB GPU        : ~8.5 - 9.0 GB")

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    # Tokenizer
    print(f"\n[Loading Tokenizer]: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Base Model
    print(f"[Loading Base Model in 4-bit NF4]: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    # Rationale: Rank 16 gives optimal capacity for learning numerical threshold rules
    # and constraint mapping without overfitting on 170 examples.
    # Alpha 32 scales updates stably (alpha = 2 * r).
    # All linear projection modules (attention + MLP) ensure both contextual attention
    # and factual/numerical representations are updated.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    print("\n[Trainable Parameters Breakdown]:")
    model.print_trainable_parameters()

    # Load and Format Datasets
    print("\n[Loading Datasets]...")
    with open("train.jsonl", "r", encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f]
    with open("val.jsonl", "r", encoding="utf-8") as f:
        val_raw = [json.loads(line) for line in f]

    train_ds = Dataset.from_dict({"text": [tokenizer.apply_chat_template(s, tokenize=False) for s in train_raw]})
    val_ds = Dataset.from_dict({"text": [tokenizer.apply_chat_template(s, tokenize=False) for s in val_raw]})

    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # Training Configuration
    output_dir = "./qlora_checkpoints"
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=512,
        report_to="none",
        logging_first_step=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("\n" + "="*70)
    print("PHASE 4: FIRST TRAINING PASS (MECHANICAL TEST RUN)")
    print("="*70)
    print(f"Starting training for {sft_config.num_train_epochs} epochs (~31 total optimization steps)...")

    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    print(f"\n[Training Finished in {training_time:.2f}s ({training_time/60:.2f} min)]")

    # Save final model adapter
    adapter_save_dir = "./final_qlora_adapter"
    print(f"Saving fine-tuned adapter to {adapter_save_dir}...")
    trainer.model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    # Loss curve summary
    print("\n[Training Loss Log History]:")
    loss_history = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            loss_history.append((entry.get("step", 0), entry.get("epoch", 0), entry.get("loss", 0)))
            print(f"  Step {entry.get('step', 0):3d} | Epoch {entry.get('epoch', 0):.2f} | Loss: {entry.get('loss', 0):.4f}")
        elif "eval_loss" in entry:
            print(f"  --> Eval Step {entry.get('step', 0):3d} | Eval Loss: {entry.get('eval_loss', 0):.4f}")

    # Check GPU memory
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\n[VRAM Summary]: Peak Allocated: {max_allocated_gb:.2f} GB | Current: {allocated_gb:.2f} GB")

    # -------------------------------------------------------------
    # Validation Inference Comparison (3-5 examples)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 4 EVALUATION: HELD-OUT VALIDATION SAMPLES (SIDE-BY-SIDE)")
    print("="*70)

    model.eval()
    test_indices = [0, 5, 12, 17, 25]  # Diverse spread across val set
    test_samples = [val_raw[i] for i in test_indices if i < len(val_raw)]

    eval_results = []
    for idx, sample in enumerate(test_samples):
        sys_msg = sample[0]
        user_msg = sample[1]
        gt_assistant = sample[2]['content']

        # Format input prompt (without ground truth assistant response)
        input_prompt = tokenizer.apply_chat_template(
            [sys_msg, user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        predicted_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

        eval_results.append({
            "sample_index": idx + 1,
            "inputs": user_msg['content'].strip(),
            "predicted": predicted_text,
            "ground_truth": gt_assistant.strip(),
        })

        print(f"\n--- [Validation Case #{idx+1}] ---")
        print(f"INPUT DESIGN VARIABLES:\n{user_msg['content'].strip()}")
        print("\n--- PREDICTED VERDICT ---")
        print(predicted_text)
        print("\n--- GROUND TRUTH ---")
        print(gt_assistant.strip())
        print("-" * 50)

    # Save evaluation summary to JSON
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    stop_monitor = True
    print("\nTraining & evaluation completed successfully. Results saved to eval_results.json and nvidia_smi.log.")

if __name__ == '__main__':
    main()
