"""Smoke test: does Qwen2.5-1.5B-Instruct actually load in 4-bit and fit
on this GPU (GTX 1660 Ti, 6GB VRAM)? Confirms the local QLoRA path is
viable before committing to a full training run.
"""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def gb(bytes_val: int) -> float:
    return round(bytes_val / 1e9, 2)


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {gb(torch.cuda.get_device_properties(0).total_memory)} GB")
    print(f"Already allocated (other apps): {gb(torch.cuda.memory_allocated(0))} GB")

    print(f"\nLoading {MODEL_NAME} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    print(f"VRAM after loading base model: {gb(torch.cuda.memory_allocated(0))} GB")

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"VRAM after adding LoRA adapters: {gb(torch.cuda.memory_allocated(0))} GB")

    # Tiny forward+backward pass to confirm training actually works, not just loading
    print("\nRunning one forward+backward pass...")
    messages = [
        {"role": "system", "content": "You are a customer support ticket classifier."},
        {"role": "user", "content": "I want to cancel my order."},
        {"role": "assistant", "content": "cancel_order"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Peak VRAM during forward+backward: {gb(torch.cuda.max_memory_allocated(0))} GB")

    print("\nSMOKE TEST PASSED — local QLoRA training is viable on this GPU.")


if __name__ == "__main__":
    main()
