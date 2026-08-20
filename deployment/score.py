"""Scoring script for the managed online endpoint.

Not an MLflow-format model (trained on Colab, registered as a custom
model), so Azure ML can't auto-generate a scorer -- this is it. Loads
the base model in bf16 (no 4-bit/bitsandbytes: the endpoint runs on CPU
since GPU quota never landed, and bitsandbytes 4-bit requires CUDA) with
the registered LoRA adapter applied on top.
"""

import json
import logging
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 16

# Fixed at training time (train_df["intent"].unique()) -- not shipped as a
# separate artifact, so hardcoded here to build the identical system prompt.
LABELS = sorted([
    "cancel_order", "change_order", "change_shipping_address",
    "check_cancellation_fee", "check_invoice", "check_payment_methods",
    "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice",
    "get_refund", "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review",
    "set_up_shipping_address", "switch_account", "track_order", "track_refund",
])

model = None
tokenizer = None


def _find_adapter_dir(base_dir: str) -> str:
    for root, _, files in os.walk(base_dir):
        if "adapter_config.json" in files:
            return root
    raise FileNotFoundError(f"adapter_config.json not found under {base_dir}")


def init():
    global model, tokenizer

    model_root = os.getenv("AZUREML_MODEL_DIR")
    adapter_dir = _find_adapter_dir(model_root)
    logging.info("Loading adapter from %s", adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    logging.info("Model ready.")


def run(raw_data: str) -> dict:
    try:
        data = json.loads(raw_data)
        message = data["message"]
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        return {"error": f"Expected JSON body like {{'message': '...'}}: {e}"}

    system = (
        "You are a customer support ticket classifier. Given a customer "
        "message, respond with exactly one intent label from this list, and "
        "nothing else:\n" + ", ".join(LABELS)
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1] :]
    predicted = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {"message": message, "predicted_intent": predicted}
