# Code we used to fine-tune Qwen3-VL and Qwen2.5-VL models one epoch at a time.


# It doesn't run properly with torch 2.9.0
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126


import zipfile
import os
import json
import random
import time
import torch
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, default_data_collator, AutoProcessor
from datasets import load_dataset
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm


train_image_zip = "" # Insert paths to your training set here
train_text_zip  = "" 
val_image_zip   = "" # Insert paths to your validation set here
val_text_zip    = ""


def my_unzipper(zip_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, dst_dir)
    print(f"\nExtracted {len(members)} files to {dst_dir}")


def build_jsonl(image_zip, text_zip, out_jsonl, subset_name):
    image_dir = os.path.join(work_dir, f"images_{subset_name}")
    text_dir  = os.path.join(work_dir, f"texts_{subset_name}")

    print(f"\nExtracting {subset_name} set...")
    my_unzipper(image_zip, image_dir)
    my_unzipper(text_zip, text_dir)

    print(f"Building {subset_name} JSONL index...")
    images = sorted(os.listdir(image_dir))
    texts  = sorted(os.listdir(text_dir))

    text_lookup = {t[:10]: t for t in texts}
    pairs = []

    for img in images:
        key = img[:10]
        if key in text_lookup:
            with open(os.path.join(text_dir, text_lookup[key]), "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            pairs.append({
                "image": os.path.join(image_dir, img),
                "transcript": transcript
            })

    print(f"  Found {len(pairs)} matched pairs for {subset_name} set.")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in pairs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"{subset_name.capitalize()} JSONL ready at: {out_jsonl}")


work_dir = "/content/ocr_data"
os.makedirs(work_dir, exist_ok=True)
train_jsonl = os.path.join(work_dir, "train_ocr.jsonl")
val_jsonl   = os.path.join(work_dir, "val_ocr.jsonl")
build_jsonl(train_image_zip, train_text_zip, train_jsonl, "train")
build_jsonl(val_image_zip, val_text_zip, val_jsonl, "val")


def save_hyperparameters(output_dir, start_time, end_time, train_loss, val_loss):
    file_path = os.path.join(output_dir, "hyperparameters.txt")
    total_time = end_time - start_time
    start_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "w") as f:
        f.write(f"train_image_zip = {train_image_zip}\n")
        f.write(f"train_text_zip = {train_text_zip}\n")
        f.write(f"val_image_zip = {val_image_zip}\n")
        f.write(f"val_text_zip = {val_text_zip}\n\n")
        f.write(f"MODEL_ID = {MODEL_ID}\n")
        f.write(f"START_DIR = {START_DIR}\n")
        f.write(f"OUTPUT_DIR = {OUTPUT_DIR}\n")
        f.write(f"MAX_TOKENS = {MAX_TOKENS}\n\n")
        f.write(f"LORA_R = {LORA_R}\n")
        f.write(f"LORA_ALPHA = {LORA_ALPHA}\n")
        f.write(f"LORA_DROPOUT = {LORA_DROPOUT}\n")
        f.write(f"TARGET_MODULES = {TARGET_MODULES}\n\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"GRAD_ACCUM = {GRAD_ACCUM}\n")
        f.write(f"LR = {LR}\n")
        f.write(f"WEIGHT_DECAY = {WEIGHT_DECAY}\n\n")
        f.write(f"Training started: {start_str}\n")
        f.write(f"Training ended:   {end_str}\n")
        f.write(f"Total runtime:    {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
        f.write(f"Average training loss:   {train_loss:.6f}\n")
        f.write(f"Average validation loss: {val_loss:.6f}\n")


def collate_fn(batch, processor):
    images = [b["image"] for b in batch]
    transcripts = [b["transcript"] for b in batch]

    template = [
        {"role": "system", "content": "You are an OCR engine."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Transcribe the text exactly as it appears in the image. Do not write anything but the actual transcription!"}
        ]}
    ]

    system_user_text = processor.apply_chat_template(template, tokenize=False)
    mask_len = len(processor.tokenizer(system_user_text, add_special_tokens=False).input_ids)

    messages = [
        template + [{"role": "assistant", "content": transcript}]
        for transcript in transcripts
    ]
    text_inputs = [processor.apply_chat_template(m, tokenize=False) for m in messages]

    inputs = processor(
        images=images,
        text=text_inputs,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt"
    )

    labels = inputs["input_ids"].clone()
    labels[:, :mask_len] = -100
    inputs["labels"] = labels

    return inputs


def main():
    accelerator = Accelerator(mixed_precision="bf16")
    os.makedirs(OUTPUT_DIR, exist_ok=False)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if MODEL_ID[:15] == "Qwen/Qwen2.5-VL":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
    elif MODEL_ID[:13] == "Qwen/Qwen3-VL":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
    else:
        raise ValueError(f"This code currently doesn't support {MODEL_ID}. Extend it or chose another model.")
    model.gradient_checkpointing_enable()
    for n, p in model.named_parameters():
        if ("vision" in n) or ("image" in n) or ("pixel" in n):
            p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        bias="none",
        lora_dropout=LORA_DROPOUT,
    )

    if START_DIR is not None and os.path.exists(os.path.join(START_DIR, "adapter_model.safetensors")):
        print(f"Loading existing LoRA adapter from {START_DIR}")
        model = get_peft_model(model, lora_config)
        model.load_adapter(START_DIR, adapter_name="default", is_trainable=True)
    else:
        print("Starting from stock model (no prior adapter).")
        model = get_peft_model(model, lora_config)

    train_ds = load_dataset("json", data_files=TRAINSET_JSONL, split="train")
    val_ds = load_dataset("json", data_files=VALSET_JSONL, split="train")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, processor))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    start_time = time.time()

    model.train()
    total_train_loss = 0.0
    num_train_steps = 0
    with tqdm(train_loader, desc=f"Training") as pbar:
        for step, batch in enumerate(pbar):
            try:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                total_train_loss += loss.detach().float()
                num_train_steps += 1
                if (step + 1) % GRAD_ACCUM == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                pbar.set_postfix(loss=float(accelerator.gather_for_metrics(loss.detach()).mean().item()))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM encountered in training batch {step}. Skipping batch.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise
    avg_train_loss = (accelerator.gather_for_metrics(total_train_loss).mean() / num_train_steps).item()

    model.eval()
    total_val_loss = 0.0
    num_val_steps = 0
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Computing validation loss") as vbar:
            for step, batch in enumerate(vbar):
                try:
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_val_loss += loss.detach().float()
                    num_val_steps += 1
                    vbar.set_postfix(loss=float(accelerator.gather_for_metrics(loss.detach()).mean().item()))
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f" OOM encountered in validation batch {step}. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
    avg_val_loss = (accelerator.gather_for_metrics(total_val_loss).mean() / num_val_steps).item()

    model.save_pretrained(OUTPUT_DIR)
    end_time = time.time()
    save_hyperparameters(OUTPUT_DIR, start_time, end_time, avg_train_loss, avg_val_loss)
    print("Training complete. Adapter/saved to", OUTPUT_DIR)


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    START_DIR = "" # Directory of the LoRA adapter to start from. Leave blank to start from stock
    OUTPUT_DIR = "" # Directory where the new LoRA adapter should be saved
    TRAINSET_JSONL = "/content/ocr_data/train_ocr.jsonl"
    VALSET_JSONL = "/content/ocr_data/val_ocr.jsonl"
    DEVICE = "cuda"
    MAX_TOKENS = 1024*6 # Choose large enough for the longest paragraphs in your dataset to fit
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    BATCH_SIZE = 4 # Choose as large as your GPU allows without too many OOMs
    GRAD_ACCUM = int(64/BATCH_SIZE)
    LR = 1e-6
    WEIGHT_DECAY = 0.0
    main()
