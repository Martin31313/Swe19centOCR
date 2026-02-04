# This is code for running inference with our models using the vLLM inference
# engine. It takes a few minutes to start, but once it gets going, the inference
# is really fast. If you are planning to run inference on less than 100 images,
# we recommend using the basic Hugging Face Transformers code available on the
# model card of the Hugging Face repo (huggingface.co/J0hanski/Swe19centOCR-8B)
# instead.

# The function takes a zip containing png-files as input, and outputs a zip
# containing txt-files with the corresponding base names, each containing the
# OCR prediction of the image.

#It doesn't run properly with torch 2.9.0
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

!pip uninstall transformers -y
!pip install transformers==4.57.3

!pip install vllm==0.11.1 -q

import os
import random
import json
import time
import zipfile
import tempfile
import shutil
import torch
from dataclasses import asdict
from PIL import Image
from argparse import Namespace
from vllm import LLM, EngineArgs, SamplingParams

def main(args):
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(args.input_zip, "r") as zip_ref:
        all_members = [m for m in zip_ref.namelist() if m.lower().endswith(".png")]
        if not all_members:
            raise ValueError(f"No .png files found inside {args.input_zip}")
        for member in all_members:
            zip_ref.extract(member, temp_dir)

    image_paths = [os.path.normpath(os.path.join(temp_dir, m)) for m in all_members]
    print(f"Extracted {len(image_paths)} images from zip -> '{temp_dir}'")

    os.makedirs(args.out_dir, exist_ok=True)
    out_args_path = os.path.join(args.out_dir, f"{args.out_zip_name}.txt")
    with open(out_args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved run arguments to '{out_args_path}'")

    engine_args = EngineArgs(
        model=args.model_dir,
        max_model_len=args.max_model_len,
        mm_processor_kwargs={"min_pixels": 784, "max_pixels": 2007040},
        limit_mm_per_prompt={"image": 1},
    )
    llm = LLM(**asdict(engine_args))
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_out_tokens)

    prompt = ("<|im_start|>system\nYou are an OCR engine.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{args.prompt}<|im_end|>\n"
              "<|im_start|>assistant\n"
    )

    inputs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    out_zip_path = os.path.join(args.out_dir, args.out_zip_name)
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for img_path, o in zip(image_paths, outputs):
            generated_text = o.outputs[0].text
            base = os.path.splitext(os.path.basename(img_path))[0]
            zipf.writestr(base+".txt", generated_text)
    print(f"Saved outputs to '{out_zip_path}'")

    for inp in inputs:
        inp["multi_modal_data"]["image"].close()
    del inputs, outputs
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_time = time.time() - start_time
    with open(out_args_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n# Total inference time: {total_time:.2f} seconds\n")
        f.write(f"# GPU used: {torch.cuda.get_device_name(0)}\n")


if __name__ == "__main__":
    args = Namespace(
        model_dir="J0hanski/Swe19centOCR-8B", # or J0hanski/Swe19centOCR-2B
        max_model_len=4096,
        max_out_tokens=1280,
        input_zip_path="", # Path to a zip-file containing pngs
        out_dir="", # Path to directory where the output is to be saved
        out_zip_name="", # Don't forget ".zip" at the end
        prompt="Transcribe the text exactly as it appears in the image. "
              +"Do not write anything but the actual transcription!"
        # The models were fine-tuned using this exact prompt.
        # We recommend not to change it.
    )
    main(args)
