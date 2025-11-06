import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForImageTextToText
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ----------------
# Local imports
# ----------------
from vlm.inference.utils import process_translations, set_prompts, ANNOTATION_PATH, IMAGE_FOLDER, CAPTION_FOLDER

# ----------------
# Config
# ----------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "gradcam_outputs"
PROMPT_TEXT = "<|image_pad|> \n Is this meme hateful? Answer YES or NO.\nASSISTANT:"
TARGET_WORD = "YES"
LANGUAGES = ["en", "de", "es", "hi", "zh"]

# ----------------
# Grad-CAM Wrappers
# ----------------
class QwenVLWrapper(torch.nn.Module):
    def __init__(self, model, processor, prompt):
        super().__init__()
        self.model = model
        self.processor = processor
        self.prompt = prompt

    def forward(self, pixel_values, image_sizes=None):
        enc = self.processor(
            text=[self.prompt],
            images=[self._last_pil_image],
            return_tensors="pt"
        )
        enc = {k: v.to(self.model.device if hasattr(self.model, "device") else v.device) for k, v in enc.items()}
        enc["pixel_values"] = pixel_values
        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc.get("image_grid_thw", None),
            use_cache=False,
        )
        return outputs.logits[:, -1, :]


class QwenTarget:
    def __init__(self, processor, target_word):
        tok = processor.tokenizer
        self.targets = [
            ClassifierOutputTarget(tok(f" {target_word}").input_ids[-1]),
            ClassifierOutputTarget(tok(target_word).input_ids[-1]),
        ]
        print(f"Grad-CAM target set to token ids for '{target_word}'")

    def __call__(self, model_output):
        try:
            return self.targets[0](model_output)
        except Exception:
            return self.targets[1](model_output)

# ----------------
# Grad-CAM for one meme
# ----------------
def run_gradcam_on_meme(model, processor, image_pil, meme_id, language):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_pil = image_pil.resize((224, 224))

    enc = processor(text=[PROMPT_TEXT], images=[image_pil], return_tensors="pt")
    print("image tokens:", (enc.input_ids == processor.tokenizer.convert_tokens_to_ids("<image>")))


    print("Input keys:", enc.keys())
    print("pixel_values shape:", enc.pixel_values.shape)
    print("input_ids shape:", enc.input_ids.shape)


    proc_inputs = processor(
        text=[PROMPT_TEXT],
        images=[image_pil],
        return_tensors="pt"
    ).to(device)
    
    pixel_values = proc_inputs.pixel_values
    #pixel_values = processor(text=[""], images=[image_pil], return_tensors="pt").pixel_values

    print(f"DEBUG: pixel_values shape is: {pixel_values.shape}")
    # pixel_values = pixel_values.to(device)
    if getattr(model.config, "torch_dtype", None) == torch.bfloat16:
        pixel_values = pixel_values.to(torch.bfloat16)
    elif getattr(model.config, "torch_dtype", None) == torch.float16:
        pixel_values = pixel_values.to(torch.float16)

    pixel_values = pixel_values.to(device)

    image_rgb_np = np.asarray(image_pil).astype(np.float32) / 255.0
    target_layer = model.model.visual.blocks[-4].attn

    wrapped = QwenVLWrapper(model, processor, PROMPT_TEXT).eval()
    wrapped._last_pil_image = image_pil
    #wrapped = wrapped.to(device)

    for p in target_layer.parameters():
        p.requires_grad_(True)

    cam = GradCAM(model=wrapped, target_layers=[target_layer])
    targeter = QwenTarget(processor, TARGET_WORD)
    grayscale_cam = cam(input_tensor=pixel_values, targets=[targeter], aug_smooth=True)
    grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(image_rgb_np, grayscale_cam, use_rgb=True, image_weight=0.5)
    os.makedirs(os.path.join(OUTPUT_DIR, language), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, language, f"{meme_id}_gradcam.png")
    plt.imsave(out_path, visualization)
    print(f"[{language}] Saved Grad-CAM: {out_path}")

# ----------------
# Main Pipeline
# ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + processor once
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype="auto", device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)

    for language in LANGUAGES:
        print(f"\n--- Processing {language.upper()} ---")
        caption_file = os.path.join(CAPTION_FOLDER, f"{language}.csv")
        if not os.path.exists(caption_file):
            print(f"❌ Caption file missing for {language}, skipping.")
            continue

        df = pd.read_csv(caption_file)
        df["Meme ID"] = df["Meme ID"].astype(str)

        image_dir = os.path.join(IMAGE_FOLDER, language)
        if not os.path.exists(image_dir):
            print(f"❌ Image folder missing for {language}, skipping.")
            continue

        # Iterate over all images that exist in caption file
        meme_ids = df["Meme ID"].unique()
        for meme_id in tqdm(meme_ids, desc=f"{language} memes"):
            found_path = None
            # Walk subfolders (e.g., Advicejew/222.jpg)
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.startswith(str(meme_id)) and file.lower().endswith((".jpg", ".png", ".jpeg")):
                        found_path = os.path.join(root, file)
                        break
                if found_path:
                    break

            if not found_path:
                print(f"⚠️ Missing image for {language} meme {meme_id}")
                continue

            try:
                image_pil = Image.open(found_path).convert("RGB")
                print(f"Processing {language} meme {meme_id}")
                run_gradcam_on_meme(model, processor, image_pil, meme_id, language)
            except Exception as e:
                print(f"⚠️ Error on {language} meme {meme_id}: {e}")
                continue

if __name__ == "__main__":
    main()
