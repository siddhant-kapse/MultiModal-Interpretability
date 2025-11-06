import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ----------------
# Config
# ----------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_CSV = "data/captions/en.csv"
IMAGE_DIR = r"data/memes/en/Advicejew"
SAMPLE_MEME_ID = 222
PROMPT_TEXT = "<|image_pad|> \n Is this meme hateful? Answer YES or NO.\nASSISTANT:"  # ✅ FIXED
TARGET_WORD = "YES"

# ----------------
# Data loading
# ----------------
def load_data(meme_id):
    df = pd.read_csv(DATASET_CSV)
    row = df[df["Meme ID"] == meme_id].iloc[0]
    meme_text = row["Translation"]
    image_filename = "222.jpg"  # adapt if needed
    image_path = os.path.join(IMAGE_DIR, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    img = Image.open(image_path).convert("RGB")
    print(f"Loaded Meme {meme_id}: {meme_text}")
    return img, meme_text


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
            images=[self._last_pil_image],  # see below
            return_tensors="pt"
        )
        enc = {k: v.to(self.model.device if hasattr(self.model, "device") else v.device) for k, v in enc.items()}
        # Overwrite pixel_values to use the one Grad-CAM perturbs
        enc["pixel_values"] = pixel_values
        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc.get("image_grid_thw", None),
            use_cache=False,
        )
        return outputs.logits[:, -1, :]


# ----------------
# Target: YES token id (with and without leading space)
# ----------------
class QwenTarget:
    def __init__(self, processor, target_word):
        tok = processor.tokenizer
        self.targets = [
            ClassifierOutputTarget(tok(f" {target_word}").input_ids[-1]),
            ClassifierOutputTarget(tok(target_word).input_ids[-1]),
        ]
        print(f"Grad-CAM target set to token ids for '{target_word}' (with/without space)")

    def __call__(self, model_output):
        # try with-leading-space first
        try:
            return self.targets[0](model_output)
        except Exception:
            return self.targets[1](model_output)

# ----------------
# Main
# ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model/processor
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype="auto",          
        device_map="cuda",
    )
    #model = model.to(device)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)

    print("Image token used by processor:", processor.image_token)


    # Load one meme
    image_pil, _ = load_data(SAMPLE_MEME_ID)

    enc = processor(text=[PROMPT_TEXT], images=[image_pil], return_tensors="pt")
    print("image tokens:", (enc.input_ids == processor.tokenizer.convert_tokens_to_ids("<image>")))


    print("Input keys:", enc.keys())
    print("pixel_values shape:", enc.pixel_values.shape)
    print("input_ids shape:", enc.input_ids.shape)
        # Preprocess image via processor to get pixel_values AND image_grid_thw computed internally
    # FIX: Always pass images via processor, not manual transforms
    proc_inputs = processor(
        text=[PROMPT_TEXT],
        images=[image_pil],
        return_tensors="pt"
    ).to(device)

    # Extract pixel_values only; Grad-CAM will call wrapper with this tensor
    #pixel_values = proc_inputs.pixel_values
    pixel_values = processor(text=[""], images=[image_pil], return_tensors="pt").pixel_values

    # Match dtype with model config
    if getattr(model.config, "torch_dtype", None) == torch.bfloat16:
        pixel_values = pixel_values.to(torch.bfloat16)
    elif getattr(model.config, "torch_dtype", None) == torch.float16:
        pixel_values = pixel_values.to(torch.float16)
    #pixel_values = pixel_values.to(device)

    # Prepare a 0–1 RGB numpy image for overlay
    image_rgb_np = np.asarray(image_pil).astype(np.float32) / 255.0

    # FIX: choose the correct visual target layer: last visual attention block
    # Inspect printed model: model.model.visual.blocks[-1].attn is a good choice
    target_layer = model.model.visual.blocks[-1].attn


    # Wrap model to return next-token logits
    wrapped = QwenVLWrapper(model, processor, PROMPT_TEXT).eval()
    wrapped._last_pil_image = image_pil

    # Enable gradients on target layer
    for p in target_layer.parameters():
        p.requires_grad_(True)

    cam = GradCAM(model=wrapped, target_layers=[target_layer])
    targeter = QwenTarget(processor, TARGET_WORD)

    # Compute Grad-CAM
    grayscale_cam = cam(input_tensor=pixel_values, targets=[targeter], aug_smooth=True)
    
    grayscale_cam = grayscale_cam[0]

    # Overlay
    visualization = show_cam_on_image(image_rgb_np, grayscale_cam, use_rgb=True, image_weight=0.5)
    out_path = "gradcam_qwen2p5vl_yes.jpg"
    plt.imsave(out_path, visualization)
    print(f"SAVED: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
