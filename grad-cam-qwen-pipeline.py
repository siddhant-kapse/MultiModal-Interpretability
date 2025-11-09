import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import gc

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================================
# CONFIGURATION
# =========================================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QUESTION = "Is this meme hateful? Answer YES or NO."

# Pipeline Paths
DATA_ROOT = "data"
CAPTION_DIR = os.path.join(DATA_ROOT, "captions")
IMAGE_ROOT = os.path.join(DATA_ROOT, "memes")
OUTPUT_DIR = "gradcam_outputs_pipeline"

LANGUAGES = ["en", "de", "es", "hi", "zh"]
TARGETS = ["YES", "NO"]

MIN_PIXELS = 224 * 224
MAX_PIXELS = 256 * 256

# =========================================
# CORE FUNCTIONS (EXACTLY AS WORKING SCRIPT)
# =========================================

def qwen_reshape_transform(tensor):
    # Based on your logs, the grid is 18x18
    H, W = 18, 18
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.size(-1) != H * W:
         tensor = tensor.permute(0, 2, 1)
    return tensor.reshape(tensor.size(0), -1, H, W)

def find_best_visual_block(model, enc, processor, target_token_id):
    image_tensor = enc["pixel_values"]
    text_ids = enc["input_ids"]
    best_idx = None
    best_score = -float("inf")
    handles = []
    activations, gradients = {}, {}

    def save_act(name):
        def hook(module, inp, out): activations[name] = out
        return hook
    def save_grad(name):
        def hook(module, grad_in, grad_out): gradients[name] = grad_out[0]
        return hook

    # Optimization: scan last 15 blocks only
    start_block = max(0, len(model.model.visual.blocks) - 15)
    for i in range(start_block, len(model.model.visual.blocks)):
        blk = model.model.visual.blocks[i]
        name = f"block_{i}"
        handles.append(blk.attn.register_forward_hook(save_act(name)))
        handles.append(blk.attn.register_full_backward_hook(save_grad(name)))

    with torch.enable_grad():
        out = model(input_ids=text_ids, pixel_values=image_tensor, 
                   image_grid_thw=enc.get("image_grid_thw"), use_cache=False)
        out.logits[:, -1, target_token_id].backward(retain_graph=True)

    for name in activations.keys():
        score = gradients[name].abs().mean().item()
        if score > best_score:
            best_score = score
            best_idx = int(name.split("_")[1])
    for h in handles: h.remove()
    return best_idx

def build_prompt_and_inputs(processor, image_pil, device):
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=[image_pil], return_tensors="pt")
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    return enc

class Wrapper(torch.nn.Module):
    def __init__(self, model, enc_fixed):
        super().__init__()
        self.m = model
        self.enc_fixed = enc_fixed
    def forward(self, ignored_input):
        return self.m(
            input_ids=self.enc_fixed["input_ids"],
            attention_mask=self.enc_fixed.get("attention_mask"),
            pixel_values=self.enc_fixed["pixel_values"],
            image_grid_thw=self.enc_fixed.get("image_grid_thw"),
            use_cache=False,
        ).logits[:, -1, :]

# =========================================
# PIPELINE LOOP
# =========================================

def process_image(model, processor, image_path, meme_id, output_dir):
    device = model.device
    try:
        image_pil = Image.open(image_path).convert("RGB")
    except:
        print(f"⚠️ Error reading {image_path}")
        return

    enc = build_prompt_and_inputs(processor, image_pil, device)
    image_rgb_np = np.asarray(image_pil).astype(np.float32) / 255.0
    wrapped = Wrapper(model, enc)
    
    # --- FIX: Use exactly the dummy tensor shape from the working script ---
    dummy_tensor = torch.zeros(1, 3, 512, 512, device=device)

    for target_text in TARGETS:
        out_path = os.path.join(output_dir, f"{meme_id}_{target_text}.jpg")
        if os.path.exists(out_path): continue

        target_id = processor.tokenizer(" " + target_text).input_ids[-1]

        try:
            best_idx = find_best_visual_block(model, enc, processor, target_id)
            target_layers = [model.model.visual.blocks[best_idx].attn]

            cam = GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=qwen_reshape_transform)
            grayscale_cam = cam(input_tensor=dummy_tensor, 
                                targets=[ClassifierOutputTarget(target_id)],
                                aug_smooth=False, eigen_smooth=False)[0, :]

            visualization = show_cam_on_image(image_rgb_np, grayscale_cam, use_rgb=True, image_weight=0.5)
            plt.imsave(out_path, visualization)
            
        except RuntimeError as e:
            if "shape" in str(e) and "input_tensor" not in str(e):
                 # This catches the reshape 18x18 mismatch error
                 print(f"⚠️ Skipping {meme_id} [{target_text}]: Grid mismatch (not 18x18).")
            else:
                 print(f"❌ Runtime Error on {meme_id} [{target_text}]: {e}")
        except Exception as e:
            print(f"❌ Error on {meme_id} [{target_text}]: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

def main():
    print("--- Loading Model ---")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="cuda"
        )
    except:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
        )
    model.eval()

    for lang in LANGUAGES:
        print(f"\n=== Language: {lang.upper()} ===")
        lang_out = os.path.join(OUTPUT_DIR, lang)
        os.makedirs(lang_out, exist_ok=True)

        caption_path = os.path.join(CAPTION_DIR, f"{lang}.csv")
        if not os.path.exists(caption_path): continue
        df = pd.read_csv(caption_path)
        meme_ids = df["Meme ID"].astype(str).unique()
        lang_img_root = os.path.join(IMAGE_ROOT, lang)

        for meme_id in tqdm(meme_ids, desc=f"{lang}"):
            found = None
            for root, _, files in os.walk(lang_img_root):
                for f in files:
                    if f.startswith(meme_id) and f.lower().endswith(('.jpg','.png','.jpeg')):
                        if os.path.splitext(f)[0] == meme_id:
                            found = os.path.join(root, f)
                            break
                if found: break
            
            if found:
                process_image(model, processor, found, meme_id, lang_out)

if __name__ == "__main__":
    main()