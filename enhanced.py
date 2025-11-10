import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import gc
from functools import partial

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pytorch_grad_cam import XGradCAM  # <-- Upgraded
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
OUTPUT_DIR = "gradcam_outputs_pipeline_v2" # <-- New output dir

LANGUAGES = ["en"]
# LANGUAGES = ["en", "de", "es", "hi", "zh"]

MIN_PIXELS = 224 * 224
MAX_PIXELS = 256 * 256

# =========================================
# NEW: UPGRADED TARGET CLASSES
# =========================================

class SummedLogitsTarget:
    """
    A target class that sums the logits of a list of token IDs.
    """
    def __init__(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self.token_ids = token_ids
    
    def __call__(self, model_output):
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        # Sum logits for all specified token IDs
        score = model_output[:, self.token_ids].sum(dim=-1)
        return score

class BinaryMarginTarget:
    """
    Calculates the margin (YES_score - NO_score) by summing
    all valid token IDs for each class.
    """
    def __init__(self, yes_ids, no_ids):
        self.yes_ids = yes_ids
        self.no_ids = no_ids
        
    def __call__(self, model_output):
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
            
        yes_score = model_output[:, self.yes_ids].sum(dim=-1)
        no_score = model_output[:, self.no_ids].sum(dim=-1)
        
        return yes_score - no_score

# =========================================
# NEW: UPGRADED CORE FUNCTIONS
# =========================================

def qwen_reshape_transform_dynamic(tensor, height, width):
    """
    Handles dynamic grid sizes and the [CLS] token for ViT blocks.
    Input tensor from hook is (B, N, C).
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    
    expected_tokens = height * width
    
    # Check for [CLS] token: (B, N, C) where N = H*W + 1
    if tensor.size(1) == expected_tokens + 1:
        tensor = tensor[:, 1:, :]  # Drop the [CLS] token
    
    # Now tensor is (B, H*W, C). Permute to (B, C, H*W)
    tensor = tensor.permute(0, 2, 1)
    
    # Reshape to (B, C, H, W)
    return tensor.reshape(tensor.size(0), tensor.size(1), height, width)


def find_best_visual_block(model, enc, target, top_k=3):
    """
    Finds the top-K visual blocks based on mean gradient magnitude
    using the provided target (e.g., margin_target).
    """
    image_tensor = enc["pixel_values"]
    text_ids = enc["input_ids"]
    scores = []
    handles, activations, gradients = [], {}, {}

    def save_act(name):
        def hook(m, i, o): activations[name] = o
        return hook
    def save_grad(name):
        def hook(m, gi, go): gradients[name] = go[0]
        return hook

    # Scan last 20 blocks
    start_block = max(0, len(model.model.visual.blocks) - 20)
    for i in range(start_block, len(model.model.visual.blocks)):
        blk = model.model.visual.blocks[i]
        name = f"block_{i}"
        handles.append(blk.attn.register_forward_hook(save_act(name)))
        handles.append(blk.attn.register_full_backward_hook(save_grad(name)))

    try:
        with torch.enable_grad():
            out = model(input_ids=text_ids, pixel_values=image_tensor, 
                        image_grid_thw=enc.get("image_grid_thw"), use_cache=False)
            
            logits = out.logits[:, -1, :]
            score_to_backprop = target(logits) 
            score_to_backprop.backward(retain_graph=True)

        for name in activations:
            if name in gradients: 
                score = gradients[name].abs().mean().item()
                scores.append((score, int(name.split("_")[1])))
            
    except Exception as e:
        print(f"⚠️ Block scan failed: {e}. Defaulting to last block.")
        scores = [(1.0, len(model.model.visual.blocks) - 1)] 
    finally:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        gc.collect()

    if not scores:
        print("⚠️ No scores captured, defaulting to last block.")
        return [len(model.model.visual.blocks) - 1]

    scores.sort(key=lambda x: x[0], reverse=True)
    actual_k = min(top_k, len(scores))
    best_indices = [idx for score, idx in scores[:actual_k]]
    
    # print(f"Top-{actual_k} blocks: {best_indices}") # Optional: for debugging
    return best_indices

# --- These functions are unchanged ---
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
# NEW: UPGRADED PIPELINE LOOP
# =========================================

def process_image(model, processor, image_path, meme_id, output_dir):
    device = model.device
    try:
        image_pil = Image.open(image_path).convert("RGB")
    except:
        print(f"⚠️ Error reading {image_path}")
        return

    try:
        enc = build_prompt_and_inputs(processor, image_pil, device)
        image_rgb_np = np.asarray(image_pil).astype(np.float32) / 255.0
        wrapped = Wrapper(model, enc)
        
        dummy_tensor = torch.zeros(1, 3, 512, 512, device=device)

        # --- 1. Get DYNAMIC grid size ---
        # This is CRITICAL for non-English images
        grid_h = enc["image_grid_thw"][0, 1].item()
        grid_w = enc["image_grid_thw"][0, 2].item()
        
        # --- 2. Create dynamic reshape function for THIS image ---
        dynamic_reshape = partial(qwen_reshape_transform_dynamic, height=grid_h, width=grid_w)

        # --- 3. Define all targets ---
        yes_id_list = [
            processor.tokenizer(" YES").input_ids[-1],
            processor.tokenizer("YES").input_ids[-1],
        ]
        no_id_list = [
            processor.tokenizer(" NO").input_ids[-1],
            processor.tokenizer("NO").input_ids[-1],
        ]
        
        yes_target = SummedLogitsTarget(yes_id_list)
        no_target = SummedLogitsTarget(no_id_list)
        margin_target = BinaryMarginTarget(yes_id_list, no_id_list)

        # --- 4. Find best layers ONCE using margin target ---
        best_indices = find_best_visual_block(model, enc, margin_target, top_k=3)
        target_layers = [model.model.visual.blocks[i].attn for i in best_indices]

        # --- 5. Run CAM for all 3 targets ---
        grayscale_cam_yes = None
        grayscale_cam_no = None
        grayscale_cam_margin = None
        
        with XGradCAM(model=wrapped, target_layers=target_layers, reshape_transform=dynamic_reshape) as cam:
            
            # Target 1: YES
            out_path_yes = os.path.join(output_dir, f"{meme_id}_YES.jpg")
            if not os.path.exists(out_path_yes):
                grayscale_cam_yes = cam(input_tensor=dummy_tensor, 
                                        targets=[yes_target], 
                                        aug_smooth=False, eigen_smooth=False)[0, :]
                vis = show_cam_on_image(image_rgb_np, grayscale_cam_yes, use_rgb=True, image_weight=0.5)
                plt.imsave(out_path_yes, vis)
            
            # Target 2: NO
            out_path_no = os.path.join(output_dir, f"{meme_id}_NO.jpg")
            if not os.path.exists(out_path_no):
                grayscale_cam_no = cam(input_tensor=dummy_tensor, 
                                       targets=[no_target], 
                                       aug_smooth=False, eigen_smooth=False)[0, :]
                vis = show_cam_on_image(image_rgb_np, grayscale_cam_no, use_rgb=True, image_weight=0.5)
                plt.imsave(out_path_no, vis)

            # Target 3: MARGIN
            out_path_margin = os.path.join(output_dir, f"{meme_id}_MARGIN.jpg")
            if not os.path.exists(out_path_margin):
                grayscale_cam_margin = cam(input_tensor=dummy_tensor, 
                                           targets=[margin_target], 
                                           aug_smooth=False, eigen_smooth=False)[0, :]
                vis = show_cam_on_image(image_rgb_np, grayscale_cam_margin, use_rgb=True, image_weight=0.5)
                plt.imsave(out_path_margin, vis)

    except RuntimeError as e:
        if "shape" in str(e):
            print(f"⚠️ Skipping {meme_id}: Grid mismatch error. H={grid_h}, W={grid_w}. Error: {e}")
        else:
            print(f"❌ Runtime Error on {meme_id}: {e}")
    except Exception as e:
        print(f"❌ Error on {meme_id}: {e}")
    finally:
        # Clear cache after each image
        torch.cuda.empty_cache()
        gc.collect()

# =========================================
# MAIN (Unchanged, but points to new OUTPUT_DIR)
# =========================================

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
        lang_out = os.path.join(OUTPUT_DIR, lang) # <-- Will save to v2 folder
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