import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
import gc

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# Switched back to GradCAM for stability, you can try LayerCAM again once this works
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMPlusPlus, XGradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------------
# Config
# ----------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_CSV = "data/captions/en.csv"
IMAGE_DIR = r"data/memes/en/Nice-White-Girl"
SAMPLE_MEME_ID = 88
QUESTION = "Is this meme hateful? Answer YES or NO."
OUT_PATH = "gradcam_qwen2p5vl_margin_yes_no_k5.jpg"

MIN_PIXELS = 224 * 224
MAX_PIXELS = 256 * 256

# ----------------
# Custom Classes & Functions
# ----------------

class SummedLogitsTarget:
    """
    A target class that sums the logits of a list of token IDs.
    This is used when a word (like "YES") might have multiple
    valid tokenizations.
    """
    def __init__(self, token_ids):
        # Ensure token_ids is always a list
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self.token_ids = token_ids
    
    def __call__(self, model_output):
        # Ensure 2D [Batch, Vocab]
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
            
        # Get all relevant logits and sum them along the vocab dimension
        # This turns the list of logits into a single scalar score
        score = model_output[:, self.token_ids].sum(dim=-1)
        return score

class BinaryMarginTarget:
    def __init__(self, yes_ids, no_ids):
        # Store the lists of IDs
        self.yes_ids = yes_ids
        self.no_ids = no_ids
        
    def __call__(self, model_output):
        # Ensure 2D [Batch, Vocab]
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
            
        # Sum the logits for all "YES" token variations
        yes_score = model_output[:, self.yes_ids].sum(dim=-1)
        
        # Sum the logits for all "NO" token variations
        no_score = model_output[:, self.no_ids].sum(dim=-1)
        
        # The result is a single scalar (per batch item)
        return yes_score - no_score

def qwen_reshape_transform(tensor, height, width):
    if tensor.dim() == 2: tensor = tensor.unsqueeze(0)
    if tensor.size(-1) != height * width: tensor = tensor.permute(0, 2, 1)
    return tensor.reshape(tensor.size(0), -1, height, width)

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

def find_best_visual_block(model, enc, target, top_k=3):
    """
    Finds the top-K visual blocks based on mean gradient magnitude.
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

    # Scan last 15 blocks
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
            
            # Use the target callable (e.g., margin_target)
            logits = out.logits[:, -1, :]
            score_to_backprop = target(logits) 
            score_to_backprop.backward(retain_graph=True)

        for name in activations:
            if name in gradients: # Ensure gradient was captured
                score = gradients[name].abs().mean().item()
                scores.append((score, int(name.split("_")[1])))
            
    except Exception as e:
        print(f"⚠️ Block scan failed: {e}. Defaulting to last block.")
        # Fallback to just the last block
        scores = [(1.0, len(model.model.visual.blocks) - 1)] 
    finally:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        gc.collect()

    if not scores:
        print("⚠️ No scores captured, defaulting to last block.")
        return [len(model.model.visual.blocks) - 1]

    # Sort by score (highest first) and get top K indices
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Ensure we don't request more blocks than we found
    actual_k = min(top_k, len(scores))
    
    best_indices = [idx for score, idx in scores[:actual_k]]
    best_scores = [score for score, idx in scores[:actual_k]]
    
    print(f"Top-{actual_k} blocks: {best_indices} (scores: {[f'{s:.6f}' for s in best_scores]})")
    return best_indices

# ----------------
# Main
# ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Loading Model ---")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    # Load Data
    df = pd.read_csv(DATASET_CSV)
    row = df[df["Meme ID"] == SAMPLE_MEME_ID].iloc[0]
    img_path = os.path.join(IMAGE_DIR, f"{SAMPLE_MEME_ID}.jpg")
    image_pil = Image.open(img_path).convert("RGB")
    print(f"Loaded {SAMPLE_MEME_ID}: {row['Translation']}")

    # Build Inputs
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=[image_pil], return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    grid_h, grid_w = enc["image_grid_thw"][0, 1].item(), enc["image_grid_thw"][0, 2].item()
    print(f"Grid: {grid_h}x{grid_w}")

    # --- CHANGE 1: Define all 3 targets ---
# 1. Define your ID lists (this is now correct)
    yes_id_list = [
        processor.tokenizer(" YES").input_ids[-1],
        processor.tokenizer("YES").input_ids[-1],
    ]
    no_id_list = [
        processor.tokenizer(" NO").input_ids[-1],
        processor.tokenizer("NO").input_ids[-1],
    ]

    # 2. Use the new target classes
    
    # Target 1: YES-only
    yes_target = SummedLogitsTarget(yes_id_list)
    
    # Target 2: NO-only
    no_target = SummedLogitsTarget(no_id_list)
    
    # Target 3: Margin
    margin_target = BinaryMarginTarget(yes_id_list, no_id_list)

    # 3. Find best block (your code for this is already correct)
    best_indices = find_best_visual_block(model, enc, margin_target, top_k=3)
    target_layers = [model.model.visual.blocks[i].attn for i in best_indices]

    wrapped = Wrapper(model, enc)
    dynamic_reshape = partial(qwen_reshape_transform, height=grid_h, width=grid_w)

    # Initialize all 3 maps to None
    grayscale_cam_yes = None
    grayscale_cam_no = None
    grayscale_cam_margin = None

    try:
        # Use the same CAM method and layers for all 3
        with GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=dynamic_reshape) as cam:
            
            dummy_tensor = torch.zeros(1, 3, 512, 512, device=device)
            
            # --- CHANGE 2: Run CAM three times ---
            print("Running CAM for YES target...")
            grayscale_cam_yes = cam(input_tensor=dummy_tensor,
                                    targets=[yes_target],
                                    aug_smooth=False, eigen_smooth=False)[0, :]
            
            print("Running CAM for NO target...")
            grayscale_cam_no = cam(input_tensor=dummy_tensor,
                                   targets=[no_target],
                                   aug_smooth=False, eigen_smooth=False)[0, :]

            print("Running CAM for MARGIN target...")
            grayscale_cam_margin = cam(input_tensor=dummy_tensor,
                                       targets=[margin_target],
                                       aug_smooth=False, eigen_smooth=False)[0, :]
            
    except Exception as e:
        print(f"❌ CAM generation failed: {e}")
        import traceback
        traceback.print_exc()

    # --- CHANGE 3: Visualize all 3 maps ---
    img_np = np.asarray(image_pil).astype(np.float32) / 255.0

    # Visualize YES
    if grayscale_cam_yes is not None:
        vis_yes = show_cam_on_image(img_np, grayscale_cam_yes, use_rgb=True, image_weight=0.6)
        path_yes = f"gradcam_{SAMPLE_MEME_ID}_TARGET_YES.jpg"
        plt.imsave(path_yes, vis_yes)
        print(f"SAVED: {os.path.abspath(path_yes)}")
    else:
        print("❌ Skipping YES save because CAM failed.")

    # Visualize NO
    if grayscale_cam_no is not None:
        vis_no = show_cam_on_image(img_np, grayscale_cam_no, use_rgb=True, image_weight=0.6)
        path_no = f"gradcam_{SAMPLE_MEME_ID}_TARGET_NO.jpg"
        plt.imsave(path_no, vis_no)
        print(f"SAVED: {os.path.abspath(path_no)}")
    else:
        print("❌ Skipping NO save because CAM failed.")

    # Visualize MARGIN
    if grayscale_cam_margin is not None:
        vis_margin = show_cam_on_image(img_np, grayscale_cam_margin, use_rgb=True, image_weight=0.6)
        path_margin = f"gradcam_{SAMPLE_MEME_ID}_TARGET_MARGIN.jpg"
        plt.imsave(path_margin, vis_margin)
        print(f"SAVED: {os.path.abspath(path_margin)}")
    else:
        print("❌ Skipping MARGIN save because CAM failed.")

if __name__ == "__main__":
    main()