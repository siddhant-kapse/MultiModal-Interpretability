import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
import gc
import argparse
import sys
from tqdm import tqdm # Added for a nice progress bar

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------------
# Config - These are now defaults, can be overridden by argparse
# ----------------
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DATA_DIR = "data/"
DEFAULT_OUTPUT_DIR = "gradcam_failures"

# --- Prompting Constants ---
LANGUAGES = ["en", "de", "es", "hi", "zh"]
MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "hi": "IN",
    "zh": "CN"
}

SIMPLE_QUESTION = "Is this meme hateful? Answer YES or NO."

MIN_PIXELS = 224 * 224
MAX_PIXELS = 256 * 256

# ----------------
# Custom Classes & Functions
# (THESE ARE UNCHANGED)
# ----------------

class SummedLogitsTarget:
    """
    A target class that sums the logits of a list of token IDs.
    This is used when a word (like "YES") might have multiple
    valid tokenizations.
    """
    def __init__(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self.token_ids = token_ids
    
    def __call__(self, model_output):
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        score = model_output[:, self.token_ids].sum(dim=-1)
        return score

class BinaryMarginTarget:
    def __init__(self, yes_ids, no_ids):
        self.yes_ids = yes_ids
        self.no_ids = no_ids
        
    def __call__(self, model_output):
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        yes_score = model_output[:, self.yes_ids].sum(dim=-1)
        no_score = model_output[:, self.no_ids].sum(dim=-1)
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
        print(f"    ⚠️ Block scan failed for meme {enc.get('meme_id', 'unknown')}: {e}. Defaulting to last block.")
        scores = [(1.0, len(model.model.visual.blocks) - 1)] 
    finally:
        for h in handles: h.remove()
        torch.cuda.empty_cache()
        gc.collect()

    if not scores:
        return [len(model.model.visual.blocks) - 1]

    scores.sort(key=lambda x: x[0], reverse=True)
    actual_k = min(top_k, len(scores))
    best_indices = [idx for score, idx in scores[:actual_k]]
    return best_indices

# ----------------
# NEW: Processing function for a single meme
# ----------------
def process_single_meme(model, processor, meme_id, args):
    """
    Runs the full Grad-CAM analysis for a single meme_id.
    Assumes model and processor are already loaded.
    """
    device = model.device
    print(f"  Processing Meme ID: {meme_id}")
    
    try:
        # --- 1. Load Data ---
        dataset_csv = os.path.join(args.data_dir, 'captions', f"{args.lang}.csv")
        df_captions = pd.read_csv(dataset_csv)
        row = df_captions[df_captions["Meme ID"] == meme_id].iloc[0]
        meme_text = row['Template Name'] 
        caption_text =  row['Translation']
        
        image_dir = os.path.join(args.data_dir, 'memes', args.lang)
        img_path = os.path.join(image_dir, f"{meme_id}.jpg")
        
        if not os.path.exists(img_path):
             image_dir_fallback = os.path.join(args.data_dir, 'memes', args.lang, meme_text.replace(" ", "-"))
             img_path = os.path.join(image_dir_fallback, f"{meme_id}.jpg")

        image_pil = Image.open(img_path).convert("RGB")

        # --- 2. Build Dynamic Prompt ---
        question = SIMPLE_QUESTION
        country = MAPPING[args.lang]

        text_prompt_1_content = question
        text_prompt_2_content = "" 

        if args.country_insertion:
            text_prompt_1_content = f"{SIMPLE_QUESTION} (in {country})"
        
        if args.caption:
            text_prompt_2_content = f"Caption inside the meme: {caption_text}"

        content_list = []
        content_list.append({"type": "text", "text": text_prompt_1_content})
        
        image_to_process = None
        if not args.unimodal:
            content_list.append({"type": "image"})
            image_to_process = [image_pil]
        
        if text_prompt_2_content:
            content_list.append({"type": "text", "text": text_prompt_2_content})

        conversation = [{"role": "user", "content": content_list}]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[prompt], images=image_to_process, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        enc['meme_id'] = meme_id # For error logging

        grid_h, grid_w = enc["image_grid_thw"][0, 1].item(), enc["image_grid_thw"][0, 2].item()

        # --- 3. Setup CAM ---
        yes_id_list = [processor.tokenizer(" YES").input_ids[-1], processor.tokenizer("YES").input_ids[-1]]
        no_id_list = [processor.tokenizer(" NO").input_ids[-1], processor.tokenizer("NO").input_ids[-1]]

        yes_target = SummedLogitsTarget(yes_id_list)
        no_target = SummedLogitsTarget(no_id_list)
        margin_target = BinaryMarginTarget(yes_id_list, no_id_list)

        best_indices = find_best_visual_block(model, enc, margin_target, top_k=3)
        target_layers = [model.model.visual.blocks[i].attn for i in best_indices]

        wrapped = Wrapper(model, enc)
        dynamic_reshape = partial(qwen_reshape_transform, height=grid_h, width=grid_w)

        grayscale_cam_yes = None
        grayscale_cam_no = None
        grayscale_cam_margin = None

        # --- 4. Run CAM ---
        with XGradCAM(model=wrapped, target_layers=target_layers, reshape_transform=dynamic_reshape) as cam:
            dummy_tensor = torch.zeros(1, 3, 512, 512, device=device)
            
            grayscale_cam_yes = cam(input_tensor=dummy_tensor, targets=[yes_target])[0, :]
            grayscale_cam_no = cam(input_tensor=dummy_tensor, targets=[no_target])[0, :]
            grayscale_cam_margin = cam(input_tensor=dummy_tensor, targets=[margin_target])[0, :]

        # --- 5. Save Results ---
        img_np = np.asarray(image_pil).astype(np.float32) / 255.0
        
        base_filename = f"gradcam_{meme_id}_lang-{args.lang}"
        if args.caption: base_filename += "_cap"
        if args.country_insertion: base_filename += "_country"
        if args.unimodal: base_filename += "_unimodal"
        
        if grayscale_cam_yes is not None:
            vis_yes = show_cam_on_image(img_np, grayscale_cam_yes, use_rgb=True, image_weight=0.6)
            path_yes = os.path.join(args.output_dir, f"{base_filename}_TARGET_YES.jpg")
            plt.imsave(path_yes, vis_yes)
        
        if grayscale_cam_no is not None:
            vis_no = show_cam_on_image(img_np, grayscale_cam_no, use_rgb=True, image_weight=0.6)
            path_no = os.path.join(args.output_dir, f"{base_filename}_TARGET_NO.jpg")
            plt.imsave(path_no, vis_no)

        if grayscale_cam_margin is not None:
            vis_margin = show_cam_on_image(img_np, grayscale_cam_margin, use_rgb=True, image_weight=0.6)
            path_margin = os.path.join(args.output_dir, f"{base_filename}_TARGET_MARGIN.jpg")
            plt.imsave(path_margin, vis_margin)
            
        print(f"    ✅ Saved 3 maps for Meme ID: {meme_id}")

    except Exception as e:
        print(f"    ❌ FAILED processing Meme ID: {meme_id}. Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up memory after each loop
        del enc, wrapped, grayscale_cam_yes, grayscale_cam_no, grayscale_cam_margin
        gc.collect()
        torch.cuda.empty_cache()


# ----------------
# NEW: Main function to run the batch
# ----------------
def main(args):
    # --- 1. Load Model (ONCE) ---
    print(f"--- Loading Model: {args.model_path} ---")
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    print("--- Model Loaded ---")

    # --- 2. Load and Filter Results CSV ---
    try:
        df_results = pd.read_csv(args.results_csv)
    except FileNotFoundError:
        print(f"Error: Results CSV not found at {args.results_csv}")
        sys.exit(1)
    
    if args.accuracy_column not in df_results.columns:
        print(f"Error: Column '{args.accuracy_column}' not found in {args.results_csv}")
        print(f"Available columns are: {df_results.columns.tolist()}")
        sys.exit(1)
        
    if "Meme ID" not in df_results.columns:
        print(f"Error: 'Meme ID' column not found in {args.results_csv}")
        sys.exit(1)

    # Filter for memes BELOW the threshold
    df_failures = df_results[df_results[args.accuracy_column] == args.accuracy_threshold].copy()
    
    # Get unique Meme IDs to process
    meme_ids_to_process = df_failures["Meme ID"].unique().tolist()
    
    if not meme_ids_to_process:
        print(f"No memes found below the {args.accuracy_threshold}% threshold. Exiting.")
        return
        
    print(f"--- Found {len(meme_ids_to_process)} memes with accuracy < {args.accuracy_threshold}% ---")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    # --- 3. Loop and Process ---
    cnt = 0
    for meme_id in tqdm(meme_ids_to_process, desc="Processing Failed Memes"):
        process_single_meme(model, processor, meme_id, args)
        cnt += 1
        if(cnt == 10): break
            

    print("--- Batch Grad-CAM Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Batch Grad-CAM on model failures.')
    
    # --- Inputs for finding failures ---
    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to the accuracy CSV file (e.g., "my_eval/US_acc.csv").')
    parser.add_argument('--accuracy_column', type=str, required=True,
                        help='Name of the accuracy column to check (e.g., "US_acc").')
    parser.add_argument('--accuracy_threshold', type=float, default=100.0,
                        help='Accuracy threshold (e.g., 80.0). Will run CAM on all memes *below* this value.')

    # --- Inputs for running the model ---
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_ID,
                        help='Path to the model snapshot or Hugging Face ID.')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Base directory where /captions and /memes folders are located.')
    parser.add_argument('--lang', type=str, default="en", choices=LANGUAGES,
                        help='Language of the memes to process (must match the CSV).')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save the output heatmaps.')

    # --- Prompting flags (MUST MATCH the experiment you are analyzing) ---
    parser.add_argument('--caption', action='store_true',
                        help='Include the meme caption in the prompt.')
    parser.add_argument('--country_insertion', action='store_true',
                        help='Insert country information into the prompt.')
    parser.add_argument('--unimodal', action='store_true',
                        help='Run the model in text-only (unimodal) mode.')

    args = parser.parse_args()
    main(args)