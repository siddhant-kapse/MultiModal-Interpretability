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

# CHANGED: Import AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------------
# Config
# ----------------
# CHANGED: Model ID updated to Gemma
DEFAULT_MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_DATA_DIR = "data/"
DEFAULT_MEME_ID = 88
DEFAULT_LANG = "en"
DEFAULT_OUTPUT_DIR = "gradcam_outputs_gemma"

# --- Prompting Constants ---
LANGUAGES = ["en", "de", "es", "hi", "zh"]
MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "hi": "IN",
    "zh": "CN"
}

DEFINITION = "Definition of Hate Speech:\n\nAny kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor."
SIMPLE_QUESTION = "Is this meme hateful? Answer YES or NO."

MIN_PIXELS = 224 * 224
MAX_PIXELS = 256 * 256

# CHANGED: Gemma 3's ViT (SigLIP-L) uses a 14x14 patch size(Need to study)
PATCH_SIZE = 14

# ----------------
# Custom Classes & Functions
# (THESE ARE UNCHANGED)
# ----------------

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
        score = model_output[:, self.token_ids].sum(dim=-1)
        return score

class BinaryMarginTarget:
    """
    A target class that computes the margin: score(YES) - score(NO)
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

# ----------------
# CHANGED: Model-Specific Functions for Gemma
# ----------------

def gemma_reshape_transform(tensor, grid_h, grid_w):
    """
    Reshapes the output of a standard ViT block for Grad-CAM.
    The input tensor is [batch, num_patches + 1, embed_dim]
    or a tuple (tensor, ...).
    """
    # === THIS IS THE FIX ===
    # Check if the layer output is a tuple, and grab the first
    # element (the hidden_state tensor) if it is.
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    # ========================

    if tensor.dim() == 3 and tensor.shape[1] == (grid_h * grid_w) + 1:
        # This is [batch, num_patches + 1, embed_dim]
        # Remove CLS token
        tensor = tensor[:, 1:, :]
    
    # Now tensor is [batch, num_patches, embed_dim]
    # Reshape to [batch, grid_h, grid_w, embed_dim]
    B, N, C = tensor.shape
    if N != grid_h * grid_w:
        print(f"Warning: Tensor shape {tensor.shape} mismatch with grid {grid_h}x{grid_w}")
        # Attempt to proceed if possible, though this may fail
        H = W = int(N**0.5)
        if H * W != N:
            raise ValueError(f"Cannot reshape {N} patches into a square grid.")
        print(f"Assuming {H}x{W} grid instead.")
        grid_h, grid_w = H, W

    tensor = tensor.reshape(B, grid_h, grid_w, C)
    
    # Permute to [batch, embed_dim, grid_h, grid_w] (Channels-first)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor

class Wrapper(torch.nn.Module):
    """
    Wrapper for the Gemma model to be compatible with pytorch-grad-cam.
    """
    def __init__(self, model, enc_fixed):
        super().__init__()
        self.m = model
        self.enc_fixed = enc_fixed
        
    def forward(self, ignored_input):
        
        # === THIS IS THE FIX ===
        # We force this specific forward pass (called by Grad-CAM)
        # to run in float32. This ensures the activations are in a
        # format that NumPy can read, fixing the bfloat16 error.
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            return self.m(
                input_ids=self.enc_fixed["input_ids"],
                attention_mask=self.enc_fixed.get("attention_mask"),
                pixel_values=self.enc_fixed["pixel_values"],
                use_cache=False,
            ).logits[:, -1, :]
        
        
def find_best_visual_block(model, enc, target, top_k=3):
    """
    Identify the most relevant visual transformer blocks by gradient × activation analysis.
    """
    image_tensor = enc["pixel_values"]
    text_ids = enc["input_ids"]
    scores = []
    handles, activations, gradients = [], {}, {}

    def save_act(name):
        def hook(m, i, o): activations[name] = o[0] # ViT blocks often return a tuple
        return hook

    def save_grad(name):
        def hook(m, gi, go): gradients[name] = go[0]
        return hook

    # CHANGED: Model architecture path for Gemma's ViT
    try:
        vision_blocks = model.vision_tower.vision_model.encoder.layers
    except AttributeError as e:
        print(f"❌ Could not find vision blocks at default path.")
        print(e)
        print("Defaulting to last block index 31 (SigLIP-L has 32 blocks).")
        return [31]
        
    start_block = max(0, len(vision_blocks) - 12)
    for i in range(start_block, len(vision_blocks)):
        blk = vision_blocks[i]
        name = f"block_{i}"
        handles.append(blk.register_forward_hook(save_act(name)))
        handles.append(blk.register_full_backward_hook(save_grad(name)))

    try:
        with torch.enable_grad():
            # CHANGED: Gemma's forward pass
            out = model(
                input_ids=text_ids,
                pixel_values=image_tensor,
                attention_mask=enc.get("attention_mask"),
                use_cache=False
            )
            logits = out.logits[:, -1, :]
            score_to_backprop = target(logits)
            score_to_backprop.backward(retain_graph=True)

        for name in activations:
            if name in gradients:
                act = activations[name]
                grad = gradients[name]
                score = (grad * act).abs().mean().item()
                scores.append((score, int(name.split("_")[1])))

    except Exception as e:
        print(f"⚠️ Block scan failed: {e}. Defaulting to last block.")
        scores = [(1.0, len(vision_blocks) - 1)]

    finally:
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
        gc.collect()

    if not scores:
        print("⚠️ No scores captured, defaulting to last block.")
        return [len(vision_blocks) - 1]

    scores.sort(key=lambda x: x[0], reverse=True)
    actual_k = min(top_k, len(scores))
    best_indices = [idx for _, idx in scores[:actual_k]]
    best_scores = [s for s, _ in scores[:actual_k]]

    print(f"Top-{actual_k} blocks: {best_indices} (scores: {[f'{s:.6f}' for s in best_scores]})")
    return best_indices

# ----------------
# Main
# ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Loading Model ---")

    processor_config = {
        "size": {"height": 336, "width": 336}
    }
    
    # CHANGED: Added token=True for Gemma
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, 
                                              token=True)
    
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32
    )
    
    # CHANGED: Use AutoModelForCausalLM, bfloat16, and token=True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        token=True,
        quantization_config=quantization_config
    )
    model.eval()

    # --- DYNAMIC DATA LOADING (Unchanged) ---
    print(f"--- Loading Data for Meme ID: {args.meme_id}, Lang: {args.lang} ---")
    try:
        dataset_csv = os.path.join(args.data_dir, 'captions', f"{args.lang}.csv")
        df = pd.read_csv(dataset_csv)
        row = df[df["Meme ID"] == args.meme_id].iloc[0]
        
        caption_text = row['Template Name']
        caption = row['Translation'] 
        
        image_dir = os.path.join(args.data_dir, 'memes', args.lang)
        img_path = os.path.join(image_dir, f"{args.meme_id}.jpg")
        
        if not os.path.exists(img_path):
             image_dir_fallback = os.path.join(args.data_dir, 'memes', args.lang, caption_text.replace(" ", "-"))
             img_path = os.path.join(image_dir_fallback, f"{args.meme_id}.jpg")
             if not os.path.exists(img_path):
                 print(f"Error: Could not find image file at {img_path} or in primary dir.")
                 sys.exit(1)

        image_pil = Image.open(img_path).convert("RGB")
        print(f"Loaded {args.meme_id}: {caption_text}")

    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Please check --data_dir, --lang, and --meme_id")
        sys.exit(1)


    # --- DYNAMIC PROMPT BUILDING (Unchanged logic) ---
    # This logic matches our working Gemma inference script
    
    question = f"{DEFINITION}\n\n{SIMPLE_QUESTION}"
    country = MAPPING[args.lang]

    text_prompt_1_content = question
    text_prompt_2_content = "" 

    if args.country_insertion:
        text_prompt_1_content = f"{SIMPLE_QUESTION} (in {country})"
    
    if args.caption:
        text_prompt_2_content = f"Caption inside the meme: {caption}"

    content_list = []
    content_list.append({"type": "text", "text": text_prompt_1_content})
    
    image_to_process = None
    if not args.unimodal:
        content_list.append({"type": "image"})
        image_to_process = [image_pil] # Pass the PIL image to processor
    
    if text_prompt_2_content:
        content_list.append({"type": "text", "text": text_prompt_2_content})

    conversation = [{"role": "user", "content": content_list}]
    
    print("--- Generated Prompt ---")
    print(conversation)
    print("--------------------------")

    # --- Processor Call (Unchanged logic) ---
    # This is the same logic as our working Gemma inference script
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=image_to_process, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    # --- CHANGED: Grid Calculation for Gemma ---
    # We can't get 'image_grid_thw'. We must calculate the grid
    # from the pixel_values tensor shape and the ViT patch size.
    try:
        pv_shape = enc['pixel_values'].shape
        grid_h = pv_shape[2] // PATCH_SIZE
        grid_w = pv_shape[3] // PATCH_SIZE
        print(f"Grid: {grid_h}x{grid_w} (from pixel_values {pv_shape} / patch_size {PATCH_SIZE})")
    except KeyError:
        print("❌ 'pixel_values' not in processor output. Cannot run CAM.")
        sys.exit(1)


    # --- CAM LOGIC (Adapted for Gemma) ---
    
    # This token logic is generic and should work for Gemma
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

    best_indices = find_best_visual_block(model, enc, margin_target, top_k=3)
    
    # CHANGED: Model architecture path for target layers
    try:
        target_layers = [model.vision_tower.vision_model.encoder.layers[i] for i in best_indices]
    except Exception as e:
        print(f"❌ Failed to get target layers: {e}")
        print("Aborting.")
        sys.exit(1)

    # num_blocks = len(model.vision_model.encoder.layers)

    # target_layers = [model.vision_model.encoder.layers[i].self_attn 
    #                  for i in range(num_blocks - 3, num_blocks)]

    wrapped = Wrapper(model, enc)
    
    # CHANGED: Use gemma_reshape_transform and pass grid_h/grid_w
    dynamic_reshape = partial(gemma_reshape_transform, grid_h=grid_h, grid_w=grid_w)

    grayscale_cam_yes = None
    grayscale_cam_no = None
    grayscale_cam_margin = None

    try:
        with XGradCAM(model=wrapped, target_layers=target_layers, reshape_transform=dynamic_reshape) as cam:
            dummy_tensor = torch.zeros(1, 3, 512, 512, device=device)
            
            print("Running CAM for YES target...")
            grayscale_cam_yes = cam(input_tensor=dummy_tensor,
                                    targets=[yes_target],
                                    aug_smooth=False, eigen_smooth=False)[0, :]
            torch.cuda.empty_cache()
            gc.collect()
            
            print("Running CAM for NO target...")
            grayscale_cam_no = cam(input_tensor=dummy_tensor,
                                   targets=[no_target],
                                   aug_smooth=False, eigen_smooth=False)[0, :]

            torch.cuda.empty_cache()
            gc.collect()

            print("Running CAM for MARGIN target...")
            grayscale_cam_margin = cam(input_tensor=dummy_tensor,
                                       targets=[margin_target],
                                       aug_smooth=False, eigen_smooth=False)[0, :]
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        print(f"❌ CAM generation failed: {e}")
        import traceback
        traceback.print_exc()

    # --- DYNAMIC VISUALIZATION & SAVING (Unchanged) ---
    img_np = np.asarray(image_pil).astype(np.float32) / 255.0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a dynamic base filename
    base_filename = f"gradcam_G3_{args.meme_id}_lang-{args.lang}" # Added G3 prefix
    if args.caption: base_filename += "_cap"
    if args.country_insertion: base_filename += "_country"
    if args.unimodal: base_filename += "_unimodal"

    # Visualize YES
    if grayscale_cam_yes is not None:
        vis_yes = show_cam_on_image(img_np, grayscale_cam_yes, use_rgb=True, image_weight=0.6)
        path_yes = os.path.join(args.output_dir, f"{base_filename}_TARGET_YES.jpg")
        plt.imsave(path_yes, vis_yes)
        print(f"SAVED: {os.path.abspath(path_yes)}")
    else:
        print("❌ Skipping YES save because CAM failed.")

    # Visualize NO
    if grayscale_cam_no is not None:
        vis_no = show_cam_on_image(img_np, grayscale_cam_no, use_rgb=True, image_weight=0.6)
        path_no = os.path.join(args.output_dir, f"{base_filename}_TARGET_NO.jpg")
        plt.imsave(path_no, vis_no)
        print(f"SAVED: {os.path.abspath(path_no)}")
    else:
        print("❌ Skipping NO save because CAM failed.")

    # Visualize MARGIN
    if grayscale_cam_margin is not None:
        vis_margin = show_cam_on_image(img_np, grayscale_cam_margin, use_rgb=True, image_weight=0.6)
        path_margin = os.path.join(args.output_dir, f"{base_filename}_TARGET_MARGIN.jpg")
        plt.imsave(path_margin, vis_margin)
        print(f"SAVED: {os.path.abspath(path_margin)}")
    else:
        print("❌ Skipping MARGIN save because CAM failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Grad-CAM analysis on a Gemma-3 model.')
    
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_ID,
                        help='Path to the model snapshot or Hugging Face ID.')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Base directory where /captions and /memes folders are located.')
    parser.add_argument('--meme_id', type=int, default=DEFAULT_MEME_ID,
                        help='The ID of the meme to analyze.')
    parser.add_argument('--lang', type=str, default=DEFAULT_LANG, choices=LANGUAGES,
                        help='Language to use for captions and prompt context.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save the output heatmaps.')

    # Flags from inference script
    parser.add_argument('--caption', action='store_true',
                        help='Include the meme caption in the prompt.')
    parser.add_argument('--country_insertion', action='store_true',
                        help='Insert country information into the prompt.')
    parser.add_argument('--unimodal', action='store_true',
                        help='Run the model in text-only (unimodal) mode.')

    args = parser.parse_args()
    main(args)