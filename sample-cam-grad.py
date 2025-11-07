import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ----------------
# Config
# ----------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_CSV = "data/captions/en.csv"
IMAGE_DIR = r"data/memes/en/American-Flag-Shotgun-Guy"
SAMPLE_MEME_ID = 142
QUESTION = "Is this meme hateful? Answer YES or NO."
#TARGET_WORD = "YES"
OUT_PATH = "gradcam_qwen2p5vl_yes2.jpg"

# Token budget via processor: lower these if you still hit OOM
MIN_PIXELS = 224 * 224
# Example: target a few hundred visual tokens; adjust as needed
MAX_PIXELS = 256 * 256

def load_data(meme_id):
    df = pd.read_csv(DATASET_CSV)
    row = df[df["Meme ID"] == meme_id].iloc[0]
    meme_text = row["Translation"]
    image_filename = f"{meme_id}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    img = Image.open(image_path).convert("RGB")
    print(f"Loaded Meme {meme_id}: {meme_text}")
    return img, meme_text

def qwen_reshape_transform(tensor):
    # Based on your logs, the grid is 18x18
    H, W = 18, 18
    # Ensure tensor is (Batch, Dim, Tokens)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    # Permute to (Batch, Dim, Tokens) if needed, then reshape to grid
    if tensor.size(-1) != H * W:
         # If shape is (B, Tokens, Dim), swap last two
         tensor = tensor.permute(0, 2, 1)
    
    # Reshape to (Batch, Dim, H, W)
    return tensor.reshape(tensor.size(0), -1, H, W)

def find_best_visual_block(model, enc, processor, target_token_id):
    device = model.device
    image_tensor = enc["pixel_values"]
    text_ids = enc["input_ids"]

    best_idx = None
    best_score = -float("inf")

    # Store activations and gradients
    handles = []
    activations, gradients = {}, {}

    def save_act(name):
        def hook(module, inp, out):
            activations[name] = out
        return hook

    def save_grad(name):
        def hook(module, grad_in, grad_out):
            gradients[name] = grad_out[0]
        return hook

    for i, blk in enumerate(model.model.visual.blocks):
        name = f"block_{i}"
        h1 = blk.attn.register_forward_hook(save_act(name))
        h2 = blk.attn.register_full_backward_hook(save_grad(name))
        handles.extend([h1, h2])

    # Forward
    out = model(
        input_ids=text_ids,
        pixel_values=image_tensor,
        image_grid_thw=enc.get("image_grid_thw"),
        use_cache=False,
    )

    # Choose the token we want to visualize ("YES")
    logit = out.logits[:, -1, target_token_id]
    logit.backward()

    # Evaluate gradient magnitude per block
    for name in activations.keys():
        g = gradients[name]
        score = g.abs().mean().item()
        if score > best_score:
            best_score = score
            best_idx = int(name.split("_")[1])

    for h in handles:
        h.remove()

    print(f"Best visual block index: {best_idx} (gradient strength = {best_score:.4f})")
    return best_idx

def build_prompt_and_inputs(processor, image_pil, device):
    # Build the chat template so vision tokens are inserted correctly
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QUESTION}
        ]}
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    # Create model inputs; keep everything on the model device
    enc = processor(text=[prompt], images=[image_pil], return_tensors="pt")
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    return enc

class Wrapper(torch.nn.Module):
    def __init__(self, model, enc_fixed):
        super().__init__()
        self.m = model
        self.enc_fixed = enc_fixed

    def forward(self, ignored_input):
        # We ignore the input from GradCAM (which will be a dummy tensor)
        # and use the pre-calculated real inputs instead.
        return self.m(
            input_ids=self.enc_fixed["input_ids"],
            attention_mask=self.enc_fixed.get("attention_mask"),
            pixel_values=self.enc_fixed["pixel_values"],
            image_grid_thw=self.enc_fixed.get("image_grid_thw"),
            use_cache=False,
        ).logits[:, -1, :]

def main():
    assert torch.cuda.is_available(), "CUDA is required for this script."
    device = torch.device("cuda")

    # Processor with pixel caps to limit visual tokens and memory
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # Model in half precision on GPU; try FlashAttention2 if available
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # or torch.float16
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
    except Exception:
        # Fallback if flash_attention_2 is unsupported
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # or torch.float16
            device_map="cuda",
        )
    model.eval()

    # Load image/data
    image_pil, _ = load_data(SAMPLE_MEME_ID)

    # Build prompt + inputs once, on the model device
    enc = build_prompt_and_inputs(processor, image_pil, model.device)
    print("Input keys:", list(enc.keys()))
    print("pixel_values shape:", tuple(enc["pixel_values"].shape))
    if "image_grid_thw" in enc:
        print("image_grid_thw:", enc["image_grid_thw"].tolist())

    # Grad-CAM target layer: Particular visual attention block
    #target_layers = [model.model.visual.blocks[-10].attn]
    
    # Grad-CAM target layer: a range of visual attention block
    # visual_blocks = model.model.visual.blocks
    # target_layers = [visual_blocks[i].attn for i in range(-1, -11, -1)]

    # Grad-CAM target layer: for best visual attention block

    
    
    # Targets: token id for "YES" (leading space variant often used by tokenizers)
    yes_ids = [
        processor.tokenizer(" YES").input_ids[-1],
        processor.tokenizer("YES").input_ids[-1],
    ]
    targets = [ClassifierOutputTarget(yes_ids[0])]

    best_idx = find_best_visual_block(model, enc, processor, yes_ids[0])
    target_layers = [model.model.visual.blocks[best_idx].attn]

    # Wrap model
    wrapped = Wrapper(model, enc).eval()

    # Ensure pixel_values matche model dtype/device
    px = enc["pixel_values"].to(device=model.device)

    dummy_tensor = torch.zeros(1, 3, 512, 512, device=model.device)

    # Compute CAM without augmentation/eigen smoothing to reduce VRAM
    with GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=qwen_reshape_transform) as cam:
        grayscale_cam = cam(
            input_tensor=dummy_tensor,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )

    # Prepare overlay
    cam_map = grayscale_cam[0]
    image_rgb_np = np.asarray(image_pil).astype(np.float32) / 255.0
    visualization = show_cam_on_image(image_rgb_np, cam_map, use_rgb=True, image_weight=0.5)
    plt.imsave(OUT_PATH, visualization)
    print(f"SAVED: {os.path.abspath(OUT_PATH)}")

if __name__ == "__main__":
    main()
