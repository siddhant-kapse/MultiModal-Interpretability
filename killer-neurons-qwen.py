import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import os
from functools import partial

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_DIR = "data/"
LANG = "en"
FINAL_LAYER_NAME = 'model.language_model.layers.35.mlp' 

# Load the "Bad" Neurons found in Step 1
# This file should contain e.g., [45, 1024, 200, ...]
NEURON_INDICES_FILE = "top_k_mean2_neurons.npy" 

# The list of memes we know are HATE (Prediction = YES)
CORRECT_HATE_MEMES = ['222', '58', '78', '138', '142', '180', '279', '281', '289', '46', '199', 
                       '24', '149', '169', '283', '139', '96', '109', '33', '266', '154', '274'] # (Shortened for testing)
CORRECT_NOHATE_MEMES = ['38', '74', '134', '137', '217', '257', '11', '12', '8', 
                         '243', '42', '59', '60', '65', '94', '135', '160', '167', '201', '22', '23', '26', '27']
# --- THE ABLATION HOOK ---
def nullify_neurons_hook(module, input, output, neurons_to_kill):
    """
    Intervenes in the forward pass.
    Sets the specified neuron indices to 0.0 for all tokens.
    """
    # Qwen MLP output is usually a single tensor, but sometimes a tuple in HF
    if isinstance(output, tuple):
        tensor_to_modify = output[0]
    else:
        tensor_to_modify = output

    # CRITICAL: modifying in-place or returning a modified clone
    # Shape is usually (Batch, Seq_Len, Hidden_Dim)
    # We want to kill these neurons for the ENTIRE sequence (all tokens)
    
    # We define which indices to zero out
    # neurons_to_kill is a list/array of integers [index1, index2, ...]
    
    # Set those specific neurons to 0
    tensor_to_modify[:, :, neurons_to_kill] = 0.0
    
    # If it was a tuple, we must return a tuple structure
    if isinstance(output, tuple):
        return (tensor_to_modify,) + output[1:]
    
    return tensor_to_modify

# --- MAIN LOOP ---
def run_ablation_study():
    # 1. Load Data
    if not os.path.exists(NEURON_INDICES_FILE):
        print("Run Step 1 first to generate neuron indices!")
        return
        
    neurons_to_kill = np.load(NEURON_INDICES_FILE)
    print(f"🔫 Loaded {len(neurons_to_kill)} target neurons to nullify.")
    
    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    # 3. Find Layer
    target_module = model
    for name in FINAL_LAYER_NAME.split('.'):
        target_module = getattr(target_module, name)

    # 4. Attach the "Killer" Hook
    # We attach it ONCE here, and it stays active for all memes
    # Use partial to pass the list of neurons
    hook_fn = partial(nullify_neurons_hook, neurons_to_kill=neurons_to_kill)
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    print(f"⚠️ Ablation Hook Active on {FINAL_LAYER_NAME}. Neurons are being suppressed.")

    # 5. Run Inference Loop
    df_captions = pd.read_csv(os.path.join(DATA_DIR, 'captions', f"{LANG}.csv"))
    df_captions["Meme ID"] = df_captions["Meme ID"].astype(str)

    flipped_count = 0
    total_count = 0

    print("\n--- Starting Ablation Testing ---")
    
    for meme_id in CORRECT_HATE_MEMES:
        total_count += 1
        # ... (Image/Prompt Setup same as before) ...
        filtered = df_captions[df_captions["Meme ID"] == str(meme_id)]
        if filtered.empty: continue
        
        row = filtered.iloc[0]
        question = "Given the image and the following text, answer if this is hate or not. Answer YES or NO."
        text_promt = f"{question}\nCaption: {row['Translation']}"
        
        # Build Inputs
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_promt}]}]
        text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
    # --- Load Image (Multi-step Fallback) ---
        template_col = 'Template Name' if 'Template Name' in row.index else 'Translation'
        template_name = str(row[template_col]).replace(" ", "-").replace("/", "-")
        image_path = os.path.join(DATA_DIR, 'memes', LANG, f"{meme_id}.jpg")
        
        if not os.path.exists(image_path):
                image_dir_fallback = os.path.join(DATA_DIR, 'memes', LANG, template_name)
                image_path = os.path.join(image_dir_fallback, f"{meme_id}.jpg")
                if not os.path.exists(image_path):
                    print(f"Skipping {meme_id}: Image not found at expected paths.")
                    hook_handle.remove()
                    continue   
        
        image_pil = Image.open(image_path).convert("RGB")

        inputs = processor(text=[text_prompt], images=[image_pil], return_tensors="pt").to(device)

        # GENERATE (The hook will implicitly modify activations during this)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10)
            
        generated_ids = [output_ids[len(inputs.input_ids[0]):] for output_ids in output_ids]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # CHECK RESULT
        # We expect "YES" (Hate). If it says "NO", we flipped it!
        clean_response = response.strip().upper()
        
        print(f"Meme {meme_id}: Prediction -> {clean_response}")
        
        if "NO" in clean_response:
            print(f"   >>> 🚨 FLIP DETECTED! Meme {meme_id} is now safe!")
            flipped_count += 1

    # Cleanup
    hook_handle.remove()
    
    print("-" * 30)
    print(f"Experiment Result: {flipped_count}/{total_count} memes flipped to 'NO'.")
    print(f"Success Rate of Ablation: {(flipped_count/total_count)*100:.2f}%")

if __name__ == "__main__":
    run_ablation_study()