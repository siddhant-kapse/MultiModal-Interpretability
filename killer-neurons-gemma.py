import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import sys
import gc
from functools import partial

# --- CONFIGURATION ---
MODEL_ID = "google/gemma-3-4b-it"
LANG = "en"
DATA_DIR = "data/"

# TOP 20 NEURONS FROM YOUR MEAN2.PY RESULT
# (814, 435, 896, 1571, 1313, 468, 2482, 172, 362, 1092, 1886, 86, 1646, 2552, 1305, 1179, 1040, 807, 545, 31)
NEURONS_TO_ABLATE = [814, 435, 896, 1571, 1313, 468, 2482, 172, 362, 1092, 
                     1886, 86, 1646, 2552, 1305, 1179, 1040, 807, 545, 31]

NEURONS_TO_ABLATE = [443, 732, 814,435, 1646, 896, 2095, 278, 2128, 1313,
                     1802, 921, 2062, 1692, 81, 1571, 1834, 2482, 634, 953]

# Use the HATE list because we want to see if we can "turn off" the hate detection
# (Or turn off the safety detection if the neuron is Non-Hate (+))
CORRECT_HATE_MEMES = ['222', '58', '78', '138', '142', '180', '279', '281', '289', '46', '199', '24']
CORRECT_NOHATE_MEMES = ['137', '217', '257', '261', '11', '12', '8', '59', '60', '65', '94', '135']

# Target the same layer we analyzed
TARGET_LAYER = 'model.language_model.layers.33.mlp'
HIDDEN_DIM = 2560

# --- THE ABLATION HOOK ---
def ablation_hook(module, input, output, neurons_to_zero):
    """
    Sets specific neurons in the output vector to 0.
    Output shape: [Batch, Seq_Len, Hidden_Dim]
    """
    # We want to ablate the LAST token's activation (where prediction happens)
    # output is a tensor, we can modify it in-place
    
    # Iterate over the batch (usually just 1)
    for b in range(output.shape[0]):
        # Iterate over the specific neurons and set them to 0
        for neuron_idx in neurons_to_zero:
            # [Batch, Last_Token, Neuron_Index]
            output[b, -1, neuron_idx] = 0.0
            
    return output
# --- MAIN LOOP ---
def run_ablation_experiment():
    print(f"--- Starting Ablation on {MODEL_ID} ---")
    print(f"Nullifying {len(NEURONS_TO_ABLATE)} neurons: {NEURONS_TO_ABLATE}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", token=True
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Attach Hook
    target_module = model
    try:
        current_module = model
        for name in TARGET_LAYER.split('.'):
            current_module = getattr(current_module, name)
        target_module = current_module
    except AttributeError:
        print(f"Error finding layer {TARGET_LAYER}")
        return

    hook_fn = partial(ablation_hook, neurons_to_zero=NEURONS_TO_ABLATE)
    hook_handle = target_module.register_forward_hook(hook_fn)
    print(f"✅ Ablation Hook Registered on {TARGET_LAYER}")

    # 3. Load Metadata
    caption_path = os.path.join(DATA_DIR, 'captions', f"{LANG}.csv")
    df_captions = pd.read_csv(caption_path)
    df_captions["Meme ID"] = df_captions["Meme ID"].astype(str)

    flipped_count = 0
    total_count = 0

    print(f"\nTesting on {len(CORRECT_HATE_MEMES)} Hate Memes...")
    print("-" * 50)
    
    # 4. Processing Loop
    for meme_id in CORRECT_HATE_MEMES:
        meme_id_str = str(meme_id)
        
        try:
            filtered_rows = df_captions[df_captions["Meme ID"] == meme_id_str]
            if filtered_rows.empty: continue
            row = filtered_rows.iloc[0]
            
            # Prepare Prompt
            question_text = "Given the image and the following text, answer if this is hate or not. Answer YES or NO."
            caption_text = str(row['Translation'])
            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{question_text}\nCaption: {caption_text}"}]}]
            complex_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

            # Load Image
            template_col = 'Template Name' if 'Template Name' in row.index else 'Translation'
            template_name = str(row[template_col]).replace(" ", "-").replace("/", "-")
            image_path = os.path.join(DATA_DIR, 'memes', LANG, f"{meme_id_str}.jpg")
            if not os.path.exists(image_path):
                 image_dir_fallback = os.path.join(DATA_DIR, 'memes', LANG, template_name)
                 image_path = os.path.join(image_dir_fallback, f"{meme_id_str}.jpg")
                 if not os.path.exists(image_path): continue
            
            image_pil = Image.open(image_path).convert("RGB")
            
            # Inference
            with torch.no_grad():
                inputs = processor(text=[complex_prompt], images=[image_pil], return_tensors="pt").to(model.device)
                
                # Generate
                outputs = model.generate(**inputs, max_new_tokens=10)
                
                # --- YOUR REQUESTED OUTPUT LOGIC ---
                generated_ids = [output_ids[len(inputs.input_ids[0]):] for output_ids in outputs]
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # CHECK RESULT
                # We expect "YES" (Hate). If it says "NO", we flipped it!
                clean_response = response.strip().upper()
                
                print(f"Meme {meme_id}: Prediction -> {clean_response}")
                
                if "NO" in clean_response:
                    print(f"   >>> 🚨 FLIP DETECTED! Meme {meme_id} is now safe!")
                    flipped_count += 1
                
                total_count += 1

        except Exception as e:
            print(f"Error on {meme_id}: {e}")

    # Cleanup
    hook_handle.remove()
    
    # Final Stats
    print("\n" + "=" * 40)
    print(f"Experiment Result: {flipped_count}/{total_count} memes flipped to 'NO'.")
    print(f"Success Rate of Ablation: {(flipped_count/total_count)*100:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    run_ablation_experiment()