# --- Essential Imports ---
import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import sys
import gc
from functools import partial

# --- CONFIGURATION ---
MODEL_ID = "google/gemma-3-4b-it"

# List of Meme IDs verified as CORRECTLY predicted (Keep your lists here)
CORRECT_HATE_MEMES =  ['222', '37', '58', '78', '138', '13', '279', '281', '289', '46', '199', '226', '170', '24', '169', '224', '283', '139', '96', '33', '266', '154', '274', '52', '106', '148', '165', '234', '112', '146', '173', '43', '77', '158', '203', '298', '76', '292', '83', '88', '202', '249', '235', '258', '270', '132', '136', '164', '147', '16', '17', '247', '248', '275', '228', '237', '263', '151', '89', '19', '20', '21', '116', '144', '145', '62', '91', '119', '141', '157', '215', '256', '156', '196', '231']

CORRECT_NOHATE_MEMES =  ['137', '217', '257', '261', '11', '12', '8', '59', '60', '65', '94', '135', '160', '167', '201', '27', '99', '216', '278', '219', '297', '44', '29', '32', '171', '176', '110', '183', '252', '63', '179', '103', '204', '288', '290', '39', '41', '168', '200', '259', '260', '291', '293', '294', '295', '296', '84', '85', '93', '129', '284', '195', '276', '282', '130', '133', '265', '271', '286', '166', '181', '190', '208', '239', '0', '3', '6', '7', '71', '108', '72', '98', '114', '273', '66', '213', '264']      

LANG = "en"
DATA_DIR = "data/" 

# --- CRITICAL GEMMA CONFIGURATION ---
# Based on your logs: "model" -> "language_model" -> "layers" (0 to 33)
FINAL_LAYER_NAME = 'model.language_model.layers.33.mlp' 
HIDDEN_DIM = 2560 # Based on "Linear(in_features=2560...)"

# --- GLOBAL STORAGE ---
ACTIVATION_STORAGE_LIST = [] 

# --- HOOK FUNCTION ---
def capture_activation_hook(module, input, output, meme_id_hook):
    """
    Captures the final activation vector.
    """
    global ACTIVATION_STORAGE_LIST, HIDDEN_DIM
    
    # Check dimensions. Gemma output might vary slightly in shape index depending on batch
    # Usually: [Batch, Seq, Hidden]
    if output.dim() != 3 or output.shape[2] != HIDDEN_DIM:
        print(f"Warning: Hook output shape mismatch: {output.shape}. Expected {HIDDEN_DIM}. Skipping capture.")
        return
        
    final_token_vector = output[0, -1, :].cpu().float().numpy() 
    ACTIVATION_STORAGE_LIST.append((meme_id_hook, final_token_vector))


# --- MAIN ANALYSIS LOOP ---
def run_activation_analysis():
    global ACTIVATION_STORAGE_LIST
    
    device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    print(f"Using device: {device}")
    
    # --- MODEL LOADING ---
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=True)
        # Use AutoModel for Gemma
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=True
        )
        model.eval()
    except Exception as e:
        print(f"FATAL: Could not load model or processor. Error: {e}")
        return

    # --- Find Target Module ---
    # We walk down the path 'model.language_model.layers.33.mlp'
    target_module = model
    try:
        current_module = model
        for name in FINAL_LAYER_NAME.split('.'):
            current_module = getattr(current_module, name)
        target_module = current_module
        print(f"✅ Successfully hooked into: {FINAL_LAYER_NAME}")
    except AttributeError:
        print(f"ERROR: Could not find module '{FINAL_LAYER_NAME}'. Check model structure.")
        return

    try:
        caption_path = os.path.join(DATA_DIR, 'captions', f"{LANG}.csv")
        df_captions = pd.read_csv(caption_path)
        df_captions["Meme ID"] = df_captions["Meme ID"].astype(str)
        
        # NOTE: CHANGE THIS LIST TO 'CORRECT_HATE_MEMES' WHEN RUNNING FOR HATE
        # Currently set to NOHATE based on your previous code snippet
        TARGET_MEME_LIST = CORRECT_HATE_MEMES 
        
        for meme_id in TARGET_MEME_LIST:
            meme_id_str = str(meme_id)
            print(f"Processing meme {meme_id}...")
            
            # Register Hook
            hook_function_with_id = partial(capture_activation_hook, meme_id_hook=meme_id_str)
            hook_handle = target_module.register_forward_hook(hook_function_with_id)
            
            try:
                filtered_rows = df_captions[df_captions["Meme ID"] == meme_id_str]
                if filtered_rows.empty:
                    print(f"Skipping {meme_id}: ID not found in csv.")
                    hook_handle.remove()
                    continue
                    
                row = filtered_rows.iloc[0]
                
                # --- Prompting for Gemma ---
                question_text = "Given the image and the following text, answer if this is hate or not. Answer YES or NO."
                caption_text = str(row['Translation'])
                
                # Standard HF Multimodal format
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"}, 
                            {"type": "text", "text": f"{question_text}\nCaption: {caption_text}"}
                        ]
                    }
                ]
                
                # Apply chat template
                complex_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                
                # Image Loading Logic
                template_col = 'Template Name' if 'Template Name' in row.index else 'Translation'
                template_name = str(row[template_col]).replace(" ", "-").replace("/", "-")
                image_path = os.path.join(DATA_DIR, 'memes', LANG, f"{meme_id_str}.jpg")
                
                if not os.path.exists(image_path):
                     image_dir_fallback = os.path.join(DATA_DIR, 'memes', LANG, template_name)
                     image_path = os.path.join(image_dir_fallback, f"{meme_id_str}.jpg")
                     if not os.path.exists(image_path):
                        print(f"Skipping {meme_id}: Image not found.")
                        hook_handle.remove()
                        continue

                image_pil = Image.open(image_path).convert("RGB")
                
                # Inference
                with torch.no_grad():
                    inputs = processor(
                        text=[complex_prompt], 
                        images=[image_pil], 
                        return_tensors="pt"
                    ).to(model.device)
                    model(**inputs)

            except Exception as e:
                print(f"Error processing meme {meme_id}: {e}")
                
            finally:
                hook_handle.remove()


        # --- FINAL SAVING ---
        if ACTIVATION_STORAGE_LIST:
            ids, vectors = zip(*ACTIVATION_STORAGE_LIST)
            activation_matrix = np.stack(vectors) 
            
            # Save format
            dt = np.dtype([('Meme ID', 'U20'), ('Activation Vector', (np.float32, HIDDEN_DIM))])
            structured_array = np.empty(len(ids), dtype=dt)
            structured_array['Meme ID'] = ids
            structured_array['Activation Vector'] = activation_matrix

            # DYNAMIC FILENAME based on what list you ran
            type_label = "hate" if TARGET_MEME_LIST == CORRECT_HATE_MEMES else "nohate"
            save_filename = f"gemma_raw_{type_label}_activations_{len(ids)}samples.npy"
            
            np.save(save_filename, structured_array)
            print(f"\n--- Analysis Complete ({len(ids)} samples) ---")
            print(f"✅ Saved to '{save_filename}'")
        else:
            print("No samples processed.")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try: del model, processor, target_module; torch.cuda.empty_cache(); gc.collect()
        except: pass

if __name__ == "__main__":
    run_activation_analysis()