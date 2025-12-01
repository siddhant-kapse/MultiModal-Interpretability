# --- Essential Imports ---
import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import os
import sys
import gc
from functools import partial

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# List of Meme IDs verified as CORRECTLY predicted 'HATE'
CORRECT_HATE_MEMES =  ['222', '58', '78', '138', '142', '180', '279', '281', '289', '46', '199', 
                       '24', '149', '169', '283', '139', '96', '109', '33', '266', '154', '274', '52',
                         '148', '43', '158', '203', '298', '76', '272', '292', '88', '198', '202', '245', 
                         '249', '258', '267', '270', '136', '164', '192', '193', '182', '218', '242', '247', 
                         '248', '275', '237', '2', '117', '263', '151', '89', '116', '144', '145', '91', '119', '141', 
                         '157', '256', '156', '196', '231']

CORRECT_NOHATE_MEMES =  ['38', '74', '134', '137', '217', '257', '11', '12', '8', 
                         '243', '42', '59', '60', '65', '94', '135', '160', '167', '201', '22', '23', '26', '27', 
                         '99', '216', '220', '225', '229', '278', '211', '219', '297', '44', '29', '30', '128', '171', 
                         '172', '174', '176', '110', '183', '252', '63', '179', '103', '204', '288', '290', '39', '41', 
                         '177', '233', '51', '168', '200', '259', '260', '269', '291', '293', '294', '295', '296', '84', 
                         '85', '93', '284', '195', '221', '276', '282', '130', '131', '133', '265', '271', '286', '227',
                           '232', '125', '14', '15', '166', '181', '190', '205', '208', '239', '0', '3', '6', '7', '40',
                            '71', '101', '108', '72', '87', '98', '163', '114', '273', '115', '61', '66', '90', '184', '212', '213', '264', '73', '209']
LANG = "en"
DATA_DIR = "data/" 
# CRITICAL: Target Layer Name for the final MLP of the Text Decoder
FINAL_LAYER_NAME = 'model.language_model.layers.35.mlp' 

# --- GLOBAL STORAGE ---
# Now stores a list of tuples: [(meme_id, vector), (meme_id, vector), ...]
ACTIVATION_STORAGE_LIST = [] 
HIDDEN_DIM = 2048 

# --- HOOK FUNCTION ---
def capture_activation_hook(module, input, output, meme_id_hook):
    """
    Captures the final activation vector and stores it in the global list, 
    tagged with the meme ID.
    """
    global ACTIVATION_STORAGE_LIST, HIDDEN_DIM
    
    # We only care about the final token position for prediction
    # output shape: [Batch=1, Seq_Len, Hidden_Dim]
    if output.dim() != 3 or output.shape[2] != HIDDEN_DIM:
        print(f"Warning: Hook output shape mismatch: {output.shape}. Skipping capture.")
        return
        
    final_token_vector = output[0, -1, :].cpu().float().numpy() 
    
    # Store the ID and the raw vector
    ACTIVATION_STORAGE_LIST.append((meme_id_hook, final_token_vector))


# --- MAIN ANALYSIS LOOP ---
def run_activation_analysis():
    global ACTIVATION_STORAGE_LIST
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    print(f"Using device: {device}")
    
    # ... (Model loading, error checking omitted for brevity, identical to last response) ...
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"FATAL: Could not load model or processor. Error: {e}")
        return

    # --- Find Target Module ---
    target_module = model
    try:
        current_module = model
        for name in FINAL_LAYER_NAME.split('.'):
            current_module = getattr(current_module, name)
        target_module = current_module
    except AttributeError:
        print(f"ERROR: Could not find module '{FINAL_LAYER_NAME}'. Check model structure.")
        return

    # NOTE: The hook is registered INSIDE the loop below using `partial`
    
    try:
        # Load Caption Data
        caption_path = os.path.join(DATA_DIR, 'captions', f"{LANG}.csv")
        df_captions = pd.read_csv(caption_path)
        df_captions["Meme ID"] = df_captions["Meme ID"].astype(str)
        
        # --- Processing Loop ---
        for meme_id in CORRECT_NOHATE_MEMES:
            meme_id_str = str(meme_id)
            print(f"Processing meme {meme_id}...")
            
            # --- Register the Hook (MUST BE DONE HERE) ---
            # We use functools.partial to inject the current meme_id_str into the hook function
            hook_function_with_id = partial(capture_activation_hook, meme_id_hook=meme_id_str)
            hook_handle = target_module.register_forward_hook(hook_function_with_id)
            
            try:
                # 1. Get relevant caption row
                filtered_rows = df_captions[df_captions["Meme ID"] == meme_id_str]
                if filtered_rows.empty:
                    print(f"Skipping {meme_id}: Meme ID not found in {caption_path}.")
                    hook_handle.remove()
                    continue
                    
                row = filtered_rows.iloc[0]
                
                # --- Build Prompt (Multimodal Fix) ---
                question_text = "Given the image and the following text, answer if this is hate or not. Answer YES or NO."
                caption_text = str(row['Translation'])
                conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{question_text}\nCaption: {caption_text}"}]}]
                complex_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                
                # --- Load Image (Multi-step Fallback) ---
                template_col = 'Template Name' if 'Template Name' in row.index else 'Translation'
                template_name = str(row[template_col]).replace(" ", "-").replace("/", "-")
                image_path = os.path.join(DATA_DIR, 'memes', LANG, f"{meme_id_str}.jpg")
                
                if not os.path.exists(image_path):
                     image_dir_fallback = os.path.join(DATA_DIR, 'memes', LANG, template_name)
                     image_path = os.path.join(image_dir_fallback, f"{meme_id_str}.jpg")
                     
                     if not os.path.exists(image_path):
                        print(f"Skipping {meme_id}: Image not found at expected paths.")
                        hook_handle.remove()
                        continue

                image_pil = Image.open(image_path).convert("RGB")
                
                # --- Run Inference (Hook Captures Data) ---
                with torch.no_grad():
                    inputs = processor(
                        text=[complex_prompt], 
                        images=[image_pil], 
                        return_tensors="pt"
                    ).to(device)
                    model(**inputs)

            except Exception as e:
                print(f"Error processing meme {meme_id}: {e}")
                
            finally:
                # CRITICAL: Always remove the hook immediately after use!
                hook_handle.remove()


        # --- FINAL AGGREGATION & SAVING ---
        if ACTIVATION_STORAGE_LIST:
            # Separate the IDs and the vectors
            ids, vectors = zip(*ACTIVATION_STORAGE_LIST)
            
            # Convert the list of vectors into a single 2D NumPy array
            activation_matrix = np.stack(vectors) 
            
            # Create a structured NumPy array to save both IDs and data
            dt = np.dtype([('Meme ID', 'U20'), ('Activation Vector', (np.float32, HIDDEN_DIM))])
            structured_array = np.empty(len(ids), dtype=dt)
            structured_array['Meme ID'] = ids
            structured_array['Activation Vector'] = activation_matrix

            save_filename = f"raw_nohate_activations_{len(ids)}samples.npy"
            np.save(save_filename, structured_array)
            
            print(f"\n--- Analysis Complete ({len(ids)} samples) ---")
            print(f"✅ Successfully saved raw activations to '{save_filename}' (Shape: {activation_matrix.shape})")
            
            # Example of how to load and analyze later:
            # data = np.load('raw_hate_activations_3samples.npy', allow_pickle=True)
            # vector_46 = data[data['Meme ID'] == '46']['Activation Vector'][0]
        else:
            print("No samples were successfully processed.")
        
    except Exception as e:
        print(f"Fatal error outside of meme loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Final memory cleanup.")
        # Ensure model variables are deleted if they exist
        try: del model, processor, target_module; torch.cuda.empty_cache(); gc.collect()
        except: pass


# --- EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    run_activation_analysis()