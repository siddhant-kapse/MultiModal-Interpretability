import argparse
# REMOVED: from qwen_vl_utils import process_vision_info
# CHANGED: Import AutoModelForCausalLM instead of AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import sys
import os
import torch  # NEW: Added for data types and device handling

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference, create_prompt_for_input


# LANGUAGES = ["en", "de", "es", "hi", "zh"]
LANGUAGES = ["en"]
def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    # Input for model_inference()
    processor = AutoProcessor.from_pretrained(model_path, token=True)
    processed_prompts = []
    
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            text_prompt_1, text_prompt_2 = create_prompt_for_input(
                raw_prompt, df_captions, image_path, add_caption)

            # Your unimodal text-replacement logic
            if unimodal:
                text_prompt_1["text"] = text_prompt_1["text"][:-7]
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "Caption inside the meme:", "Text:")
                text_prompt_1["text"] = text_prompt_1["text"].replace(
                    "meme", "text")
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "meme", "text")
                
                # Content list *without* the image
                content = [text_prompt_1, text_prompt_2]
                conversation = [{"role": "user", "content": content}]
            else:
                # Content list *with* the image
                content = [
                    text_prompt_1,
                    {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    text_prompt_2
                ]
                conversation = [{"role": "user", "content": content}]

            # === KEY CHANGE ===
            # We call apply_chat_template to get the formatted *string*.
            # This string will NOT contain the image, but it will have the
            # text formatted correctly (e.g., <bos><start_of_turn>user...)
            processed_prompt_string = processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True,
                # We do NOT pass return_tensors="pt" here
            )

            # We now append a list, just like the Qwen script:
            # [raw_conversation, formatted_text_string]
            processed_prompts.append(
                {"prompt": [conversation, processed_prompt_string]})

    return processor, processed_prompts


def model_creator(model_path):
    # CHANGED: Load as AutoModelForCausalLM, not AutoModelForVision2Seq
    # This is the standard for new VLMs like Gemma 3 and LLaVA
    # NEW: Added 4-bit loading for local laptop use
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # From Gemma 3 docs
        device_map="auto",            # Automatically uses GPU if available
        token=True
    )
    return model

def model_inference(prompt, model, processor, unimodal):
    # 'prompt' is the list: [conversation, processed_prompt_string]
    import torch
    
    # Get the raw conversation and the formatted text string
    conversation = prompt[0]
    text_string = prompt[1]
    
    # Get the device from the model
    device = model.device 
    print("Model device:", device)

    # === KEY CHANGE ===
    # We must manually find the image in the conversation list,
    # just like Qwen's 'process_vision_info' does.
    image = None
    if not unimodal:
        try:
            # Find the image object in the conversation content
            for item in conversation[0]['content']:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item["image"] # This is the PIL.Image object
                    break
        except Exception as e:
            print(f"Error extracting image from conversation: {e}")
            print(f"Problematic conversation: {conversation}")
            raise

    # === KEY CHANGE ===
    # Now we call the processor's main __call__ method,
    # just like the Qwen script does. This will correctly
    # process BOTH text and images and return a dict of tensors.
    try:
        inputs = processor(
            text=text_string,  # Pass the formatted string
            images=image,      # Pass the PIL image (or None)
            padding=True,
            return_tensors="pt",
        )
    except Exception as e:
        print(f"Error in processor __call__: {e}")
        print(f"Text String: {text_string}")
        print(f"Image object: {image}")
        raise

    # 'inputs' is now the dict of tensors. We can move it to the device.
    try:
        inputs_on_device = {key: tensor.to(device) for key, tensor in inputs.items()}
    except AttributeError:
        # This will catch if 'inputs' is still a string (it shouldn't be)
        print(f"ERROR: processor() call did not return a dict.")
        print(f"Processor output: {inputs}")
        raise
        
    # Get the input length to decode only new tokens
    input_length = inputs_on_device['input_ids'].shape[1]

    # Generate output
    output = model.generate(**inputs_on_device, max_new_tokens=1024,
                            do_sample=False, top_k=None)

    # Decode only the newly generated part
    response_ids = output[0][input_length:]
    response_text = processor.decode(response_ids, skip_special_tokens=True)
    
    return response_text

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run pipeline inference with specified model path.')

    # CHANGED: Updated the default model path
    parser.add_argument('--model_path', type=str, required=False,
                        default='google/gemma-3-4b-it')
    parser.add_argument('--caption', action='store_true',
                        help='Enable captioning')
    parser.add_argument('--multilingual', action='store_true',
                        help='Enable captioning')
    parser.add_argument('--country_insertion',
                        action='store_true', help='Enable captioning')
    parser.add_argument('--unimodal', action='store_true',
                        help='Enable captioning')
    args = parser.parse_args()

    pipeline_inference(args.model_path, LANGUAGES, input_creator, model_creator, model_inference,
                       add_caption=args.caption, multilingual=args.multilingual, country_insertion=args.country_insertion,
                       unimodal=args.unimodal)