import argparse
import time
import PIL.Image
import google.generativeai as genai
import base64
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference


LANGUAGES = ["en", "de", "es", "hi", "zh"]
API_KEY_GEMINI = "xxxxxxxoooooooooo"
# genai.configure(api_key="AIzaSyCIZOS1Z5eTfWpAE5hi4JRI8YwntBmnOPo")

# try:
#     models = genai.list_models()
#     print("✅ API Key is valid!")
#     print("Available Models:")
#     for m in models:
#         print(" -", m.name)
# except Exception as e:
#     print("❌ API key problem:", e)

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Input for model_inference()
    processor = None
    processed_prompts = []
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            prompt_1 = raw_prompt[0]
            prompt_2 = raw_prompt[1]
            if add_caption:
                id_image = image_path.split("/")[-1].split(".jpg")[0]
                caption = df_captions[df_captions["ID"]
                                      == id_image]["Translation"].iloc[0]
                text_prompt_1 = {"type": "text",
                                 "text": prompt_1.format(str(caption))}
                text_prompt_2 = {"type": "text",
                                 "text": prompt_2.format(str(caption))}
            else:
                text_prompt_1 = {"type": "text", "text": prompt_1}
                text_prompt_2 = {"type": "text", "text": prompt_2}

            image_pil = PIL.Image.open(image_path)

            if unimodal:
                text_prompt_1["text"] = text_prompt_1["text"][:-7]
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "Caption inside the meme:", "Text:")
                text_prompt_1["text"] = text_prompt_1["text"].replace(
                    "meme", "text")
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "meme", "text")
                processed_prompts.append(
                    {"prompt": [text_prompt_1["text"], text_prompt_2["text"]]})
            else:
                processed_prompts.append(
                    {"prompt": [text_prompt_1["text"], image_pil, text_prompt_2["text"]]})

    return processor, processed_prompts


def model_creator(model_path):
    # Model Configuration

    model_config = genai.GenerationConfig(
        max_output_tokens=500,
        temperature=0.0,
    )
    genai.configure(api_key=API_KEY_GEMINI)
    model = genai.GenerativeModel(
        "gemini-2.5-flash", generation_config=model_config, safety_settings=SAFETY_SETTINGS)
    return model


def model_inference(prompt, model, processor, unimodal):
    time.sleep(0.2)

    response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
    try:
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        
        # --- START ROBUST FIX ---
        # Check if the response was blocked or is empty
        if not response.candidates or response.candidates[0].finish_reason.value != 1: # 1 = STOP (success)
            
            # Get the block reason (if available)
            block_reason = "UNKNOWN"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
            elif response.candidates:
                 block_reason = response.candidates[0].finish_reason.name

            print(f"  > [INFO] Response BLOCKED by API. Reason: {block_reason}")
            final_response_text = f"\nAssistant: BLOCKED (Reason: {block_reason})"
        
        else:
            # If not blocked, safely access the text
            final_response_text = "\nAssistant: " + response.text
        # --- END ROBUST FIX ---

        # Add the prompt back for logging
        response_text = prompt[0] + prompt[-1] + final_response_text

    except Exception as e:
        # Catch any other weird errors
        print(f"  > [ERROR] An exception occurred: {e}")
        response_text = prompt[0] + prompt[-1] + f"\nAssistant: ERROR - {str(e)}"

    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str,
                        required=False, default='gemini_pro/dont/matter')
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
