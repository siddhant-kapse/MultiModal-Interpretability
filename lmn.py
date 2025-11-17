# YOU MUST INSTALL bitsandbytes FIRST: pip install bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# --- This is the new part ---
# Define the 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
# ------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Pass the config to the .from_pretrained method
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config, # This tells it to load in 4-bit
    device_map="auto" # This will spread the 4-bit model across your GPUs
)

# ... rest of your code ...
# Now you can use the model...
prompt = "Hello, what can you do?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
