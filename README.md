# Copy of Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models

## Update Dec-1st

new file:   accuracy.py -  The logic of EXTRACTING the meme-id for correctly classified Hate/Non-hate meme above certain threshold(80% is used)
new file:   roughy-g.py | roughy-q.py - Logic of getting the last layer information during forward pass for gemma and qwen model resp.
new file:   mean2.py | variance.py - Logic to find top-k neurons to ablate.
new file:   killer-neurons-gemma.py | killer-neurons-qwen.py - Prototype code for the ablation of selected neurons for gemma and qwen model resp.
new file:   t-test.py - Advance t-test Logic(IGNORE)


## 🗂️ Dataset Structure    
The dataset is organized in the `data/` folder:

- **Images**: `data/memes/` - Meme images categorized by language in subfolders.
- **Annotations**:
  - `data/final_annotations.csv` - Aggregated annotations.
  - `data/raw_annotations.csv` - Annotations by individual annotators.

## 🚀 Running VLM Inference

### 1. Model Inference
Use the scripts in `vlm/inference/` to run inference with Vision-Language Models (VLMs). Below are commands for the used models:

```bash
python vlm/inference/qwen2.py 
python vlm/inference/gemini_pro.py
```
