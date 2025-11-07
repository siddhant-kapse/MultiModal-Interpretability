# Copy of Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models

## Update 7-Nov
sample-cam-grad.py File contains working Grad-cam code for a single image on Qwen/Qwen2.5-VL-3B-Instruct Model.
TODOS
 - Need to build full pipeline for this Model.
 - More imporvements in the Grad analyser code
 - Replicate the similar pipeline foe llama4:17b-scout-16e-instruct-q4_K_M



## 🗂️ Dataset Structure
The dataset is organized in the `data/` folder:

- **Images**: `data/memes/` - Meme images categorized by language in subfolders.
- **Annotations**:
  - `data/final_annotations.csv` - Aggregated annotations.
  - `data/raw_annotations.csv` - Annotations by individual annotators.

## 🚀 Running VLM Inference

### 1. Model Inference
Use the scripts in `vlm/inference/` to run inference with Vision-Language Models (VLMs). Below are commands for each available model:

```bash
python vlm/inference/qwen2.py 
python vlm/inference/gemini_pro.py
```

## 📈 Model Evaluation

To evaluate model predictions, use this command, replacing `<folder>` with the path to your model's prediction folder:

```bash
python vlm/evaluation/eval --model_predictions <folder>
```
