# Copy of Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models

## 🗂️ Dataset Structure
The dataset is organized in the `data/` folder:

- **Images**: `data/memes/` - Meme images categorized by language in subfolders.
- **Annotations**:
  - `data/final_annotations.csv` - Aggregated annotations.
  - `data/raw_annotations.csv` - Annotations by individual annotators.

## 🚀 Running VLM Inference

### 1. Install Dependencies
To get started, install the required dependencies:

```bash
pip install -r requirements-new.txt
```

### 2. Model Inference
Use the scripts in `vlm/inference/` to run inference with Vision-Language Models (VLMs). Below are commands for each available model:

```bash
python vlm/inference/qwen2.py 
python vlm/inference/gemini_pro.py
```


> **Important Notes**:
> - For closed-source models, provide a valid API key.
> - Ensure you have the correct version of `transformers` installed for `internvl`:
>   ```bash
>   pip install transformers==4.37.2
>   ```

## 📈 Model Evaluation

To evaluate model predictions, use this command, replacing `<folder>` with the path to your model's prediction folder:

```bash
python vlm/evaluation/eval --model_predictions <folder>
```
