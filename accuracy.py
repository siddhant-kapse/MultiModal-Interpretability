import argparse
import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION (Must match your data structure) ---
# We assume the annotation file is accessible via this path:
from vlm.inference.local_paths import ANNOTATION_PATH 

# We assume the Meme ID column in the GT file contains the label for the US culture.
# Adjust if your target culture is different (e.g., 'DE', 'MX', etc.)
TARGET_GT_COLUMN = "US" 

# --- MAIN LOGIC ---

def load_data(args):
    """Loads and merges the three required data tables: GT, Accuracy, and Responses."""
    
    # 1. Load Ground Truth (GT)
    try:
        df_gt = pd.read_csv(ANNOTATION_PATH)
        df_gt["Meme ID"] = df_gt["Meme ID"].astype(str)
        # Filter GT to only keep the Meme ID and the target culture's label
        df_gt = df_gt[['Meme ID', TARGET_GT_COLUMN]]
        df_gt.rename(columns={TARGET_GT_COLUMN: 'Is_Hate_Meme'}, inplace=True)
        print(f"Loaded GT file, using '{TARGET_GT_COLUMN}' as the hate label.")
    except Exception as e:
        print(f"Error loading Ground Truth from ANNOTATION_PATH: {e}")
        sys.exit(1)

    # 2. Load Per-Meme Accuracy CSV
    try:
        # We assume the accuracy CSV uses the format: Meme ID, US_acc
        df_acc = pd.read_csv(args.accuracy_csv_path)
        df_acc["Meme ID"] = df_acc["Meme ID"].astype(str)
        print(f"Loaded Accuracy file.")
    except Exception as e:
        print(f"Error loading Accuracy CSV: {e}")
        sys.exit(1)
        
    # Check if the accuracy column exists (e.g., 'US_acc')
    if args.accuracy_column not in df_acc.columns:
        print(f"Error: Accuracy column '{args.accuracy_column}' not found in the CSV.")
        print(f"Available columns: {df_acc.columns.tolist()}")
        sys.exit(1)

    # 3. Merge ACCURACY and GROUND TRUTH
    # This is the three-way merge to get the final required information
    df_final = pd.merge(df_acc, df_gt, on='Meme ID', how='inner')
    
    if df_final.empty:
        print("FATAL: Final merge resulted in an empty table. Check if Meme IDs match.")
        sys.exit(1)
        
    print(f"Final merged dataset size: {len(df_final)} memes.")
    return df_final


def run_analysis(args):
    """Performs filtering and prints the required lists."""
    
    # 1. Load and merge the data
    df = load_data(args)
    
    acc_col = args.accuracy_column
    
    # 2. Filter by the desired threshold
    df_filtered = df[df[acc_col] >= args.accuracy_threshold].copy()
    
    if df_filtered.empty:
        print(f"\nNo memes were found with accuracy >= {args.accuracy_threshold}%. Exiting.")
        return

    # 3. Categorize by Ground Truth (GT) Label E:\Ghent\Multi3Hate\vlm\results\Qwen2.5-VL-3B-Instruct\per_meme_acc_en.csv E:\Ghent\Multi3Hate\vlm\results\gemma-3-4b-it\per_meme_acc_en.csv
    
    # Memes where GT is HATE (1) AND Accuracy is high
    df_hate_correct = df_filtered[df_filtered['Is_Hate_Meme'] == 1]
    
    # Memes where GT is NON-HATE (0) AND Accuracy is high
    df_non_hate_correct = df_filtered[df_filtered['Is_Hate_Meme'] == 0]
    
    # 4. Extract and print the final results
    
    print("\n" + "="*70)
    print(f"| ANALYSIS RESULTS (Accuracy >= {args.accuracy_threshold}%)")
    print("="*70)
    
    # --- HATE MEMES (To find the 'Hate' Circuit) ---
    hate_ids = df_hate_correct['Meme ID'].tolist()
    print(f"| HATE MEME CIRCUIT ({len(hate_ids)} Memes)")
    print("| These memes are consistently predicted 'HATE' and are correct.")
    print("|\n| MEME IDs (Array Format):")
    print(f"| {hate_ids}")
    print("-"*70)

    # --- NON-HATE MEMES (To find the 'Non-Hate' Circuit) ---
    non_hate_ids = df_non_hate_correct['Meme ID'].tolist()
    print(f"| NON-HATE MEME CIRCUIT ({len(non_hate_ids)} Memes)")
    print("| These memes are consistently predicted 'NON-HATE' and are correct.")
    print("|\n| MEME IDs (Array Format):")
    print(f"| {non_hate_ids}")
    print("="*70 + "\n")
    
    # OPTIONAL: Save the combined table for reference
    output_filename = f"verified_correct_memes_{TARGET_GT_COLUMN}_gemma.csv"
    df_filtered.to_csv(output_filename, index=False)
    print(f"Saved combined data to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Categorize Meme IDs based on high prediction accuracy and Ground Truth label.')
    
    parser.add_argument('--accuracy_csv_path', type=str, required=True,
                        help='Path to the per-meme accuracy CSV (e.g., "per_meme_acc_en.csv").')
    parser.add_argument('--accuracy_column', type=str, default='US_acc',
                        help='Name of the accuracy column to filter (e.g., "US_acc").')
    parser.add_argument('--accuracy_threshold', type=float, default=80.0,
                        help='Minimum accuracy percentage (0-100) for a prediction to be considered reliable.')
                        
    args = parser.parse_args()
    run_analysis(args)