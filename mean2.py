import numpy as np
import os

# --- CONFIGURATION ---
HATE_FILE = "raw_hate_activations_66samples.npy"
NON_HATE_FILE = "raw_nohate_activations_113samples.npy"

#--- UPDATED TO NEW FILES FROM GEMMA RUNS ---
HATE_FILE = "gemma_raw_hate_activations_75samples.npy"
NON_HATE_FILE = "gemma_raw_nohate_activations_77samples.npy"

TOP_K = 20

def analyze_discriminative_activations():
    # 1. Load Data
    try:
        A_H = np.load(HATE_FILE, allow_pickle=True)
        A_NH = np.load(NON_HATE_FILE, allow_pickle=True)
        
        # Assume data extraction logic to get the raw activation matrices
        # (This is a simplified assumption based on your previous code structure)
        A_H = A_H['Activation Vector'] if A_H.dtype.names and 'Activation Vector' in A_H.dtype.names else A_H 
        A_NH = A_NH['Activation Vector'] if A_NH.dtype.names and 'Activation Vector' in A_NH.dtype.names else A_NH

        print(f"Hate Activations shape: {A_H.shape}")
        print(f"Non-Hate Activations shape: {A_NH.shape}")

    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Calculate Mean Activations for each group
    mu_H = np.mean(A_H, axis=0)
    mu_NH = np.mean(A_NH, axis=0)

    # 3. Calculate Separator Score: Absolute Difference in Means
    separator_score = np.abs(mu_H - mu_NH)

    # 4. Identify Top-K Neurons by Separator Score
    # argsort gives indices from low to high, so we reverse it [::-1]
    top_k_indices = np.argsort(separator_score)[::-1][:TOP_K]
    # 5. Print Results with CORRECTED Logic
    print(f"\n--- Top {TOP_K} Neurons by DISCRIMINATION (Absolute Mean Difference) ---")
    print(f"{'Rank':<5} | {'Neuron ID':<10} | {'Score':<10} | {'Mean Hate':<12} | {'Mean Non-Hate':<15} | {'Dominant Behavior'}")
    print("-" * 90)

    for rank, neuron_idx in enumerate(top_k_indices):
        score = separator_score[neuron_idx]
        mean_h = mu_H[neuron_idx]
        mean_nh = mu_NH[neuron_idx]

        # LOGIC FIX: Determine dominance by MAGNITUDE (Intensity), not just algebraic value
        if abs(mean_h) > abs(mean_nh):
            # Hate is the stronger driver
            direction = "(+)" if mean_h > 0 else "(-)" # Is it excitatory or inhibitory?
            label_focus = f"HATE {direction}"
        else:
            # Non-Hate is the stronger driver
            direction = "(+)" if mean_nh > 0 else "(-)"
            label_focus = f"NON-HATE {direction}"
        
        print(f"{rank+1:<5} | {neuron_idx:<10} | {score:<10.4f} | {mean_h:<12.4f} | {mean_nh:<15.4f} | {label_focus}")

    # Optional: Save these indices
    np.save("top_k_mean2_gemma_neurons.npy", top_k_indices)
    print("\n✅ Saved Top-K indices to 'top_k_mean2_neurons.npy'")
if __name__ == "__main__":
    analyze_discriminative_activations()