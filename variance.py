import numpy as np
import os

# --- CONFIGURATION ---
HATE_FILE = "raw_hate_activations_66samples.npy"
NON_HATE_FILE = "raw_nohate_activations_113samples.npy"

#--- UPDATED TO NEW FILES FROM GEMMA RUNS ---
HATE_FILE = "gemma_raw_hate_activations_75samples.npy"
NON_HATE_FILE = "gemma_raw_nohate_activations_77samples.npy"

TOP_K = 20

def analyze_mixed_variance():
    # 1. Load Data
    if not os.path.exists(HATE_FILE) or not os.path.exists(NON_HATE_FILE):
        print("Error: One or both input files not found.")
        return

    print("Loading data...")
    try:
        data_hate = np.load(HATE_FILE, allow_pickle=True)
        data_nonhate = np.load(NON_HATE_FILE, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load numpy files: {e}")
        return

    # 2. Extract Activation Vectors
    # Helper function to extract vectors depending on how they were saved
    def extract_activations(data):
        if data.dtype.names and 'Activation Vector' in data.dtype.names:
            return data['Activation Vector']
        else:
            # Fallback for list of tuples [('id', vector), ...]
            return np.stack([item[1] for item in data])

    act_hate = extract_activations(data_hate)
    act_nonhate = extract_activations(data_nonhate)

    print(f"Hate Samples: {act_hate.shape}")
    print(f"Non-Hate Samples: {act_nonhate.shape}")

    # 3. Create the Mixed Dataset
    # We combine them to calculate the TOTAL variance across the whole distribution
    act_mixed = np.concatenate([act_hate, act_nonhate], axis=0)
    print(f"Combined Mixed Shape: {act_mixed.shape}")

    # 4. Calculate Metrics
    # A. Variance on the MIXED set (This is the sorting metric)
    # High variance here means the neuron behaves very differently across the whole dataset
    mixed_variance = np.var(act_mixed, axis=0)

    mixed_mean = np.mean(act_mixed, axis=0)
    # B. Means for the individual groups (For analysis)
    mean_hate = np.mean(act_hate, axis=0)
    mean_nonhate = np.mean(act_nonhate, axis=0)

    # 5. Identify Top-K Neurons by MIXED VARIANCE
    top_k_indices = np.argsort(mixed_variance)[::-1][:TOP_K]

    # 6. Display Results
    print(f"\n--- Top {TOP_K} Neurons by MIXED VARIANCE ---")
    print(f"{'Rank':<5} | {'Neuron ID':<10} | {'Mixed Var':<12} | {'Mixed Mean' :<12} | {'Mean Hate':<12} | {'Mean Non-Hate':<15}")
    print("-" * 85)

    for rank, neuron_idx in enumerate(top_k_indices):
        var_val = mixed_variance[neuron_idx]
        h_val = mean_hate[neuron_idx]
        nh_val = mean_nonhate[neuron_idx]
        mixed_mean_val = mixed_mean[neuron_idx]
        # Determine behavior based on the difference
        # If Hate is significantly higher than Non-Hate
        if abs(h_val) > abs(nh_val):
            # The Hate magnitude is stronger
            direction = "(+)" if h_val > 0 else "(-)"
            behavior = f"HATE {direction}"
        else:
            # The Non-Hate magnitude is stronger
            direction = "(+)" if nh_val > 0 else "(-)"
            behavior = f"NON-HATE {direction}"

        print(f"{rank+1:<5} | {neuron_idx:<10} | {var_val:<12.4f} | {mixed_mean_val :<12.4f}| {h_val:<12.4f} | {nh_val:<15.4f}")

    # Optional: Save these indices
    np.save("top_k_mixed_gemma_variance_neurons.npy", top_k_indices)
    print("\n✅ Saved Top-K indices to 'top_k_mixed_variance_neurons.npy'")

if __name__ == "__main__":
    analyze_mixed_variance()