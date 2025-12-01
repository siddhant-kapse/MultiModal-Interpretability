import numpy as np
import scipy.stats as stats
import os

# --- CONFIGURATION ---
HATE_FILE = "raw_hate_activations_66samples.npy"
NON_HATE_FILE = "raw_nohate_activations_113samples.npy"
TARGET_NEURONS = [1874, 1819, 1495, 1230, 1979, 571, 465, 514, 132, 684] 

def cohens_d(x, y):
    """Calculates Effect Size (How big is the difference?)"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def run_significance_test():
    # 1. Load Data
    try:
        data_hate = np.load(HATE_FILE, allow_pickle=True)
        data_nonhate = np.load(NON_HATE_FILE, allow_pickle=True)
        
        def get_vecs(d):
            if d.dtype.names and 'Activation Vector' in d.dtype.names:
                return d['Activation Vector']
            return np.stack([item[1] for item in d])

        A_H = get_vecs(data_hate)     
        A_NH = get_vecs(data_nonhate) 
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- CRITICAL STATISTICAL CORRECTION ---
    # Bonferroni Correction: Divides alpha by number of tests
    # If we test 10 neurons, we need p < 0.005, not 0.05, to be sure.
    num_tests = len(TARGET_NEURONS)
    alpha = 0.05
    corrected_alpha = alpha / num_tests

    print(f"\n--- Statistical Significance Analysis (N={num_tests}) ---")
    print(f"Standard Alpha: {alpha}")
    print(f"Bonferroni Corrected Alpha: {corrected_alpha:.5f} (Must be lower than this to be real)")
    print("-" * 100)
    print(f"{'Neuron':<8} | {'T-Test (p)':<12} | {'MW-U (p)':<12} | {'Effect (d)':<10} | {'Result'}")
    print("-" * 100)

    for neuron_idx in TARGET_NEURONS:
        # Get data
        hate_vals = A_H[:, neuron_idx]
        nonhate_vals = A_NH[:, neuron_idx]

        # Test 1: Welch's T-Test (Standard, assumes normality)
        t_stat, p_t = stats.ttest_ind(hate_vals, nonhate_vals, equal_var=False)

        # Test 2: Mann-Whitney U (Robust, safer for ReLU/Sparse data)
        u_stat, p_u = stats.mannwhitneyu(hate_vals, nonhate_vals, alternative='two-sided')

        # Measure Effect Size (Cohen's d)
        # 0.2 = Small, 0.5 = Medium, 0.8 = Large
        d_val = cohens_d(hate_vals, nonhate_vals)

        # Decision Logic
        # We generally trust the Mann-Whitney U more for neurons, but checking both is robust.
        # We require p < corrected_alpha (Bonferroni)
        
        sig_marker = ""
        if p_u < corrected_alpha:
            sig_marker = "SIGNIFICANT ✅"
        else:
            sig_marker = "Not Sig. ❌"

        # Formatting
        p_t_str = f"{p_t:.2e}" if p_t < 0.001 else f"{p_t:.4f}"
        p_u_str = f"{p_u:.2e}" if p_u < 0.001 else f"{p_u:.4f}"
        
        print(f"{neuron_idx:<8} | {p_t_str:<12} | {p_u_str:<12} | {d_val:<10.2f} | {sig_marker}")

    print("-" * 100)
    print("Note: 'MW-U' (Mann-Whitney) is safer if your neuron activations are sparse (lots of zeros).")
    print("Note: 'Effect (d)' tells you how strong the difference is. d > 0.8 is huge.")

if __name__ == "__main__":
    run_significance_test()