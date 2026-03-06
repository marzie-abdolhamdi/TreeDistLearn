import matplotlib.pyplot as plt

def parse_sampled_iqtree(filepath):
    """Extracts a list of log-likelihoods from a -z IQ-TREE output."""
    log_likelihoods = []
    in_tree_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            # Detect table start (case-insensitive check for 'logl')
            if line.startswith("Tree") and "logl" in line.lower():
                in_tree_section = True
                continue
            
            if in_tree_section:
                # Stop parsing if we hit an empty line after the table
                if not line.strip():
                    break
                
                # Skip the "--------" separator line
                if line.strip().startswith("---"):
                    continue
                    
                # Extract the data
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Verify col 1 is the Tree ID (int) and col 2 is LogL (float)
                        tree_num = int(parts[0]) 
                        logL = float(parts[1])
                        log_likelihoods.append(logL)
                    except ValueError:
                        pass
                        
    print(f"Parsed {len(log_likelihoods)} trees from {filepath}")
    return log_likelihoods

def parse_single_iqtree(filepath):
    """Extracts a single log-likelihood from a -te IQ-TREE output."""
    with open(filepath, 'r') as f:
        for line in f:
            if "Log-likelihood of the tree:" in line:
                # Extracts the value right after the colon (index 4)
                return float(line.split()[4])
    return None

# 1. Parse the files
# Replace these paths if your files are in a different directory
sampled_logLs = parse_sampled_iqtree('data/DS1_synth_1000/iqfiles/mb_eval_out.iqtree')
gt_logL = parse_single_iqtree('data/DS1_synth_1000/iqfiles/gt_out.iqtree')
trained_logL = parse_single_iqtree('data/DS1_synth_1000/iqfiles/trained_out.iqtree')

# 2. Create the Plot
plt.figure(figsize=(10, 6))

# Plot the histogram of sampled trees
if sampled_logLs:
    plt.hist(sampled_logLs, bins=300, color='skyblue', edgecolor='black', alpha=0.7, label='MB Trees')
else:
    print("WARNING: No sampled trees were parsed. Check the filepath and file formatting.")

# Add vertical lines for Ground Truth and Trained Trees
if gt_logL is not None:
    plt.axvline(x=gt_logL, color='green', linestyle='dashed', linewidth=2, label=f'Ground Truth ({gt_logL:.2f})')
if trained_logL is not None:
    plt.axvline(x=trained_logL, color='red', linestyle='dashed', linewidth=2, label=f'Trained Tree ({trained_logL:.2f})')

# Formatting
plt.title('Log-Likelihood Distribution of Sampled Trees vs. Reference Trees', fontsize=14)
plt.xlabel('Log-Likelihood', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
# Add this line right above plt.tight_layout()
# plt.xlim(-4330, -4280) # Adjust these numbers to match your first plot's spread
plt.xlim(-6500, -4000) # Adjust these numbers to match your first plot's spread
# Save and show
plt.tight_layout()
plt.savefig('likelihood_comparison_DS1_synth_1_mb.png', dpi=300)
print("Plot saved as 'likelihood_comparison_DS1_synth_1_mb.png'")