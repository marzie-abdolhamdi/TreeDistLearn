from Bio import Phylo

# Input and Output file paths
input_nexus = "results_ca_vi/run_20260305_142949_ntaxa27/sampled_trees.nex"
output_newick = "results_ca_vi/run_20260305_142949_ntaxa27/sampled_trees.nwk"

print(f"Converting {input_nexus} to Newick format...")

try:
    # Read the NEXUS file
    # Biopython automatically handles the 'TRANSLATE' block mapping numbers to names
    trees = list(Phylo.parse(input_nexus, "nexus"))
    
    # Write to a clean Newick file
    Phylo.write(trees, output_newick, "newick")
    
    print(f"Success! Converted {len(trees)} trees to '{output_newick}'.")
    
except Exception as e:
    print(f"Error during conversion: {e}")