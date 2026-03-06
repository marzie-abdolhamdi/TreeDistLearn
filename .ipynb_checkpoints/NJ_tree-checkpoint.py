from Bio import AlignIO
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import matplotlib.pyplot as plt

# --- STEP 1: LOAD THE SEQUENCES ---
input_file = "data/seq_gen_1000_toy_data_5.fasta"
try:
    # Changed to "fasta" as per your file extension
    alignment = AlignIO.read(input_file, "fasta") 
    print(f"Loaded alignment with {len(alignment)} sequences.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# --- STEP 2: CALCULATE DISTANCE MATRIX ---
print("Calculating distance matrix...")

# FIX: Use 'identity' instead of 'kimura'
# This calculates 1 - (percent identity)
calculator = DistanceCalculator('identity') 
dm = calculator.get_distance(alignment)
print(dm) 

# --- STEP 3: BUILD THE TREE (NJ) ---
print("Building Neighbor-Joining tree...")
constructor = DistanceTreeConstructor()
tree = constructor.nj(dm)

# --- STEP 4: VISUALIZE AND SAVE ---
print("\nASCII Tree:")
Phylo.draw_ascii(tree)

output_tree = "data/nj_seq_gen_1000_toy_data_5.nwk"
Phylo.write(tree, output_tree, "newick")
print(f"\nTree saved to {output_tree}")

# Optional: Plot
try:
    fig = plt.figure(figsize=(10, 5), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes, do_show=False)
    plt.title("Neighbor-Joining Tree")
    plt.savefig("nj_tree_plot.png")
    print("Tree image saved to nj_tree_plot.png")
except Exception as e:
    print(f"Could not save image (check matplotlib installation): {e}")