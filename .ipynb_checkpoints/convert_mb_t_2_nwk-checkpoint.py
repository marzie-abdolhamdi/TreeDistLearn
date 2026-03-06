from Bio import Phylo
import sys

# Replace 'your_mrbayes_run.t' with your actual MrBayes file name
input_file = "data/DS1_synth_1000/seq_gen_DS1_for_mb.nex.run2.t"
output_file = "data/DS1_synth_1000/mrbayes_DS1_seqgen_run2_t.nwk"

print(f"Reading MrBayes trees from {input_file}...")
# Parse the NEXUS file (this automatically handles the translate block and strips [&U] tags)
trees = list(Phylo.parse(input_file, "nexus"))

print(f"Converting {len(trees)} trees to standard Newick format...")
# Write them out as pure Newick strings
Phylo.write(trees, output_file, "newick")

print(f"Done! Saved to {output_file}")