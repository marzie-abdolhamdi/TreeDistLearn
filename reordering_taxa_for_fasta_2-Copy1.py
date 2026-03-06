import dendropy
import argparse
from typing import Dict

def get_tree_guided_order(tree_path: str):
    # 1. Load the Tree
    tree = dendropy.Tree.get(path=tree_path, schema="newick", preserve_underscores=True)
    
    # 2. Get the ACTUAL Taxon objects, not just their names
    taxa = [t for t in tree.taxon_namespace]
    
    # 3. Calculate the Distance Matrix
    pdm = tree.phylogenetic_distance_matrix()
    
    # 4. Find the diameter (two most distant taxa) using objects
    max_d = -1.0
    seeds = (None, None)
    for i in range(len(taxa)):
        for j in range(i + 1, len(taxa)):
            # Pass the taxon objects directly to pdm
            d = pdm(taxa[i], taxa[j]) 
            if d > max_d:
                max_d = d
                seeds = (taxa[i], taxa[j])
    
    ordered_taxa = [seeds[0], seeds[1]]
    remaining = set(taxa) - set(ordered_taxa)
    
    # 5. Max-Min logic using objects
    while remaining:
        best_taxon = None
        best_max_min_d = -1.0
        for candidate in remaining:
            min_d_to_set = min(pdm(candidate, existing) for existing in ordered_taxa)
            if min_d_to_set > best_max_min_d:
                best_max_min_d = min_d_to_set
                best_taxon = candidate
        
        ordered_taxa.append(best_taxon)
        remaining.remove(best_taxon)
    
    # 6. Return the string labels for FASTA matching
    return [t.label for t in ordered_taxa]

def main():
    parser = argparse.ArgumentParser(description="Reorder FASTA using Tree Branch Lengths")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Step 1: Get the order
    ordered_names = get_tree_guided_order(args.tree)

    # Step 2: Read FASTA with better name handling
    seq_data = {}
    with open(args.fasta, "r") as f:
        curr_name = None
        curr_seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if curr_name:
                    seq_data[curr_name] = "".join(curr_seq)
                # Clean name: remove '>' and split to take only the first ID
                curr_name = line[1:].split()[0]
                curr_seq = []
            else:
                curr_seq.append(line)
        if curr_name:
            seq_data[curr_name] = "".join(curr_seq)

    # Step 3: Write Output
    found_count = 0
    with open(args.output, "w") as f:
        for name in ordered_names:
            # Try exact match or underscore/space variations
            clean_name = name.replace(" ", "_")
            if name in seq_data:
                f.write(f">{name}\n{seq_data[name]}\n")
                found_count += 1
            elif clean_name in seq_data:
                f.write(f">{clean_name}\n{seq_data[clean_name]}\n")
                found_count += 1
            else:
                print(f"!!! Error: {name} from tree not found in FASTA!")

    print(f"Finished. Reordered {found_count} taxa into {args.output}")

if __name__ == "__main__":
    main()