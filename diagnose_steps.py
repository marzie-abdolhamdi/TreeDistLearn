import os
import argparse
import torch
import dendropy
from dendropy.calculate import treecompare
import matplotlib.pyplot as plt

# Import your core engine (Ensure your main file is named phylo_vi.py)
import edge_inserter_final_fixed_elbo_Feb16_last_phylo_VI as vi

def load_step_models_from_dir(run_dir, names):
    """Reconstructs the step_models dictionary from the saved directory."""
    step_models = {}
    for k in range(3, len(names)):
        d = os.path.join(run_dir, f"train_step_{k:03d}_{names[k]}")
        rho_path = os.path.join(d, "rho_net.pt")
        b_path = os.path.join(d, "b_net.pt")
        if os.path.exists(rho_path) and os.path.exists(b_path):
            step_models[k] = (rho_path, b_path)
        else:
            break
    return step_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Path to FASTA file")
    parser.add_argument("--mb_trees", required=True, help="Path to MrBayes posterior (.nwk)")
    parser.add_argument("--run_dir", required=True, help="Path to your results_ca_vi/run_... folder")
    parser.add_argument("--n_samples", type=int, default=100, help="Trees to sample per step")
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data and recreate your model environment
    names, seqs = vi.parse_fasta_in_order(args.fasta)
    leaf_lik_all = {i: vi.seq_to_likelihood_matrix(seqs[i]) for i in range(len(names))}
    e0, e1, e2 = vi.compute_t3_branch_lengths(seqs[0], seqs[1], seqs[2])
    
    # Dummy args object to pass to your sampling functions
    vi_args = argparse.Namespace(
        edge_len_0=e0, edge_len_1=e1, edge_len_2=e2,
        k_eval=100, jc_rate=1.0, sample_temp=2.0, hidden_dim=args.hidden_dim
    )

    print("Loading trained networks...")
    step_models = load_step_models_from_dir(args.run_dir, names)
    step_nets = vi.preload_step_nets_ca(step_models, args.hidden_dim, device)

    # LOAD MRBAYES TREES ONCE
    print("Loading MrBayes posterior (this takes a few seconds)...")
    mb_trees_raw = dendropy.TreeList.get(
        path=args.mb_trees, 
        schema="newick",
        preserve_underscores=True,
        rooting="force-rooted"  # Prevents DendroPy from losing track of the root
    )

    # Prepare tracking arrays
    steps_k = []
    exact_match_pct = []
    mean_rf_dist = []

    print("\nStarting Step-by-Step Pruning Analysis...")
    
    # Analyze every step from 4 taxa up to the full tree
    for k in range(4, len(names) + 1):
        names_k = names[:k]
        tns = dendropy.TaxonNamespace(names_k)
        
        # ── A. Build the Ground Truth Oracle for Step k ──
        print(f"\n[Step {k}] Pruning MrBayes trees to {k} taxa ({names_k[-1]})...")
        
        oracle_trees = dendropy.TreeList(taxon_namespace=tns)
        for t in mb_trees_raw:
            # Create a clone so we don't destroy the raw trees for the next step
            sub_t = t.clone(depth=1)
            
            # FIX: Temporarily root the tree at Taxon 0 to protect the root node from being pruned
            anchor = sub_t.find_node_with_taxon_label(names_k[0])
            if anchor and anchor.edge:
                sub_t.reroot_at_edge(anchor.edge, update_bipartitions=False)
            
            # Now it is safe to prune
            sub_t.retain_taxa_with_labels(names_k)
            sub_t.suppress_unifurcations() # Merge broken edges
            sub_t.is_rooted = False        # Make strictly unrooted again
            
            # Reparse into the new namespace so treecompare works cleanly
            clean_str = sub_t.as_string("newick").strip()
            oracle_trees.append(dendropy.Tree.get(data=clean_str, schema="newick", taxon_namespace=tns))
            
        for t in oracle_trees:
            t.encode_bipartitions()
        
        # Extract unique topologies using frozen sets (DendroPy 4.x safe)
        unique_oracle = {}
        for t in oracle_trees:
            # Handle different DendroPy versions (some return objects, some return integers)
            bitmasks = frozenset(
                b.split_bitmask if hasattr(b, 'split_bitmask') else b 
                for b in t.bipartition_encoding
            )
            unique_oracle[bitmasks] = t
        oracle_unique_list = list(unique_oracle.values())
        print(f"         Found {len(oracle_unique_list)} unique Ground Truth topologies.")

        # ── B. Sample Trees from VI Model stopping at Step k ──
        print(f"         Sampling {args.n_samples} trees from VI model...")
        vi_trees = dendropy.TreeList(taxon_namespace=tns)
        for _ in range(args.n_samples):
            # We construct a custom sampling loop that STOPS at k
            allocator = vi.InternalNodeAllocator(start=len(names))
            tree = vi.build_t3_star(e0, e1, e2, center_idx=allocator.alloc())
            
            with torch.no_grad():
                for taxon_idx in range(3, k):
                    rho_net, b_net = step_nets[taxon_idx]
                    edges, q_e, _, mu_r, lsr, _, _, h, _, _, _, _, _, _, _ = vi.evaluate_edges_ca(
                        tree, leaf_lik_all[taxon_idx], leaf_lik_all,
                        rho_net, b_net, device, k_samples=vi_args.k_eval, jc_rate=1.0, temperature=2.0
                    )
                    chosen_edge, chosen_rho, chosen_b = vi.choose_action_ca(
                        edges, q_e, mu_r, lsr, h, b_net, device, sample_continuous=True
                    )
                    vi.insert_taxon_on_edge(tree, chosen_edge, taxon_idx, chosen_rho, chosen_b, allocator)
            
            # Convert to DendroPy via Newick string
            nwk = vi.to_newick(tree, names_k)
            dt = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=tns, preserve_underscores=True)
            dt.is_rooted = False
            vi_trees.append(dt)

        for t in vi_trees:
            t.encode_bipartitions()

        # ── C. Compare VI against Oracle ──
        matches = 0
        rf_distances = []
        for vt in vi_trees:
            min_rf = float('inf')
            for ot in oracle_unique_list:
                rf = treecompare.symmetric_difference(vt, ot)
                if rf < min_rf:
                    min_rf = rf
            if min_rf == 0:
                matches += 1
            rf_distances.append(min_rf)

        match_pct = (matches / args.n_samples) * 100
        avg_rf = sum(rf_distances) / len(rf_distances)
        
        steps_k.append(k)
        exact_match_pct.append(match_pct)
        mean_rf_dist.append(avg_rf)
        
        print(f"         Result: {match_pct:.1f}% Exact Matches | Average RF Distance: {avg_rf:.2f}")

    # ── D. Plot the Trajectory ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(steps_k, exact_match_pct, 'go-', linewidth=2, label="Exact Match to MrBayes")
    ax1.set_xlabel("Number of Taxa Inserted (Step k)")
    ax1.set_ylabel("% of Samples Matching Oracle Topology", color='green')
    ax1.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps_k, mean_rf_dist, 'rs--', linewidth=2, label="Mean RF Distance")
    ax2.set_ylabel("Robinson-Foulds Distance (Lower is better)", color='red')
    
    plt.title("Step-by-Step Diagnostic: Where does the model fail?")
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "step_by_step_diagnosis_fixed.png"), dpi=150)
    print(f"\nDiagnosis complete! Plot saved to {args.run_dir}/step_by_step_diagnosis_fixed.png")

if __name__ == "__main__":
    main()