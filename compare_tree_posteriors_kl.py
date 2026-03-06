import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, Tuple, List

import numpy as np
import dendropy
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Parsing Helpers ----------

def canon_label(x: str) -> str:
    return x.strip().strip("'\"").replace(" ", "_")


def relabel_tree_inplace(tree):
    for nd in tree.leaf_node_iter():
        if nd.taxon is not None and nd.taxon.label is not None:
            nd.taxon.label = canon_label(nd.taxon.label)


def topology_key_unrooted(tree):
    tree.is_rooted = False
    tree.encode_bipartitions()
    full_mask = tree.taxon_namespace.all_taxa_bitmask()
    splits = []
    for b in tree.bipartition_encoding:
        if b.is_trivial():
            continue
        x = b.split_bitmask
        x = min(x, full_mask ^ x)
        splits.append(x)
    return tuple(sorted(splits))


# ---------- Loading Functions ----------

def load_mrbayes_posterior(trprobs_path: str, taxa) -> Dict[tuple, float]:
    mb = dendropy.TreeList.get(
        path=trprobs_path,
        schema="nexus",
        taxon_namespace=taxa,
        preserve_underscores=True,
        rooting="force-unrooted",
    )
    
    topo_weights = defaultdict(float)
    total_w = 0.0

    for t in mb:
        relabel_tree_inplace(t)
        key = topology_key_unrooted(t)
        
        # DendroPy usually stores trprobs weights in tree.weight
        w = getattr(t, 'weight', 1.0)
        if w is None:
            w = 1.0
            
        topo_weights[key] += float(w)
        total_w += float(w)

    if total_w == 0.0:
        raise ValueError(f"No trees or zero weight in: {trprobs_path}")

    return {k: v / total_w for k, v in topo_weights.items()}


def load_ca_vi_samples(sampled_path: str, taxa, schema="nexus") -> Dict[tuple, float]:
    sampled = dendropy.TreeList.get(
        path=sampled_path,
        schema=schema,
        taxon_namespace=taxa,
        preserve_underscores=True,
        rooting="force-unrooted",
    )
    
    counts = Counter()
    for t in sampled:
        relabel_tree_inplace(t)
        key = topology_key_unrooted(t)
        counts[key] += 1

    n_total = sum(counts.values())
    if n_total == 0:
        raise ValueError(f"No trees found in: {sampled_path}")

    return {k: v / float(n_total) for k, v in counts.items()}


# ---------- Evaluation Functions ----------

def compute_kl_divergences(
    p: Dict[tuple, float],
    q: Dict[tuple, float],
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    
    keys = list(set(p.keys()) | set(q.keys()))
    if not keys:
        raise ValueError("Both distributions are empty.")

    p_raw = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_raw = np.array([q.get(k, 0.0) for k in keys], dtype=float)

    p_smooth = np.clip(p_raw, eps, None)
    q_smooth = np.clip(q_raw, eps, None)

    p_arr = p_smooth / p_smooth.sum()
    q_arr = q_smooth / q_smooth.sum()

    kl_fwd = float(np.sum(p_arr * (np.log(p_arr) - np.log(q_arr))))
    kl_rev = float(np.sum(q_arr * (np.log(q_arr) - np.log(p_arr))))
    neg_ent = float(np.sum(p_arr * np.log(p_arr)))

    return kl_fwd, kl_rev, neg_ent


def kl_divergence_intersection(p: Dict, q: Dict) -> Tuple[float, float, int]:
    """
    Computes the Forward and Reverse KL Divergence STRICTLY on the intersection 
    of the two distributions (shared topologies only).
    """
    shared_keys = list(set(p.keys()).intersection(set(q.keys())))
    
    if not shared_keys:
        return 0.0, 0.0, 0
        
    p_raw = np.array([p[k] for k in shared_keys], dtype=float)
    q_raw = np.array([q[k] for k in shared_keys], dtype=float)
    
    p_norm = p_raw / p_raw.sum()
    q_norm = q_raw / q_raw.sum()
    
    kl_fwd_int = float(np.sum(p_norm * (np.log(p_norm) - np.log(q_norm))))
    kl_rev_int = float(np.sum(q_norm * (np.log(q_norm) - np.log(p_norm))))
    
    return kl_fwd_int, kl_rev_int, len(shared_keys)


def compute_tvd(p: Dict[tuple, float], q: Dict[tuple, float]) -> float:
    keys = list(set(p.keys()) | set(q.keys()))
    tvd = 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
    return float(tvd)


def plot_tvd_components_histogram(
    p_mb: Dict, q_ca: Dict, out_dir: str
) -> None:
    """
    Plots a stacked histogram of the absolute probability differences |P(x) - Q(x)| 
    for every topology, colored by whether they are shared or unshared.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_topos = list(set(p_mb.keys()) | set(q_ca.keys()))
    
    shared_diffs = []
    unshared_diffs = []
    
    for t in all_topos:
        diff = abs(p_mb.get(t, 0.0) - q_ca.get(t, 0.0))
        # Check if the topology exists in BOTH dictionaries
        if t in p_mb and t in q_ca:
            shared_diffs.append(diff)
        else:
            unshared_diffs.append(diff)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot as a stacked histogram
    ax.hist(
        [shared_diffs, unshared_diffs], 
        bins=30, 
        stacked=True,
        color=['#4C72B0', '#C44E52'],  # Blue for Shared, Red for Unshared
        alpha=0.85, 
        edgecolor='black',
        label=['Shared Topologies (5)', 'Unshared Topologies (8)']
    )
    
    ax.set_xlabel("Absolute Probability Difference |P(x) - Q(x)|")
    ax.set_ylabel("Number of Topologies")
    ax.set_title("Histogram of TVD Components\n(Shared vs. Unshared Topologies)")
    ax.legend()
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, "tvd_components_histogram3.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")


def evaluate_intersection_only(p_mb: Dict[tuple, float], q_ca: Dict[tuple, float]):
    common_topologies = set(p_mb.keys()).intersection(set(q_ca.keys()))
    
    if not common_topologies:
        print("\n=== Intersection Analysis ===")
        print("No shared topologies found! Cannot compute intersection metrics.")
        return

    print(f"\n=== Intersection Analysis ===")
    print(f"Total MrBayes trees: {len(p_mb)}")
    print(f"Total CA-VI trees:   {len(q_ca)}")
    print(f"Shared topologies:   {len(common_topologies)}")
    
    p_shared_raw = np.array([p_mb[k] for k in common_topologies])
    q_shared_raw = np.array([q_ca[k] for k in common_topologies])
    
    if len(common_topologies) > 1:
        # Ignore warning if MrBayes array is flat (variance = 0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation, _ = pearsonr(p_shared_raw, q_shared_raw)
        print(f"Pearson Correlation (R) of shared trees: {correlation:.4f}")
    else:
        print("Pearson Correlation (R) of shared trees: N/A (only 1 shared tree)")
    
    # Assign readable IDs for printing
    sorted_commons = sorted(list(common_topologies), key=lambda x: p_mb[x], reverse=True)
    topo_names = {k: f"Topo_{i+1:02d}" for i, k in enumerate(sorted_commons)}

    print("\n--- Breakdown of Shared Trees ---")
    print(f"{'Topology ID':<15} | {'MrBayes Prob':<15} | {'CA-VI Prob':<15}")
    print("-" * 50)
    for k in sorted_commons:
        print(f"{topo_names[k]:<15} | {p_mb[k]:<15.4f} | {q_ca[k]:<15.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrbayes_trprobs", required=True)
    parser.add_argument("--sampled_trees", required=True)
    parser.add_argument("--cavi_schema", type=str, default="nexus", help="'nexus' or 'newick'")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--out_dir", type=str, default="comparison_plots")
    args = parser.parse_args()

    taxa = dendropy.TaxonNamespace()

    print("Loading MrBayes posterior...")
    p_mb = load_mrbayes_posterior(args.mrbayes_trprobs, taxa)
    print(f"  Unique topologies in MrBayes posterior: {len(p_mb)}")

    print("Loading CA-VI samples...")
    q_ca = load_ca_vi_samples(args.sampled_trees, taxa, schema=args.cavi_schema)
    print(f"  Unique topologies in CA-VI samples:     {len(q_ca)}")

    # 1. Full Space Analysis
    kl_fwd, kl_rev, neg_ent = compute_kl_divergences(p_mb, q_ca, eps=args.eps)
    tvd = compute_tvd(p_mb, q_ca)

    # 2. Intersection-Only KL
    kl_fwd_int, kl_rev_int, n_shared = kl_divergence_intersection(p_mb, q_ca)

    print("\n=== Full Space Metrics ===")
    print(f"Forward KL  KL(p_MB || q_CA)  : {kl_fwd:.6f} nats")
    print(f"Reverse KL  KL(q_CA || p_MB)  : {kl_rev:.6f} nats")
    print(f"Total Variation Distance (TVD): {tvd:.4f} (scale: 0.0 to 1.0)")
    
    if n_shared > 0:
        print("\n=== Intersection-Only Metrics (Shared Topologies) ===")
        print(f"Forward KL  KL(p_MB || q_CA)  : {kl_fwd_int:.6f} nats")
        print(f"Reverse KL  KL(q_CA || p_MB)  : {kl_rev_int:.6f} nats")

    evaluate_intersection_only(p_mb, q_ca)

    print("\nGenerating TVD components histogram...")
    plot_tvd_components_histogram(p_mb, q_ca, args.out_dir)

if __name__ == "__main__":
    main()