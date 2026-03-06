import argparse
import math
from collections import Counter
import dendropy

def get_topology_distribution(tree_list: dendropy.TreeList) -> dict:
    """
    Converts a list of trees into a normalized probability distribution of topologies.
    Uses bipartition bitmasks, which are mathematically invariant to tree rotation and rooting.
    """
    freqs = Counter()
    for tree in tree_list:
        # 1. Standardize the tree: resolve polytomies and treat as unrooted
        tree.resolve_polytomies()
        tree.collapse_basal_bifurcation()
        
        # 2. Mathematically map the branches to the TaxonNamespace AFTER collapsing the root
        tree.encode_bipartitions()
        
        # 3. Extract the unique integer bitmask for each split, sort them, and tuple them!
        topo_hash = tuple(sorted(b.split_bitmask for b in tree.bipartition_encoding))
        
        freqs[topo_hash] += 1
        
    total_trees = sum(freqs.values())
    return {topo: count / total_trees for topo, count in freqs.items()}

def compute_distances(dict_q: dict, dict_p: dict, epsilon: float = 1e-9) -> tuple:
    """
    Computes KL Divergence KL(Q||P) and Total Variation Distance between two distributions.
    Q = VI Approximation (Your model)
    P = Target Posterior (MrBayes)
    """
    all_topologies = set(dict_q.keys()).union(set(dict_p.keys()))
    
    kl_div = 0.0
    tvd = 0.0
    
    for topo in all_topologies:
        # Add epsilon smoothing to prevent log(0) or division by zero
        # if one method sampled a tree the other completely missed
        q_i = dict_q.get(topo, 0.0) + epsilon
        p_i = dict_p.get(topo, 0.0) + epsilon
        
        # Normalize slightly altered probabilities
        q_i = q_i / (1.0 + len(all_topologies) * epsilon)
        p_i = p_i / (1.0 + len(all_topologies) * epsilon)
        
        kl_div += q_i * math.log(q_i / p_i)
        tvd += 0.5 * abs(q_i - p_i)
        
    return kl_div, tvd

def main(args):
    print("Loading taxa namespace to ensure exact alignment...")
    tns = dendropy.TaxonNamespace()
    
    # 1. Load MrBayes Trees
    print(f"Loading MrBayes posterior from: {args.mrbayes_file}")
    # dendropy automatically handles the TRANSLATE block in MrBayes nexus files
    mb_trees = dendropy.TreeList.get(
        path=args.mrbayes_file, 
        schema="nexus", 
        taxon_namespace=tns
    )
    
    # Discard burn-in (typically 25% of MCMC samples)
    burnin_idx = int(len(mb_trees) * args.burnin_frac)
    mb_trees = mb_trees[burnin_idx:]
    print(f"  -> Retained {len(mb_trees)} trees after {args.burnin_frac*100}% burn-in.")
    
    # 2. Load VI Trees
    print(f"Loading CA-VI sampled trees from: {args.vi_file}")
    vi_trees = dendropy.TreeList.get(
        path=args.vi_file, 
        schema="nexus", 
        taxon_namespace=tns
    )
    print(f"  -> Loaded {len(vi_trees)} trees.")
    
    # 3. Encode Bipartitions
    # This mathematically maps the branches to the TaxonNamespace
    # print("Encoding bipartitions (identifying unique topologies)...")
    # dendropy.TreeList.encode_bipartitions(mb_trees)
    # dendropy.TreeList.encode_bipartitions(vi_trees)
    
    # 4. Compute Distributions
    mb_dist = get_topology_distribution(mb_trees)
    vi_dist = get_topology_distribution(vi_trees)
    
    print(f"  -> MrBayes found {len(mb_dist)} unique topologies.")
    print(f"  -> CA-VI found {len(vi_dist)} unique topologies.")
    
    # 5. Calculate Metrics
    kl_div, tvd = compute_distances(vi_dist, mb_dist)
    
    print("\n" + "="*50)
    print(" EVALUATION METRICS")
    print("="*50)
    print(f" KL Divergence (VI || MB) : {kl_div:.6f} nats")
    print(f" Total Variation Distance : {tvd:.6f}")
    print("="*50)
    
    # Optional: Print Top 3 topologies overlap
    print("\nTop 3 Topologies in MrBayes:")
    mb_top = sorted(mb_dist.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (topo, prob) in enumerate(mb_top):
        vi_prob = vi_dist.get(topo, 0.0)
        print(f"  {i+1}. MB: {prob*100:05.2f}%  |  VI: {vi_prob*100:05.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare VI and MrBayes posteriors")
    parser.add_argument("--mrbayes_file", type=str, required=True, help="Path to MrBayes .t file")
    parser.add_argument("--vi_file", type=str, required=True, help="Path to VI sampled_trees.nex")
    parser.add_argument("--burnin_frac", type=float, default=0.25, help="Fraction of MB trees to discard")
    args = parser.parse_args()
    main(args)