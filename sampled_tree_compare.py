"""
compare_trees.py
================
Compare sampled trees from your variational method against MrBayes posterior.

Requirements:
    pip install dendropy matplotlib numpy scipy scikit-learn

Usage:
    # Topology metrics only (no RF/MDS):
    python compare_trees.py \
        --ours    results_run1/.../sampled_trees.nex \
        --mrbayes run.nex.trprobs \
        --out_dir comparison_plots

    # Full comparison including RF distances and MDS:
    python compare_trees.py \
        --ours     results_run1/.../sampled_trees.nex \
        --mrbayes  run.nex.trprobs \
        --mb_trees run.nex.t \
        --out_dir  comparison_plots \
        --top_k    20

Key fix vs previous version:
    Topology matching now uses dendropy bipartition sets instead of Newick
    string canonicalization. This correctly handles different traversal orders,
    rooting conventions, and numeric TRANSLATE IDs between the two files.
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Dendropy is now REQUIRED for topology matching ────────────────────────────
try:
    import dendropy
    from dendropy.calculate import treecompare
    HAS_DENDROPY = True
except ImportError:
    HAS_DENDROPY = False

EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEXUS / Newick parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_translate_map(nwk: str, translate_map: Dict[str, str]) -> str:
    """
    Apply a NEXUS TRANSLATE block to a newick string.
    Matches numeric (or word) tokens at leaf positions: after ( or , and
    before ) : or ,  — the only positions a leaf label can appear.
    Sorts by length descending to avoid partial-match collisions.
    """
    if not translate_map:
        return nwk
    for token in sorted(translate_map.keys(), key=len, reverse=True):
        name = translate_map[token]
        nwk = re.sub(
            r"(?<=[(,])" + re.escape(token) + r"(?=[):,])",
            name, nwk
        )
    return nwk


def _parse_translate_block(lines: List[str]) -> Dict[str, str]:
    block = " ".join(lines).rstrip(";")
    tmap: Dict[str, str] = {}
    for entry in block.split(","):
        parts = entry.strip().split()
        if len(parts) >= 2:
            tmap[parts[0]] = parts[1]
    return tmap


def parse_nexus_trees(path: str) -> List[str]:
    """
    Parse any NEXUS file and return a list of raw Newick strings
    (translate block already applied, branch lengths kept for dendropy).
    Handles MrBayes .t format with [&U] / [&R] flags.
    """
    newick_strings: List[str] = []
    translate_map: Dict[str, str] = {}
    in_trees_block = False
    in_translate = False
    translate_lines: List[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            u = s.upper()

            if u.startswith("BEGIN TREES"):
                in_trees_block = True
                continue
            if u.startswith("END;") and in_trees_block:
                in_trees_block = False
                continue
            if not in_trees_block:
                continue

            if u.startswith("TRANSLATE"):
                in_translate = True
                continue
            if in_translate:
                translate_lines.append(s)
                if s.endswith(";"):
                    in_translate = False
                    translate_map = _parse_translate_block(translate_lines)
                    translate_lines = []
                continue

            if u.startswith("TREE "):
                # Match newick after "=" optionally preceded by [&U] / [&R] etc.
                m = re.search(r"=\s*(?:\[.*?\])?\s*(\(.*\));?\s*$", s)
                if m:
                    nwk = m.group(1).rstrip(";")
                    nwk = apply_translate_map(nwk, translate_map)
                    newick_strings.append(nwk)

    return newick_strings


def parse_trprobs(path: str) -> List[Tuple[str, float]]:
    """
    Parse MrBayes .trprobs file.
    Returns list of (raw_newick_with_taxon_names, probability) pairs.
    Translate block is applied so names are human-readable.
    """
    result: List[Tuple[str, float]] = []
    translate_map: Dict[str, str] = {}
    in_trees_block = False
    in_translate = False
    translate_lines: List[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            u = s.upper()

            if u.startswith("BEGIN TREES"):
                in_trees_block = True
                continue
            if u.startswith("END;") and in_trees_block:
                in_trees_block = False
                continue
            if not in_trees_block:
                continue

            if u.startswith("TRANSLATE"):
                in_translate = True
                continue
            if in_translate:
                translate_lines.append(s)
                if s.endswith(";"):
                    in_translate = False
                    translate_map = _parse_translate_block(translate_lines)
                    translate_lines = []
                continue

            if u.startswith("TREE "):
                prob_m = re.search(r"p\s*=\s*([0-9.eE+\-]+)", s, re.IGNORECASE)
                nwk_m  = re.search(r"=\s*(?:\[.*?\])?\s*(\(.*\));?\s*$", s)
                if prob_m and nwk_m:
                    prob = float(prob_m.group(1))
                    nwk  = nwk_m.group(1).rstrip(";")
                    nwk  = apply_translate_map(nwk, translate_map)
                    result.append((nwk, prob))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Bipartition-based topology key  (the core fix)
# ─────────────────────────────────────────────────────────────────────────────

def build_shared_namespace(all_newicks: List[str]) -> "dendropy.TaxonNamespace":
    """
    Parse a sample of trees to collect all taxon labels into one shared
    TaxonNamespace. Required by dendropy for RF distance computation.
    """
    tns = dendropy.TaxonNamespace()
    for nwk in all_newicks[:20]:
        try:
            dendropy.Tree.get(
                data=nwk + ";", schema="newick",
                taxon_namespace=tns,
                preserve_underscores=True
            )
        except Exception:
            pass
    return tns


def newick_to_bipartition_key(
    nwk: str,
    tns: "dendropy.TaxonNamespace",
) -> Optional[FrozenSet[FrozenSet[str]]]:
    """
    Convert a Newick string to a frozenset of frozensets of taxon label strings.
    This is rooting-invariant and traversal-order-invariant — the correct way
    to define an unrooted tree topology identity.
    Returns None if parsing fails.
    """
    try:
        t = dendropy.Tree.get(
            data=nwk + ";", schema="newick",
            taxon_namespace=tns,
            preserve_underscores=True
        )
        t.is_rooted = False
        t.encode_bipartitions()
        bipartitions = []
        for edge in t.postorder_edge_iter():
            bp = edge.bipartition
            if bp is None:
                continue
            taxa_set = frozenset(
                tx.label for tx in bp.leafset_taxa(tns)
                if tx is not None
            )
            # Skip trivial bipartitions (leaf edges — single taxon sets)
            if 1 < len(taxa_set) < len(tns):
                bipartitions.append(taxa_set)
        return frozenset(bipartitions)
    except Exception:
        return None


def newicks_to_keys(
    newicks: List[str],
    tns: "dendropy.TaxonNamespace",
    desc: str = "",
) -> List[Optional[FrozenSet]]:
    """Batch-convert Newick strings to bipartition keys."""
    keys = []
    n_fail = 0
    for nwk in newicks:
        k = newick_to_bipartition_key(nwk, tns)
        keys.append(k)
        if k is None:
            n_fail += 1
    if n_fail:
        print(f"  [WARNING] {desc}: {n_fail}/{len(newicks)} trees failed to parse")
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Distribution alignment and metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_topology_distributions(
    ours_keys: List[Optional[FrozenSet]],
    mb_keys_probs: List[Tuple[Optional[FrozenSet], float]],
    smoothing: float = 1e-6,
) -> Tuple[Dict, Dict, List]:
    """
    Align ours (empirical from samples) and MrBayes (from .trprobs probabilities)
    over the union of all bipartition-key topologies.
    """
    valid_ours = [k for k in ours_keys if k is not None]
    ours_counts = Counter(valid_ours)
    n = len(valid_ours)
    ours_freq = {t: c / n for t, c in ours_counts.items()}

    mb_freq: Dict = {}
    for key, prob in mb_keys_probs:
        if key is None:
            continue
        mb_freq[key] = mb_freq.get(key, 0.0) + prob
    mb_total = sum(mb_freq.values())
    mb_freq = {k: v / mb_total for k, v in mb_freq.items()}

    all_topos = list(set(list(ours_freq.keys()) + list(mb_freq.keys())))

    ours_dist = {t: ours_freq.get(t, smoothing) for t in all_topos}
    mb_dist   = {t: mb_freq.get(t, smoothing)   for t in all_topos}

    ours_sum = sum(ours_dist.values())
    mb_sum   = sum(mb_dist.values())
    ours_dist = {t: v / ours_sum for t, v in ours_dist.items()}
    mb_dist   = {t: v / mb_sum   for t, v in mb_dist.items()}

    return ours_dist, mb_dist, all_topos


def kl_divergence(p: Dict, q: Dict, all_topos: List) -> float:
    return sum(p[t] * math.log(p[t] / (q[t] + EPS)) for t in all_topos if p[t] > 0)


def total_variation_distance(p: Dict, q: Dict, all_topos: List) -> float:
    return 0.5 * sum(abs(p[t] - q[t]) for t in all_topos)


def hellinger_distance(p: Dict, q: Dict, all_topos: List) -> float:
    return (1 / math.sqrt(2)) * math.sqrt(
        sum((math.sqrt(p[t]) - math.sqrt(q[t])) ** 2 for t in all_topos)
    )


def top_k_recall(ours_dist: Dict, mb_dist: Dict, k: int) -> float:
    mb_top  = set(sorted(mb_dist,   key=mb_dist.get,   reverse=True)[:k])
    our_top = set(sorted(ours_dist, key=ours_dist.get, reverse=True)[:k])
    return len(mb_top & our_top) / max(1, len(mb_top))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_topology_comparison(
    ours_dist: Dict, mb_dist: Dict, all_topos: List,
    out_dir: str, top_k: int = 20,
) -> None:
    ranked = sorted(all_topos, key=lambda t: mb_dist[t], reverse=True)[:top_k]
    x = np.arange(len(ranked))
    labels    = [f"T{i+1}" for i in range(len(ranked))]
    mb_vals   = [mb_dist[t]   for t in ranked]
    ours_vals = [ours_dist[t] for t in ranked]

    fig, ax = plt.subplots(figsize=(max(12, top_k * 0.7), 5))
    width = 0.38
    ax.bar(x - width/2, mb_vals,   width, label="MrBayes",    color="steelblue",  alpha=0.85)
    ax.bar(x + width/2, ours_vals, width, label="Our method", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Topology rank (by MrBayes probability, top {top_k})")
    ax.set_ylabel("Posterior probability")
    ax.set_title("Topology Posterior: MrBayes vs Our Method\n(bipartition-based matching)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "topology_comparison_bar.png"), dpi=150)
    plt.close()
    print("  Saved: topology_comparison_bar.png")


def plot_scatter(
    ours_dist: Dict, mb_dist: Dict, all_topos: List, out_dir: str,
) -> None:
    mb_vals   = np.array([mb_dist[t]   for t in all_topos])
    ours_vals = np.array([ours_dist[t] for t in all_topos])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mb_vals, ours_vals, alpha=0.5, s=20, color='steelblue')
    max_val = max(mb_vals.max(), ours_vals.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect agreement')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MrBayes posterior probability")
    ax.set_ylabel("Our method posterior probability")
    ax.set_title("Topology Probability Scatter\n(log-log; diagonal = perfect agreement)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "topology_scatter.png"), dpi=150)
    plt.close()
    print("  Saved: topology_scatter.png")


def plot_cumulative_coverage(
    ours_dist: Dict, mb_dist: Dict, out_dir: str,
) -> None:
    mb_sorted   = sorted(mb_dist.values(),   reverse=True)
    ours_sorted = sorted(ours_dist.values(), reverse=True)
    mb_cumsum   = np.cumsum(mb_sorted)
    ours_cumsum = np.cumsum(ours_sorted)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(mb_cumsum)+1),   mb_cumsum,
            label="MrBayes",    color="steelblue",  linewidth=2)
    ax.plot(range(1, len(ours_cumsum)+1), ours_cumsum,
            label="Our method", color="darkorange", linewidth=2)
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, label='95% coverage')
    ax.axhline(y=0.50, color='gray', linestyle=':',  linewidth=1, label='50% coverage')
    ax.set_xlabel("Number of topologies (ranked by probability)")
    ax.set_ylabel("Cumulative posterior probability")
    ax.set_title("Cumulative Topology Coverage\n(steeper = more concentrated = more certain)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_coverage.png"), dpi=150)
    plt.close()
    print("  Saved: cumulative_coverage.png")


def plot_rf_distances(
    ours_newicks: List[str],
    mb_newicks: List[str],
    tns: "dendropy.TaxonNamespace",
    out_dir: str,
    n_sample: int = 200,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(42)
    ours_s = [ours_newicks[i] for i in rng.choice(
        len(ours_newicks), min(n_sample, len(ours_newicks)), replace=False)]
    mb_s   = [mb_newicks[i]   for i in rng.choice(
        len(mb_newicks),   min(n_sample, len(mb_newicks)),   replace=False)]

    def load(newicks):
        trees = []
        for nwk in newicks:
            try:
                t = dendropy.Tree.get(
                    data=nwk + ";", schema="newick",
                    taxon_namespace=tns, preserve_underscores=True)
                t.is_rooted = False
                t.encode_bipartitions()
                trees.append(t)
            except Exception:
                pass
        return trees

    print("  Computing RF distances...")
    ours_trees = load(ours_s)
    mb_trees   = load(mb_s)

    def rf_between(a_trees, b_trees, max_pairs=2000):
        dists = []
        pairs = [(i, j) for i in range(len(a_trees)) for j in range(len(b_trees))]
        if len(pairs) > max_pairs:
            idx = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[k] for k in idx]
        for i, j in pairs:
            try:
                dists.append(treecompare.symmetric_difference(a_trees[i], b_trees[j]))
            except Exception:
                pass
        return dists

    rf_cross       = rf_between(ours_trees, mb_trees)
    rf_within_ours = rf_between(ours_trees, ours_trees)
    rf_within_mb   = rf_between(mb_trees,   mb_trees)

    bins = range(0, max(
        max(rf_cross,       default=0),
        max(rf_within_ours, default=0),
        max(rf_within_mb,   default=0)
    ) + 2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rf_within_mb,   bins=bins, alpha=0.5, density=True,
            label="RF within MrBayes",  color="steelblue")
    ax.hist(rf_within_ours, bins=bins, alpha=0.5, density=True,
            label="RF within ours",      color="darkorange")
    ax.hist(rf_cross,       bins=bins, alpha=0.5, density=True,
            label="RF ours vs MrBayes",  color="green")
    ax.set_xlabel("Robinson-Foulds distance (0 = identical topology)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"RF Distance Distributions\n"
        f"Mean RF(ours vs MB): {np.mean(rf_cross):.2f}  |  "
        f"Mean RF(within MB): {np.mean(rf_within_mb):.2f}  |  "
        f"Mean RF(within ours): {np.mean(rf_within_ours):.2f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rf_distance_histogram.png"), dpi=150)
    plt.close()
    print("  Saved: rf_distance_histogram.png")
    print(f"  Mean RF(ours vs MrBayes): {np.mean(rf_cross):.3f}")
    print(f"  Mean RF(within MrBayes):  {np.mean(rf_within_mb):.3f}")
    print(f"  Mean RF(within ours):     {np.mean(rf_within_ours):.3f}")
    return float(np.mean(rf_cross)), float(np.mean(rf_within_mb)), float(np.mean(rf_within_ours))


def plot_mds_tree_space(
    ours_newicks: List[str],
    mb_newicks: List[str],
    tns: "dendropy.TaxonNamespace",
    out_dir: str,
    n_sample: int = 100,
) -> None:
    try:
        from sklearn.manifold import MDS
    except ImportError:
        print("  [SKIP] MDS requires scikit-learn: pip install scikit-learn")
        return

    rng = np.random.default_rng(42)
    n_each = min(n_sample, len(ours_newicks), len(mb_newicks))
    ours_s = [ours_newicks[i] for i in rng.choice(len(ours_newicks), n_each, replace=False)]
    mb_s   = [mb_newicks[i]   for i in rng.choice(len(mb_newicks),   n_each, replace=False)]
    all_nwk = ours_s + mb_s
    source  = ["Ours"] * n_each + ["MrBayes"] * n_each

    trees = []
    for nwk in all_nwk:
        try:
            t = dendropy.Tree.get(
                data=nwk + ";", schema="newick",
                taxon_namespace=tns, preserve_underscores=True)
            t.is_rooted = False
            t.encode_bipartitions()
            trees.append(t)
        except Exception:
            trees.append(None)

    valid_idx   = [i for i, t in enumerate(trees) if t is not None]
    valid_trees = [trees[i] for i in valid_idx]
    valid_src   = [source[i] for i in valid_idx]
    n_valid = len(valid_trees)
    print(f"  Computing {n_valid}x{n_valid} RF matrix for MDS...")

    dm = np.zeros((n_valid, n_valid))
    for a in range(n_valid):
        for b in range(a+1, n_valid):
            try:
                d = treecompare.symmetric_difference(valid_trees[a], valid_trees[b])
                dm[a, b] = dm[b, a] = d
            except Exception:
                pass

    coords = MDS(n_components=2, dissimilarity='precomputed',
                 random_state=42, n_init=1).fit_transform(dm)

    fig, ax = plt.subplots(figsize=(8, 7))
    for color, label in [("darkorange", "Ours"), ("steelblue", "MrBayes")]:
        mask = np.array([s == label for s in valid_src])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=label, alpha=0.5, s=30)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title("MDS of Tree Space (RF distances)\nOverlap = similar posteriors")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mds_tree_space.png"), dpi=150)
    plt.close()
    print("  Saved: mds_tree_space.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare variational trees vs MrBayes (bipartition-based matching)"
    )
    parser.add_argument("--ours",      required=True,
                        help="Path to sampled_trees.nex (our method)")
    parser.add_argument("--mrbayes",   required=True,
                        help="Path to MrBayes .trprobs file")
    parser.add_argument("--mb_trees",  default=None,
                        help="Path to MrBayes .t file (for RF/MDS). Optional.")
    parser.add_argument("--out_dir",   default="comparison_plots")
    parser.add_argument("--top_k",     type=int,   default=20)
    parser.add_argument("--smoothing", type=float, default=1e-6,
                        help="Pseudocount for unseen topologies")
    parser.add_argument("--n_rf",      type=int,   default=200,
                        help="Trees to sample for RF distance histogram")
    parser.add_argument("--n_mds",     type=int,   default=100,
                        help="Trees per method for MDS plot")
    args = parser.parse_args()

    if not HAS_DENDROPY:
        print("ERROR: dendropy is required.  pip install dendropy")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Parse raw Newick strings ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Parsing our sampled trees:  {args.ours}")
    ours_newicks = parse_nexus_trees(args.ours)
    print(f"  Found {len(ours_newicks)} sampled trees")

    print(f"\n  Parsing MrBayes .trprobs:   {args.mrbayes}")
    mb_nwk_probs = parse_trprobs(args.mrbayes)
    mb_newicks_trprobs = [nwk for nwk, _ in mb_nwk_probs]
    print(f"  Found {len(mb_nwk_probs)} topology entries")
    print(f"  MrBayes prob sum: {sum(p for _, p in mb_nwk_probs):.4f}")

    # ── Build shared TaxonNamespace from ALL trees ─────────────────────────
    print(f"\n  Building shared TaxonNamespace...")
    tns = build_shared_namespace(ours_newicks + mb_newicks_trprobs)
    preview = [t.label for t in tns][:5]
    print(f"  Found {len(tns)} taxa: {preview}{'...' if len(tns) > 5 else ''}")

    # ── Convert to bipartition keys ────────────────────────────────────────
    print(f"\n  Converting trees to bipartition keys...")
    ours_keys = newicks_to_keys(ours_newicks, tns, desc="ours")
    mb_keys   = [
        (newick_to_bipartition_key(nwk, tns), prob)
        for nwk, prob in mb_nwk_probs
    ]
    mb_fail = sum(1 for k, _ in mb_keys if k is None)
    if mb_fail:
        print(f"  [WARNING] {mb_fail} MrBayes topologies failed bipartition conversion")

    n_ours_valid = sum(1 for k in ours_keys if k is not None)
    n_mb_valid   = sum(1 for k, _ in mb_keys if k is not None)
    print(f"  Valid ours: {n_ours_valid}/{len(ours_keys)}")
    print(f"  Valid MB:   {n_mb_valid}/{len(mb_keys)}")

    # ── Align distributions ────────────────────────────────────────────────
    print(f"\n  Aligning topology distributions (smoothing={args.smoothing})...")
    ours_dist, mb_dist, all_topos = compute_topology_distributions(
        ours_keys, mb_keys, smoothing=args.smoothing
    )

    thresh    = args.smoothing * 2
    n_union   = len(all_topos)
    n_ours_u  = sum(1 for t in all_topos if ours_dist[t] > thresh)
    n_mb_u    = sum(1 for t in all_topos if mb_dist[t]   > thresh)
    n_overlap = sum(1 for t in all_topos
                    if ours_dist[t] > thresh and mb_dist[t] > thresh)

    # ── Metrics ────────────────────────────────────────────────────────────
    kl_ours_mb = kl_divergence(ours_dist, mb_dist,   all_topos)
    kl_mb_ours = kl_divergence(mb_dist,   ours_dist, all_topos)
    tv         = total_variation_distance(ours_dist, mb_dist, all_topos)
    hellinger  = hellinger_distance(ours_dist, mb_dist, all_topos)
    recall_1   = top_k_recall(ours_dist, mb_dist, k=1)
    recall_5   = top_k_recall(ours_dist, mb_dist, k=5)
    recall_10  = top_k_recall(ours_dist, mb_dist, k=10)
    our_map    = max(ours_dist, key=ours_dist.get)
    mb_map     = max(mb_dist,   key=mb_dist.get)
    map_agree  = (our_map == mb_map)

    print(f"\n{'='*60}")
    print(f"  RESULTS  (bipartition-based topology matching)")
    print(f"{'='*60}")
    print(f"  Unique topologies (union):          {n_union}")
    print(f"  Topologies seen in our method:      {n_ours_u}")
    print(f"  Topologies in MrBayes:              {n_mb_u}")
    print(f"  Overlap (seen in both):             {n_overlap}")
    print(f"")
    print(f"  KL(ours || MrBayes):                {kl_ours_mb:.4f} nats")
    print(f"  KL(MrBayes || ours):                {kl_mb_ours:.4f} nats")
    print(f"  Total Variation Distance:           {tv:.4f}  (0=identical, 1=no overlap)")
    print(f"  Hellinger Distance:                 {hellinger:.4f}  (0=identical, 1=no overlap)")
    print(f"  Top-1 recall (MAP topology match):  {'YES' if recall_1 == 1.0 else 'NO'}")
    print(f"  Top-5 recall:                       {recall_5:.2f}")
    print(f"  Top-10 recall:                      {recall_10:.2f}")
    print(f"  MAP topology agreement:             {'YES' if map_agree else 'NO'}")
    print(f"  Our MAP topology prob (ours):       {ours_dist[our_map]:.4f}")
    print(f"  MB  MAP topology prob (MrBayes):    {mb_dist[mb_map]:.4f}")
    print(f"{'='*60}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print(f"\n  Generating plots...")
    plot_topology_comparison(ours_dist, mb_dist, all_topos, args.out_dir, top_k=args.top_k)
    plot_scatter(ours_dist, mb_dist, all_topos, args.out_dir)
    plot_cumulative_coverage(ours_dist, mb_dist, args.out_dir)

    # ── RF and MDS ─────────────────────────────────────────────────────────
    rf_cross_mean = rf_within_mb_mean = rf_within_ours_mean = None
    if args.mb_trees:
        print(f"\n  Parsing MrBayes .t file: {args.mb_trees}")
        mb_newicks_t = parse_nexus_trees(args.mb_trees)
        print(f"  Found {len(mb_newicks_t)} MrBayes MCMC samples")
        if mb_newicks_t:
            rf_cross_mean, rf_within_mb_mean, rf_within_ours_mean = plot_rf_distances(
                ours_newicks, mb_newicks_t, tns, args.out_dir, n_sample=args.n_rf
            )
            plot_mds_tree_space(
                ours_newicks, mb_newicks_t, tns, args.out_dir, n_sample=args.n_mds
            )
        else:
            print("  [WARNING] No trees parsed from .t file — check format")

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics = {
        "topology_matching":            "bipartition-based (correct)",
        "n_unique_topologies_union":    n_union,
        "n_unique_topologies_ours":     n_ours_u,
        "n_unique_topologies_mrbayes":  n_mb_u,
        "n_overlap":                    n_overlap,
        "KL_ours_given_mrbayes":        kl_ours_mb,
        "KL_mrbayes_given_ours":        kl_mb_ours,
        "total_variation_distance":     tv,
        "hellinger_distance":           hellinger,
        "top1_recall":                  recall_1,
        "top5_recall":                  recall_5,
        "top10_recall":                 recall_10,
        "map_topology_match":           map_agree,
        "our_map_prob":                 ours_dist[our_map],
        "mb_map_prob":                  mb_dist[mb_map],
    }
    if rf_cross_mean is not None:
        metrics["mean_rf_ours_vs_mrbayes"] = rf_cross_mean
        metrics["mean_rf_within_mrbayes"]  = rf_within_mb_mean
        metrics["mean_rf_within_ours"]     = rf_within_ours_mean

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {args.out_dir}/metrics.json")
    print(f"  All outputs saved to: {args.out_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()