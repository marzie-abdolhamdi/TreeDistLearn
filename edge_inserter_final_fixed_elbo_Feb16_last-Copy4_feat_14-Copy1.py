import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

EPS = 1e-12
PI_DNA = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
EDGE_FEATURE_DIM = 14  # 4 (DNA_uv) + 1 (scale_uv) + 4 (DNA_vu) + 1 (scale_vu) + 4 (DNA_new_leaf)
PRIOR_LAM_B = 10.0   # Exponential(10) prior on branch lengths — same as PhyloGFN / ARTree
                     # prior mean = 1/lambda = 0.1 substitutions/site


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def parse_fasta_in_order(path: str) -> Tuple[List[str], List[str]]:
    names, seqs = [], []
    curr_name, curr_seq = None, []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr_name is not None:
                    names.append(curr_name)
                    seqs.append("".join(curr_seq).upper())
                curr_name = line[1:].strip().split()[0]
                curr_seq = []
            else:
                curr_seq.append(line)
    if curr_name is not None:
        names.append(curr_name)
        seqs.append("".join(curr_seq).upper())

    if not names:
        raise ValueError(f"No FASTA records found in: {path}")

    L = len(seqs[0])
    for i, s in enumerate(seqs):
        if len(s) != L:
            raise ValueError(f"All sequences must have equal length. Record {i} has length {len(s)} != {L}")
    return names, seqs


def base_like(ch: str) -> List[float]:
    tbl = {
        "A": [1.0, 0.0, 0.0, 0.0], "C": [0.0, 1.0, 0.0, 0.0], "G": [0.0, 0.0, 1.0, 0.0], "T": [0.0, 0.0, 0.0, 1.0],
        "U": [0.0, 0.0, 0.0, 1.0], "R": [1.0, 0.0, 1.0, 0.0], "Y": [0.0, 1.0, 0.0, 1.0], "S": [0.0, 1.0, 1.0, 0.0],
        "W": [1.0, 0.0, 0.0, 1.0], "K": [0.0, 0.0, 1.0, 1.0], "M": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0, 1.0],
        "D": [1.0, 0.0, 1.0, 1.0], "H": [1.0, 1.0, 0.0, 1.0], "V": [1.0, 1.0, 1.0, 0.0], "N": [1.0, 1.0, 1.0, 1.0],
        "?": [1.0, 1.0, 1.0, 1.0], "-": [1.0, 1.0, 1.0, 1.0], ".": [1.0, 1.0, 1.0, 1.0],
    }
    return tbl.get(ch, [1.0, 1.0, 1.0, 1.0])


def seq_to_likelihood_matrix(seq: str) -> torch.Tensor:
    return torch.tensor([base_like(ch) for ch in seq], dtype=torch.float32)


@dataclass
class UnrootedTree:
    adj: Dict[int, List[int]] 
    edge_len: Dict[frozenset, float] 

    def neighbors(self, node: int) -> List[int]:
        return self.adj.get(node, []) 

    def length(self, u: int, v: int) -> float:
        return self.edge_len[frozenset((u, v))]

    def edges(self) -> List[Tuple[int, int]]:
        out = []
        for key in self.edge_len.keys():
            u, v = sorted(tuple(key))
            out.append((u, v))
        out.sort()
        return out

    def add_edge(self, u: int, v: int, t: float) -> None:
        self.adj.setdefault(u, [])
        self.adj.setdefault(v, [])
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)
        self.edge_len[frozenset((u, v))] = float(t)

    def remove_edge(self, u: int, v: int) -> float:
        key = frozenset((u, v))
        t = self.edge_len.pop(key)
        if u in self.adj:
            self.adj[u] = [x for x in self.adj[u] if x != v]
        if v in self.adj:
            self.adj[v] = [x for x in self.adj[v] if x != u]
        return t

    def copy(self) -> "UnrootedTree":
        return UnrootedTree(
            adj={k: list(v) for k, v in self.adj.items()},
            edge_len={k: float(v) for k, v in self.edge_len.items()},
        )


class InternalNodeAllocator:
    def __init__(self, start: int):
        self.next_id = start

    def alloc(self) -> int:
        out = self.next_id
        self.next_id += 1
        return out


def build_t3_star(edge_len_0: float, edge_len_1: float, edge_len_2: float, center_idx: int) -> UnrootedTree:
    adj = {0: [center_idx], 1: [center_idx], 2: [center_idx], center_idx: [0, 1, 2]}
    edge_len = {
        frozenset((0, center_idx)): edge_len_0,
        frozenset((1, center_idx)): edge_len_1,
        frozenset((2, center_idx)): edge_len_2,
    }
    return UnrootedTree(adj=adj, edge_len=edge_len)

def compute_t3_branch_lengths(seq1: str, seq2: str, seq3: str) -> Tuple[float, float, float]:
    """Calculates the exact branch lengths for the initial 3-taxon star tree."""
    
    # 1. Helper to calculate raw p-distance ignoring missing data
    def p_dist(sA: str, sB: str) -> float:
        mismatches, comparable = 0, 0
        valid_bases = set("ACGT")
        for a, b in zip(sA, sB):
            if a in valid_bases and b in valid_bases:
                comparable += 1
                if a != b:
                    mismatches += 1
        return mismatches / comparable if comparable > 0 else 0.75

    # 2. Helper to convert p-distance to Jukes-Cantor distance
    def jc_dist(p: float) -> float:
        p = min(p, 0.7499) 
        return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)

    # 3. Calculate all three pairwise JC distances
    d12 = jc_dist(p_dist(seq1, seq2))
    d13 = jc_dist(p_dist(seq1, seq3))
    d23 = jc_dist(p_dist(seq2, seq3))

    # 4. Apply the exact 3-taxon star tree formula
    e1 = (d12 + d13 - d23) / 2.0
    e2 = (d12 + d23 - d13) / 2.0
    e3 = (d13 + d23 - d12) / 2.0

    return max(1e-4, e1), max(1e-4, e2), max(1e-4, e3)


def insert_taxon_on_edge(
    tree: UnrootedTree,
    edge: Tuple[int, int],
    new_taxon: int,
    rho: float,
    b_len: float,
    allocator: InternalNodeAllocator,
) -> int:
    u, v = edge
    t_uv = tree.remove_edge(u, v)
    w = allocator.alloc()
    tree.add_edge(u, w, max(1e-8, rho * t_uv))
    tree.add_edge(w, v, max(1e-8, (1.0 - rho) * t_uv))
    tree.add_edge(w, new_taxon, max(1e-8, b_len))
    return w


def jc_transition(t: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    e = torch.exp(-4.0 * rate * t / 3.0)
    same = 0.25 + 0.75 * e
    diff = 0.25 - 0.25 * e
    p = torch.ones((4, 4), dtype=t.dtype, device=t.device) * diff
    idx = torch.arange(4, device=t.device)
    p[idx, idx] = same
    return p


def compute_messages_and_cavities(
    tree: UnrootedTree,
    leaf_lik: Dict[int, torch.Tensor],
) -> Tuple[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]], 
           Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]]:
    
    any_leaf = next(iter(leaf_lik.values()))
    n_sites = any_leaf.shape[0]
    device = any_leaf.device
    
    msg_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    cav_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def cavity(a: int, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (a, b)
        if key in cav_cache:
            return cav_cache[key]

        psi = leaf_lik[a].clone() if a in leaf_lik else torch.ones(n_sites, 4, dtype=torch.float32, device=device)
        acc_log_scale = torch.zeros(n_sites, dtype=torch.float32, device=device)
        
        for k in tree.neighbors(a):
            if k == b:
                continue
            m_k_a, m_log_scale = message(k, a)
            psi = psi * m_k_a
            acc_log_scale = acc_log_scale + m_log_scale
            
            scaler = psi.max(dim=-1, keepdim=True)[0].clamp_min(1e-30)
            psi = psi / scaler
            acc_log_scale = acc_log_scale + torch.log(scaler).squeeze(-1)

        cav_cache[key] = (psi, acc_log_scale)
        return psi, acc_log_scale

    def message(a: int, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (a, b)
        if key in msg_cache:
            return msg_cache[key]

        psi, acc_log_scale = cavity(a, b)
        t_ab = torch.tensor(tree.length(a, b), dtype=torch.float32, device=device)
        p = jc_transition(t_ab)
        m = psi @ p.T
        
        scaler = m.max(dim=-1, keepdim=True)[0].clamp_min(1e-30)
        m = m / scaler
        acc_log_scale = acc_log_scale + torch.log(scaler).squeeze(-1)
        
        msg_cache[key] = (m, acc_log_scale)
        return m, acc_log_scale

    for u, v in tree.edges():
        _ = message(u, v)
        _ = message(v, u)

    return msg_cache, cav_cache


def edge_feature(
    cav_uv, scale_uv,
    cav_vu, scale_vu,
    new_leaf_lik,
    n_nodes: int,
) -> torch.Tensor:
    cav_uv_n = cav_uv / (cav_uv.sum(dim=-1, keepdim=True) + EPS)
    cav_vu_n = cav_vu / (cav_vu.sum(dim=-1, keepdim=True) + EPS)
    y_n      = new_leaf_lik / (new_leaf_lik.sum(dim=-1, keepdim=True) + EPS)
    scale_uv_norm = (scale_uv / n_nodes).unsqueeze(-1)
    scale_vu_norm = (scale_vu / n_nodes).unsqueeze(-1)
    return torch.cat([cav_uv_n, scale_uv_norm, cav_vu_n, scale_vu_norm, y_n], dim=-1)


def build_edge_features(edges, cav_cache, new_leaf_lik, device, n_nodes: int):
    new_leaf_lik = new_leaf_lik.to(device)
    feats = []
    for (u, v) in edges:
        cav_uv, scale_uv = cav_cache[(u, v)]   # ← cavity at u excluding v
        cav_vu, scale_vu = cav_cache[(v, u)]   # ← cavity at v excluding u
        feat = edge_feature(
            cav_uv.to(device), scale_uv.to(device),
            cav_vu.to(device), scale_vu.to(device),
            new_leaf_lik,
            n_nodes=n_nodes,
        )
        feats.append(feat)
    return torch.stack(feats, dim=0)


def jc_transition_batched(t: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    e = torch.exp(-4.0 * rate * t / 3.0)
    same = 0.25 + 0.75 * e
    diff = 0.25 - 0.25 * e
    p = diff.view(-1, 1, 1).expand(-1, 4, 4).clone()
    idx = torch.arange(4, device=t.device)
    p[:, idx, idx] = same.unsqueeze(1).expand(-1, 4)
    return p


def insertion_loglik_from_edge_batched(
    tree: UnrootedTree,
    edge: Tuple[int, int],
    rho: torch.Tensor,
    b_len: torch.Tensor,
    new_leaf_lik: torch.Tensor,
    cav_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
    pi: torch.Tensor,
    rate: float,
) -> torch.Tensor:
    u, v = edge
    t_uv = torch.tensor(tree.length(u, v), dtype=torch.float32, device=rho.device)
    t_uw = rho * t_uv
    t_wv = (1.0 - rho) * t_uv

    psi_u_excl_v, log_scale_u = cav_cache[(u, v)]
    psi_v_excl_u, log_scale_v = cav_cache[(v, u)]
    
    psi_u_excl_v = psi_u_excl_v.to(rho.device)  
    psi_v_excl_u = psi_v_excl_u.to(rho.device)  
    log_scale_u = log_scale_u.to(rho.device)
    log_scale_v = log_scale_v.to(rho.device)
    
    new_leaf_lik = new_leaf_lik.to(rho.device)       
    pi = pi.to(rho.device)

    p_uw = jc_transition_batched(t_uw, rate=rate)  
    p_wv = jc_transition_batched(t_wv, rate=rate)  
    p_yw = jc_transition_batched(b_len, rate=rate) 

    m_u_to_w = torch.matmul(psi_u_excl_v.unsqueeze(0), p_uw.transpose(1, 2))
    m_v_to_w = torch.matmul(psi_v_excl_u.unsqueeze(0), p_wv.transpose(1, 2))
    m_y_to_w = torch.matmul(new_leaf_lik.unsqueeze(0), p_yw.transpose(1, 2))

    root_partial = m_u_to_w * m_v_to_w * m_y_to_w
    
    site_like = (root_partial * pi.view(1, 1, 4)).sum(dim=-1).clamp_min(1e-30)  
    
    log_site_like = torch.log(site_like) + log_scale_u.unsqueeze(0) + log_scale_v.unsqueeze(0)
    
    return log_site_like.sum(dim=-1)


# --- CNN to MLP Architecture ---

class CNNPosteriorNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        # Removed AdaptiveAvgPool1d to implement custom Masked Pooling
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4) 
        )
    
    def forward(self, x: torch.Tensor):
        # x is originally (Batch, Seq_Len, Channels), transposed to (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2) 
        
        # y_n (the normalized likelihood of the new leaf) occupies channels 10-13
        # Gap characters become exactly 0.25 uniformly. Real bases have higher max values.
        y_n = x[:, 10:14, :] 
        
        # Create a binary mask: 1.0 for informative sites, 0.0 for gaps
        mask = (y_n.max(dim=1)[0] > 0.26).float() # Shape: (Batch, Seq_Len)
        
        # Run convolutions
        h_seq = self.cnn(x) # Shape: (Batch, hidden, Seq_Len)
        
        # Apply Masked Average Pooling
        mask = mask.unsqueeze(1) # Shape: (Batch, 1, Seq_Len)
        valid_sites_count = mask.sum(dim=2).clamp_min(1.0) # Prevent division by zero
        
        # Zero out the gap vectors, sum across the sequence, and average by ONLY valid sites
        h = (h_seq * mask).sum(dim=2) / valid_sites_count # Shape: (Batch, hidden)

        out = self.mlp(h)
        
        mu_b = out[:, 0] 
        log_sigma_b = out[:, 1].clamp(-5.0, 3.0) 
        mu_r = out[:, 2] 
        log_sigma_r = out[:, 3].clamp(-5.0, 3.0) 
        return mu_b, log_sigma_b, mu_r, log_sigma_r


def exponential_kl_branch(mu: torch.Tensor, log_sigma: torch.Tensor,
                          lam: float = PRIOR_LAM_B) -> torch.Tensor:
    """
    KL( LogNormal(mu, sigma²) || Exponential(lam) )

    Closed-form derivation:
        KL = E_q[log q(b)] - E_q[log p(b)]
           = -H(LogNormal)  -  E_q[log(lam) - lam*b]
           = -(log_sigma + mu + 0.5*(1+log2pi))  -  (log(lam) - lam*E_q[b])
        where E_q[b] = exp(mu + sigma²/2)  [LogNormal mean]

    Fully differentiable w.r.t. mu and log_sigma.
    With lam=10: prior mean branch length = 1/lam = 0.1 (matches PhyloGFN/ARTree).
    """
    sigma2 = torch.exp(2.0 * log_sigma)                       # σ²
    e_b    = torch.exp(mu + 0.5 * sigma2)                     # E_q[b] = exp(μ + σ²/2)
    kl = (- log_sigma                                          # -log σ
          - mu                                                 # -μ
          - 0.5 * (1.0 + math.log(2.0 * math.pi))            # -½(1 + log2π)
          - math.log(lam)                                      # -log λ
          + lam * e_b)                                         # +λ · E_q[b]
    return kl

def gaussian_kl_standard(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """KL from N(mu, sigma^2) to N(0, 1) — used for rho only."""
    sigma2 = torch.exp(2.0 * log_sigma)
    return 0.5 * (mu.pow(2) + sigma2 - 1.0 - 2.0 * log_sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Log-density helpers for MLL importance weights
# ─────────────────────────────────────────────────────────────────────────────

def log_lognormal_density(b: float, mu: float, log_sigma: float) -> float:
    """
    Log density of LogNormal(mu, sigma²) at b > 0.
    log p(b) = -log(b) - log(sigma) - 0.5*log(2π) - 0.5*((log(b)-mu)/sigma)²
    """
    sigma = math.exp(log_sigma)
    log_b = math.log(max(b, 1e-30))
    return (- log_b
            - log_sigma
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((log_b - mu) / sigma) ** 2)


def log_logistic_normal_density(rho: float, mu_r: float, log_sigma_r: float) -> float:
    """
    Log density of the logit-normal distribution at rho in (0, 1).
    If z = logit(rho) ~ N(mu_r, sigma_r²), then by change of variables:
        log p(rho) = log N(logit(rho); mu_r, sigma_r²) - log(rho) - log(1-rho)
    The last two terms are the log absolute Jacobian of the logit transform.
    """
    sigma_r = math.exp(log_sigma_r)
    rho_c   = min(max(rho, 1e-6), 1.0 - 1e-6)
    z       = math.log(rho_c / (1.0 - rho_c))          # logit
    log_normal   = (- log_sigma_r
                    - 0.5 * math.log(2.0 * math.pi)
                    - 0.5 * ((z - mu_r) / sigma_r) ** 2)
    log_jacobian = - math.log(rho_c) - math.log(1.0 - rho_c)
    return log_normal + log_jacobian


def mc_expected_loglik_and_samples(
    tree: UnrootedTree,
    edges: List[Tuple[int, int]],
    mu_b: torch.Tensor,
    log_sigma_b: torch.Tensor,
    mu_r: torch.Tensor,
    log_sigma_r: torch.Tensor,
    new_leaf_lik: torch.Tensor,
    cav_cache: Dict[Tuple[int, int], torch.Tensor],
    pi: torch.Tensor,
    k_samples: int,
    jc_rate: float,
):
    n_edges = len(edges)
    device = mu_b.device
    std_b = torch.exp(log_sigma_b).unsqueeze(0)  
    std_r = torch.exp(log_sigma_r).unsqueeze(0)  
    mu_b_e = mu_b.unsqueeze(0)                   
    mu_r_e = mu_r.unsqueeze(0)                   

    eps_b = torch.randn((k_samples, n_edges), device=device, dtype=mu_b.dtype)
    eps_r = torch.randn((k_samples, n_edges), device=device, dtype=mu_r.dtype)
    z_b = mu_b_e + std_b * eps_b
    z_r = mu_r_e + std_r * eps_r
    b = torch.exp(z_b)         
    rho = torch.sigmoid(z_r)   

    loglik_mc_list = []
    for i, e in enumerate(edges):
        ll_k = insertion_loglik_from_edge_batched(
            tree=tree,
            edge=e,
            rho=rho[:, i],
            b_len=b[:, i],
            new_leaf_lik=new_leaf_lik,
            cav_cache=cav_cache,
            pi=pi,
            rate=jc_rate,
        )
        loglik_mc_list.append(ll_k.mean())

    loglik_mc = torch.stack(loglik_mc_list)

    rho_samples = rho.detach().cpu().numpy()  
    b_samples = b.detach().cpu().numpy()      
    return loglik_mc, rho_samples, b_samples


def node_label(node_id: int, taxon_names: List[str]) -> str:
    if node_id < len(taxon_names):
        return taxon_names[node_id]
    return f"i{node_id}"


def save_step_plots(
    out_dir: str,
    edge_labels: List[str],
    elbo_hist: List[float],
    ema_hist: np.ndarray,
    q_final: np.ndarray,
    elbo_final: np.ndarray,
    rho_samples_final: np.ndarray,
    b_samples_final: np.ndarray,
    entropy_hist: List[float],
    max_qe_hist: List[float],
    kl_b_hist: List[float],        # ← new
    kl_r_hist: List[float],        # ← new
    loglik_hist: List[float],      # ← new
    grad_norm_hist: List[float],   # ← new
    temp_start: float,             # ← new
    temp_end: float,               # ← new
    max_steps: int,                # ← new
    # Category 2
    mu_b_top_hist: List[float],
    sigma_b_top_hist: List[float],
    mu_r_top_hist: List[float],
    sigma_r_top_hist: List[float],
    mu_b_final: np.ndarray,
    log_sigma_b_final: np.ndarray,
    taxon_idx: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    steps = np.arange(1, len(elbo_hist) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, elbo_hist, alpha=0.35, label="ELBO (raw single-tree)")
    plt.plot(steps, ema_hist, linewidth=2, label="ELBO (Exponential Moving Avg)")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("Convergence Monitoring (EMA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbo_convergence.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, q_final)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge")
    plt.ylabel("q(e) = softmax(ELBO)")
    plt.title("Edge Posterior Derived from Exact Marginalization")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_posterior_qe.png"), dpi=150)
    plt.close()

    rel_elbo = elbo_final - np.max(elbo_final)
    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, rel_elbo)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge")
    plt.ylabel("ELBO - max(ELBO)")
    plt.title("Relative Edge ELBO Scores on Final Sampled Tree")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_elbo_scores.png"), dpi=150)
    plt.close()

    top_k = min(5, len(edge_labels))
    top_indices = np.argsort(q_final)[-top_k:]

    plt.figure(figsize=(8, 4))
    for i in top_indices:
        plt.hist(rho_samples_final[:, i], bins=30, alpha=0.30, density=True, label=f"post rho|{edge_labels[i]}")
    plt.xlabel("rho")
    plt.ylabel("posterior density")
    plt.title(f"Posterior for rho (Top {top_k} edges)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rho_posterior.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    for i in top_indices:
        plt.hist(b_samples_final[:, i], bins=30, alpha=0.30, density=True, label=f"post b|{edge_labels[i]}")
    plt.xlabel("b")
    plt.ylabel("posterior density")
    plt.title(f"Posterior for b (Top {top_k} edges)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "b_posterior.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, entropy_hist, color='green', linewidth=2)
    plt.xlabel("Training Iteration")
    plt.ylabel("Shannon Entropy")
    plt.title("Distribution Convergence (Entropy over time)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_trajectory.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, max_qe_hist, color='purple', linewidth=2)
    plt.xlabel("Training Iteration")
    plt.ylabel("Max q(e) (Confidence in Top Edge)")
    plt.title("Model Confidence Trajectory vs. Temperature Annealing")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_trajectory.png"), dpi=150)
    plt.close()

    # --- Plot 1: ELBO decomposition ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, loglik_hist, color='blue', alpha=0.7)
    axes[0].set_title("E[log p(data)] (likelihood term)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Nats")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, kl_b_hist, color='orange', label="KL_b (branch length)")
    axes[1].plot(steps, kl_r_hist, color='red',    label="KL_r (rho / position)")
    axes[1].set_title("KL Divergence Terms")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("KL (nats)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, elbo_hist, color='green', alpha=0.7)
    axes[2].set_title("ELBO = E[log p] - KL")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("ELBO / n_sites")
    axes[2].grid(alpha=0.3)

    plt.suptitle("ELBO Decomposition over Training")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbo_decomposition.png"), dpi=150)
    plt.close()

    # --- Plot 2: Gradient norm ---
    plt.figure(figsize=(8, 4))
    plt.plot(steps, grad_norm_hist, alpha=0.6, color='red', linewidth=1)
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Clip threshold (1.0)')
    clip_fraction = np.mean(np.array(grad_norm_hist) > 1.0) * 100
    plt.xlabel("Iteration")
    plt.ylabel("Gradient L2 Norm (pre-clip)")
    plt.title(f"Gradient Norm — {clip_fraction:.1f}% of steps were clipped")
    plt.legend()
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradient_norm.png"), dpi=150)
    plt.close()

    # --- Plot 3: Entropy vs temperature schedule ---
    temps = [
        temp_start * (temp_end / temp_start) ** ((i) / max(1, max_steps - 1))
        for i in range(len(steps))
    ]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, entropy_hist, color='green', linewidth=2, label='H(q) Shannon Entropy')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Shannon Entropy", color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    ax2.plot(steps, temps, color='orange', linestyle='--', linewidth=1.5, label='Temperature')
    ax2.set_ylabel("Temperature", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title("Entropy vs Temperature Schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_vs_temperature.png"), dpi=150)
    plt.close()

    # --- Category 2 Plot 1: Variational parameter trajectories ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, mu_b_top_hist, color='steelblue', linewidth=1.5)
    axes[0, 0].axhline(y=math.log(1.0 / PRIOR_LAM_B), color='red', linestyle='--',
                       linewidth=1.5, label=f'Prior mode (log(1/λ)={math.log(1.0/PRIOR_LAM_B):.2f})')
    axes[0, 0].set_title("mu_b of top edge (log branch-length space)")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("mu_b")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, sigma_b_top_hist, color='steelblue', linewidth=1.5)
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--',
                       linewidth=1.5, label='Prior std (1.0)')
    axes[0, 1].set_title("sigma_b of top edge (uncertainty in log-b)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("sigma_b")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(steps, mu_r_top_hist, color='darkorange', linewidth=1.5)
    axes[1, 0].axhline(y=0.0, color='red', linestyle='--',
                       linewidth=1.5, label='Prior mean (0 → rho=0.5)')
    axes[1, 0].set_title("mu_r of top edge (logit-rho space)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("mu_r (logit)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(steps, sigma_r_top_hist, color='darkorange', linewidth=1.5)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--',
                       linewidth=1.5, label='Prior std (1.0)')
    axes[1, 1].set_title("sigma_r of top edge (uncertainty in logit-rho)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("sigma_r")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle(f"Variational Parameter Trajectories — Taxon {taxon_idx} (Top Edge Each Step)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variational_params_trajectory.png"), dpi=150)
    plt.close()

    # --- Category 2 Plot 2: Implied branch lengths across all edges at convergence ---
    b_means = np.exp(mu_b_final)                        # median branch length per edge
    b_stds  = np.exp(mu_b_final) * np.exp(log_sigma_b_final)  # approx std in real space

    x = np.arange(len(edge_labels))
    plt.figure(figsize=(max(10, len(edge_labels) * 0.4), 5))
    plt.bar(x, b_means, yerr=b_stds, capsize=3,
            color='steelblue', alpha=0.7, label='mean ± std')
    plt.axhline(y=0.05, color='green', linestyle='--',
                linewidth=1.5, label='Typical short branch (0.05)')
    plt.axhline(y=0.3, color='orange', linestyle='--',
                linewidth=1.5, label='Typical long branch (0.3)')
    plt.axhline(y=1.0 / PRIOR_LAM_B, color='red', linestyle=':',
                linewidth=1.5, label=f'Prior mean (1/λ = {1.0/PRIOR_LAM_B:.3f})')
    plt.xticks(x, edge_labels, rotation=45, ha='right', fontsize=7)
    plt.ylabel("Branch length (substitutions/site)")
    plt.title(f"Inferred Branch Lengths per Edge — Taxon {taxon_idx} (Converged Model)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "branch_lengths_per_edge.png"), dpi=150)
    plt.close()

def escape_nexus_name(name: str) -> str:
    if any(c in name for c in " \t\n(),:;[]="):
        return f"'{name}'"
    return name


def to_newick(tree: UnrootedTree, taxon_names: List[str], label_mapper: Optional[Callable[[int], str]] = None) -> str:
    start_node = None
    for u in tree.adj:
        if u >= len(taxon_names):
            start_node = u
            break
    if start_node is None:
        start_node = next(iter(tree.adj.keys()), None)
        if start_node is None:
            return "();"

    visited = set()

    def build_substring(u: int) -> str:
        visited.add(u)
        children = [v for v in tree.neighbors(u) if v not in visited]

        if not children:
            if u >= len(taxon_names):
                raise RuntimeError(f"Internal node {u} became terminal in Newick traversal. Tree structure invalid.")
            return label_mapper(u) if label_mapper else taxon_names[u]

        subtrees = []
        for v in children:
            length = tree.length(u, v)
            child_str = build_substring(v)
            subtrees.append(f"{child_str}:{length:.6f}")
        return "(" + ",".join(subtrees) + ")"

    visited.add(start_node)
    root_subtrees = []
    for v in tree.neighbors(start_node):
        length = tree.length(start_node, v)
        child_str = build_substring(v)
        root_subtrees.append(f"{child_str}:{length:.6f}")
    return "(" + ",".join(root_subtrees) + ");"


def save_nexus_trees(trees: List[UnrootedTree], taxon_names: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n\n")
        f.write("BEGIN TAXA;\n")
        f.write(f"\tDIMENSIONS NTAX={len(taxon_names)};\n")
        f.write("\tTAXLABELS\n")
        for name in taxon_names:
            f.write(f"\t\t{escape_nexus_name(name)}\n")
        f.write("\t;\nEND;\n\n")

        f.write("BEGIN TREES;\n")
        f.write("\tTRANSLATE\n")
        for i, name in enumerate(taxon_names):
            sep = "," if i < len(taxon_names) - 1 else ""
            f.write(f"\t\t{i+1} {escape_nexus_name(name)}{sep}\n")
        f.write("\t;\n")

        def map_idx_to_nexus_id(idx: int) -> str:
            return str(idx + 1)

        for idx, t in enumerate(trees):
            nwk = to_newick(t, taxon_names, label_mapper=map_idx_to_nexus_id)
            f.write(f"\tTREE tree_{idx+1} = {nwk}\n")
        f.write("END;\n")

def evaluate_edges(
    tree: UnrootedTree,
    new_leaf_lik: torch.Tensor,
    leaf_lik_all: Dict[int, torch.Tensor],
    net: CNNPosteriorNet,
    device: torch.device,
    k_samples: int,
    jc_rate: float,
    temperature: float = 1.0,
):
    leaf_lik_current = {k: v for k, v in leaf_lik_all.items() if k in tree.adj and len(tree.adj[k]) == 1}

    expected_leaves = [k for k in tree.adj.keys() if k < len(leaf_lik_all)]
    assert len(leaf_lik_current) == len(expected_leaves), \
        f"Tree topology drift detected! Expected {len(expected_leaves)} leaves, but found {len(leaf_lik_current)}."

    with torch.no_grad():
        _, cav_cache = compute_messages_and_cavities(tree, leaf_lik_current)
        edges = tree.edges()
        n_nodes = len(tree.adj)
        edge_feats_raw = build_edge_features(edges=edges, cav_cache=cav_cache, new_leaf_lik=new_leaf_lik, device=device, n_nodes=n_nodes)

    edge_feats = edge_feats_raw.detach()   

    new_leaf_lik = new_leaf_lik.to(device)
    pi_dna = PI_DNA.to(device)

    mu_b, log_sigma_b, mu_r, log_sigma_r = net(edge_feats)

    loglik_mc, rho_samples, b_samples = mc_expected_loglik_and_samples(
        tree=tree, edges=edges, mu_b=mu_b, log_sigma_b=log_sigma_b, mu_r=mu_r,
        log_sigma_r=log_sigma_r, new_leaf_lik=new_leaf_lik, cav_cache=cav_cache,
        pi=pi_dna, k_samples=k_samples, jc_rate=jc_rate,
    )

    kl_b = exponential_kl_branch(mu_b, log_sigma_b)  # KL( LogNormal || Exp(10) )
    kl_r = gaussian_kl_standard(mu_r, log_sigma_r) # prior mean 0 for rho logit, fine
    
    elbo_e = loglik_mc - kl_b - kl_r
    q_e = torch.softmax(elbo_e / temperature, dim=0)
    
    return edges, q_e, elbo_e, mu_b, log_sigma_b, mu_r, log_sigma_r, rho_samples, b_samples, kl_b, kl_r, loglik_mc


def choose_action(
    edges: List[Tuple[int, int]],
    q_e: torch.Tensor,
    mu_b: torch.Tensor,
    log_sigma_b: torch.Tensor,
    mu_r: torch.Tensor,
    log_sigma_r: torch.Tensor,
    sample_continuous: bool = True
) -> Tuple[Tuple[int, int], float, float]:

    q_e = torch.nan_to_num(q_e, nan=0.0) 
    q_e = q_e / (q_e.sum() + EPS)        
    chosen_idx = torch.multinomial(q_e, 1).item()

    if sample_continuous:
        sigma_b = torch.exp(log_sigma_b[chosen_idx])
        z_b = torch.normal(mean=mu_b[chosen_idx], std=sigma_b)
        chosen_b = torch.exp(z_b).item()

        sigma_r = torch.exp(log_sigma_r[chosen_idx])
        z_r = torch.normal(mean=mu_r[chosen_idx], std=sigma_r)
        chosen_rho = torch.sigmoid(z_r).item()
    else:
        chosen_b = torch.exp(mu_b[chosen_idx]).item()
        chosen_rho = torch.sigmoid(mu_r[chosen_idx]).item()

    return edges[chosen_idx], float(chosen_rho), float(chosen_b)


def train_single_insertion_step(
    taxon_idx: int,
    step_nets: Dict[int, CNNPosteriorNet],
    new_leaf_lik: torch.Tensor,
    leaf_lik_all: Dict[int, torch.Tensor],
    taxon_names: List[str],
    device: torch.device,
    args: argparse.Namespace,
    step_out_dir: Optional[str] = None,
) -> Tuple[CNNPosteriorNet, Dict[str, object]]:
    
    n_use = len(taxon_names)
    net = CNNPosteriorNet(in_channels=EDGE_FEATURE_DIM, hidden=args.hidden_dim).to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    elbo_hist = []
    entropy_hist = []
    max_qe_hist = []
    kl_b_hist = []
    kl_r_hist = []
    loglik_hist = []
    grad_norm_hist = []
    # Category 2: variational parameter trajectories
    mu_b_top_hist = []
    sigma_b_top_hist = []
    mu_r_top_hist = []
    sigma_r_top_hist = []
    
    pbar = tqdm(total=args.max_steps, desc=f"Training Taxon {taxon_idx}", leave=False, dynamic_ncols=True)
    
    final_tree_sampled = None

    # Calculate effective sites for dynamic loss scaling
    # Any base vector that sums to exactly 4.0 is a pure gap [1,1,1,1]. 
    # We only count sites that sum to less than 3.9 as informative.
    y_n_raw = new_leaf_lik / (new_leaf_lik.sum(dim=-1, keepdim=True) + EPS)

    # effective_sites = max(1.0, (y_n_raw.max(dim=-1)[0] > 0.26).sum().float().item())
    for step_iter in range(1, args.max_steps + 1):

        progress = (step_iter - 1) / max(1, args.max_steps - 1)
        current_temp = args.temp_start * (args.temp_end / args.temp_start) ** progress
        allocator = InternalNodeAllocator(start=n_use)
        center_id = allocator.alloc()
        tree = build_t3_star(args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=center_id)
        
        with torch.no_grad():
            for step in range(3, taxon_idx):
                prev_net = step_nets[step]
                edges_past, q_e_past, _, mu_b_past, log_sigma_b_past, mu_r_past, log_sigma_r_past, _, _, _, _, _ = evaluate_edges(
                    tree, leaf_lik_all[step], leaf_lik_all, prev_net, device, k_samples=args.k_train, jc_rate=args.jc_rate, temperature=current_temp
                )
                chosen_edge, chosen_rho, chosen_b = choose_action(
                    edges_past, q_e_past, mu_b_past, log_sigma_b_past, mu_r_past, log_sigma_r_past, sample_continuous=True
                )
                insert_taxon_on_edge(tree, chosen_edge, step, chosen_rho, chosen_b, allocator)

        edges, q_e, elbo_e, mu_b, log_sigma_b, mu_r, log_sigma_r, _, _, kl_b, kl_r, loglik_mc = evaluate_edges(
            tree, new_leaf_lik, leaf_lik_all, net, device, k_samples=args.k_train, jc_rate=args.jc_rate, temperature=current_temp
        )

        n_edges = len(edges)

        log_sum_exp_term = current_temp * torch.logsumexp(elbo_e / current_temp, dim=0)
        tree_elbo = log_sum_exp_term - math.log(n_edges)

        max_qe_hist.append(q_e.max().item())
        current_entropy = -torch.sum(q_e * torch.log(q_e + EPS)).item()
        entropy_hist.append(current_entropy)
        # Track variational params of the top edge each step
        top_idx = q_e.argmax().item()
        mu_b_top_hist.append(mu_b[top_idx].detach().item())
        sigma_b_top_hist.append(torch.exp(log_sigma_b[top_idx]).detach().item())
        mu_r_top_hist.append(mu_r[top_idx].detach().item())
        sigma_r_top_hist.append(torch.exp(log_sigma_r[top_idx]).detach().item())
        # Track ELBO components
        kl_b_hist.append(kl_b.mean().item())
        kl_r_hist.append(kl_r.mean().item())
        loglik_hist.append(loglik_mc.mean().item())
        
        # Scale loss by total alignment length for consistent gradient magnitude across taxa
        n_sites = new_leaf_lik.shape[0]
        loss = -tree_elbo / n_sites

        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm BEFORE clipping
        raw_grad_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                raw_grad_norm += p.grad.norm().item() ** 2
        grad_norm_hist.append(raw_grad_norm ** 0.5)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) 
        optimizer.step()

        # Update logs using the newly scaled value
        val = (tree_elbo / n_sites).item()
        elbo_hist.append(val)

        pbar.update(1)
        pbar.set_postfix(elbo=f"{val:.3f}")

        if step_iter == args.max_steps:
            final_tree_sampled = tree

    pbar.close()

    summary = {
        "final_scaled_elbo":  elbo_hist[-1],
        "iterations":         args.max_steps,
        "final_entropy":      entropy_hist[-1],
        "final_max_qe":       max_qe_hist[-1],
        "final_kl_b":         kl_b_hist[-1],
        "final_kl_r":         kl_r_hist[-1],
        "final_loglik":       loglik_hist[-1],
        "n_edges":            len(edges),           # edges available here from last loop iter
    }

    if step_out_dir and final_tree_sampled is not None:
        with torch.no_grad():
            edges, q_e, elbo_e, mu_b, log_sigma_b, mu_r, log_sigma_r, rho_s, b_s, _, _, _ = evaluate_edges(
                final_tree_sampled, new_leaf_lik, leaf_lik_all, net, device, k_samples=args.k_eval, jc_rate=args.jc_rate, temperature=current_temp
            )
        edge_labels = [f"({node_label(u, taxon_names)}, {node_label(v, taxon_names)})" for (u, v) in edges]
        
        save_step_plots(
            out_dir=step_out_dir, edge_labels=edge_labels, elbo_hist=elbo_hist,
            ema_hist=np.array(elbo_hist), q_final=q_e.detach().cpu().numpy(),
            elbo_final=elbo_e.detach().cpu().numpy(),
            rho_samples_final=rho_s, b_samples_final=b_s,
            entropy_hist=entropy_hist, max_qe_hist=max_qe_hist,
            kl_b_hist=kl_b_hist, kl_r_hist=kl_r_hist,       # ← new
            loglik_hist=loglik_hist,                           # ← new
            grad_norm_hist=grad_norm_hist,                     # ← new
            temp_start=args.temp_start,                        # ← new
            temp_end=args.temp_end,                            # ← new
            max_steps=args.max_steps,                          # ← new
            mu_b_top_hist=mu_b_top_hist,
            sigma_b_top_hist=sigma_b_top_hist,
            mu_r_top_hist=mu_r_top_hist,
            sigma_r_top_hist=sigma_r_top_hist,
            mu_b_final=mu_b.detach().cpu().numpy(),
            log_sigma_b_final=log_sigma_b.detach().cpu().numpy(),
            taxon_idx=taxon_idx,
        )
        torch.save(net.state_dict(), os.path.join(step_out_dir, "cnn_posterior_net.pt"))

    return net, summary


def save_crosstep_plots(run_dir: str, step_meta: List[dict], names: List[str]) -> None:
    """Category 3: one plot per metric, one point per taxon insertion step."""
    os.makedirs(run_dir, exist_ok=True)

    taxon_indices = [m["taxon_idx"] for m in step_meta]
    taxon_labels  = [names[i] for i in taxon_indices]
    x = np.arange(len(taxon_indices))

    final_elbos   = [m["train_summary"]["final_scaled_elbo"] for m in step_meta]
    final_entropy = [m["train_summary"]["final_entropy"]      for m in step_meta]
    final_max_qe  = [m["train_summary"]["final_max_qe"]       for m in step_meta]
    final_kl_b    = [m["train_summary"]["final_kl_b"]         for m in step_meta]
    final_kl_r    = [m["train_summary"]["final_kl_r"]         for m in step_meta]
    final_loglik  = [m["train_summary"]["final_loglik"]        for m in step_meta]
    n_edges_list  = [m["train_summary"]["n_edges"]             for m in step_meta]

    # --- Plot 1: ELBO at convergence per taxon ---
    plt.figure(figsize=(max(8, len(taxon_indices) * 0.5), 4))
    plt.plot(x, final_elbos, marker='o', color='steelblue', linewidth=2)
    for xi, yi in zip(x, final_elbos):
        plt.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 6), ha='center', fontsize=7)
    plt.xticks(x, taxon_labels, rotation=45, ha='right', fontsize=8)
    plt.xlabel("Taxon (insertion order)")
    plt.ylabel("Final ELBO / n_sites")
    plt.title("ELBO at Convergence per Taxon\n(drop = harder insertion, more edges, weaker signal)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_elbo.png"), dpi=150)
    plt.close()

    # --- Plot 2: Final entropy and max_qe per taxon (confidence vs uncertainty) ---
    fig, ax1 = plt.subplots(figsize=(max(8, len(taxon_indices) * 0.5), 4))
    ax1.plot(x, final_entropy, marker='o', color='green', linewidth=2, label='Final H(q) entropy')
    ax1.set_ylabel("Shannon Entropy", color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    ax2.plot(x, final_max_qe, marker='s', color='purple', linewidth=2,
             linestyle='--', label='Final max q(e)')
    ax2.set_ylabel("Max q(e) — confidence in top edge", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(0, 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(taxon_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel("Taxon (insertion order)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    plt.title("Posterior Confidence at Convergence per Taxon\n(low entropy + high max_qe = confident insertion)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_confidence.png"), dpi=150)
    plt.close()

    # --- Plot 3: KL and loglik decomposition per taxon ---
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(taxon_indices) * 1.2), 4))

    axes[0].bar(x, final_loglik, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(taxon_labels, rotation=45, ha='right', fontsize=7)
    axes[0].set_title("Final E[log p(data)] per Taxon")
    axes[0].set_ylabel("Nats")
    axes[0].grid(alpha=0.3)

    axes[1].bar(x, final_kl_b, color='orange', alpha=0.8, label='KL_b (branch)')
    axes[1].bar(x, final_kl_r, color='red',    alpha=0.8, label='KL_r (rho)',
                bottom=final_kl_b)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(taxon_labels, rotation=45, ha='right', fontsize=7)
    axes[1].set_title("Final KL Terms per Taxon (stacked)")
    axes[1].set_ylabel("KL (nats)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, n_edges_list, marker='o', color='gray', linewidth=2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(taxon_labels, rotation=45, ha='right', fontsize=7)
    axes[2].set_title("Number of Edges at Each Insertion Step\n(= search space size)")
    axes[2].set_ylabel("n_edges (= 2k-3 for k taxa so far)")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Cross-Taxon Training Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_decomposition.png"), dpi=150)
    plt.close()

    print(f"\n[Category 3 plots saved to {run_dir}]")


def run_training_pass(
    args: argparse.Namespace,
    names: List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    device: torch.device,
    run_dir: str,
):
    n_use = len(names)
    insert_start = 3
    insert_end = n_use - 1

    step_models: Dict[int, str] = {}
    step_nets: Dict[int, CNNPosteriorNet] = {}
    step_meta = []

    for taxon_idx in range(insert_start, insert_end + 1):
        print(f"\n=== Train Step for taxon idx={taxon_idx} name={names[taxon_idx]} ===")
        step_dir = os.path.join(run_dir, f"train_step_{taxon_idx:03d}_{names[taxon_idx]}")
        os.makedirs(step_dir, exist_ok=True)

        net, train_summary = train_single_insertion_step(
            taxon_idx=taxon_idx, step_nets=step_nets, new_leaf_lik=leaf_lik_all[taxon_idx],
            leaf_lik_all=leaf_lik_all, taxon_names=names, device=device, args=args, step_out_dir=step_dir,
        )

        model_path = os.path.join(step_dir, "cnn_posterior_net.pt")

        step_models[taxon_idx] = model_path
        
        net.eval()
        step_nets[taxon_idx] = net

        step_meta.append({
            "taxon_idx": taxon_idx, "taxon_name": names[taxon_idx],
            "model_path": model_path, "train_summary": train_summary
        })

    trained_tree = sample_tree_from_trained_steps(args, names, leaf_lik_all, step_nets, device)

    # Category 3: cross-taxon summary plots
    save_crosstep_plots(run_dir, step_meta, names)

    return trained_tree, step_models, step_meta


def preload_step_nets(step_models: Dict[int, str], hidden_dim: int, device: torch.device) -> Dict[int, CNNPosteriorNet]:
    step_nets: Dict[int, CNNPosteriorNet] = {}
    for taxon_idx, model_path in step_models.items():
        net = CNNPosteriorNet(in_channels=EDGE_FEATURE_DIM, hidden=hidden_dim).to(device)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        net.eval()
        step_nets[taxon_idx] = net
    return step_nets


def sample_tree_from_trained_steps(
    args: argparse.Namespace,
    names: List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    step_nets: Dict[int, CNNPosteriorNet],
    device: torch.device,
) -> UnrootedTree:
    n_use = len(names)
    allocator = InternalNodeAllocator(start=n_use)
    center_id = allocator.alloc()
    tree = build_t3_star(args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=center_id)

    with torch.no_grad():
        for taxon_idx in range(3, n_use):
            net = step_nets[taxon_idx]
            edges, q_e, _, mu_b, log_sigma_b, mu_r, log_sigma_r, _, _, _, _, _ = evaluate_edges(
                tree, leaf_lik_all[taxon_idx], leaf_lik_all, net, device, k_samples=args.k_eval, jc_rate=args.jc_rate, temperature=args.sample_temp
            )
            chosen_edge, chosen_rho, chosen_b = choose_action(
                edges, q_e, mu_b, log_sigma_b, mu_r, log_sigma_r, sample_continuous=True
            )
            insert_taxon_on_edge(tree, chosen_edge, taxon_idx, chosen_rho, chosen_b, allocator)

    return tree


def save_training_tree_json(tree: UnrootedTree, names: List[str], path: str) -> None:
    rec = []
    for u, v in tree.edges():
        rec.append({"u_id": int(u), "v_id": int(v), "u_label": node_label(u, names), "v_label": node_label(v, names), "length": float(tree.length(u, v))})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exact Marginalization Trajectory Rollout")
    p.add_argument("--fasta", type=str, required=True, help="Input FASTA path")
    p.add_argument("--n_taxa_use", type=int, default=None, help="Use first N taxa")
    p.add_argument("--max_steps", type=int, default=1000, help="Safety limit for iterations per taxon")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--k_train", type=int, default=10, help="MC samples for training ELBO")
    p.add_argument("--k_eval", type=int, default=100, help="MC samples for final tree generation")
    p.add_argument("--n_tree_samples", type=int, default=1000, help="Final posterior trees to sample")
    p.add_argument("--k_mll", type=int, default=1000,
                   help="Importance samples for MLL estimation (1000 matches PhyloGFN/ARTree)")
    p.add_argument("--temp_start", type=float, default=50.0, help="Starting temperature for annealing (exploration)")
    p.add_argument("--temp_end", type=float, default=1.0, help="Ending temperature for annealing (exploitation)")
    p.add_argument("--sample_temp", type=float, default=2.0,
               help="Temperature for edge sampling (>1 = more diverse topologies)")
    
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jc_rate", type=float, default=1.0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--out_root", type=str, default="results_converge_elbo")
    return p


def save_sampled_tree_plots(
    sampled_trees: List[UnrootedTree],
    names: List[str],
    run_dir: str,
    leaf_lik_all: Dict[int, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    step_nets: Dict[int, "CNNPosteriorNet"],
) -> None:
    """Category 4: sampled tree diagnostics. Category 5: numerical health."""
    os.makedirs(run_dir, exist_ok=True)
    n_taxa = len(names)
    expected_edges = 2 * n_taxa - 3

    # ── Category 4 Plot 1: topology frequency ────────────────────────────────
    from collections import Counter
    newick_strings = [to_newick(t, names) for t in sampled_trees]
    topology_counts = Counter(newick_strings)
    n_unique = len(topology_counts)
    top_topologies = topology_counts.most_common(min(20, n_unique))
    top_freqs = [c / len(sampled_trees) for _, c in top_topologies]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top_freqs)), top_freqs, color='steelblue', alpha=0.8)
    plt.xlabel("Topology rank (most common = 0)")
    plt.ylabel("Posterior probability estimate")
    plt.title(
        f"Top-{len(top_freqs)} Topology Frequencies\n"
        f"{n_unique} unique topologies from {len(sampled_trees)} samples  |  "
        f"Top freq: {top_freqs[0]:.3f}"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "topology_frequencies.png"), dpi=150)
    plt.close()

    # Print top 5 to console for quick inspection
    print(f"\n[Cat 4] Topology diversity: {n_unique} unique / {len(sampled_trees)} samples")
    for rank, (nwk, cnt) in enumerate(top_topologies[:5]):
        print(f"  Rank {rank+1}: freq={cnt/len(sampled_trees):.4f}  {nwk[:80]}...")

    # ── Category 4 Plot 2: branch length distribution across all sampled trees ─
    all_branch_lengths = []
    for t in sampled_trees:
        for u, v in t.edges():
            all_branch_lengths.append(t.length(u, v))
    all_branch_lengths = np.array(all_branch_lengths)

    plt.figure(figsize=(8, 4))
    plt.hist(all_branch_lengths, bins=60, density=True,
             color='steelblue', alpha=0.75, label='All branches')
    med = np.median(all_branch_lengths)
    mn  = np.mean(all_branch_lengths)
    plt.axvline(x=med, color='red',    linestyle='--', linewidth=1.5,
                label=f'Median: {med:.4f}')
    plt.axvline(x=mn,  color='orange', linestyle='--', linewidth=1.5,
                label=f'Mean:   {mn:.4f}')
    plt.axvline(x=1.0 / PRIOR_LAM_B, color='black', linestyle=':',
                linewidth=1.5, label=f'Prior mean: {1.0/PRIOR_LAM_B:.3f}')
    plt.xlabel("Branch length (substitutions/site)")
    plt.ylabel("Density")
    plt.title("Distribution of Branch Lengths Across All Sampled Trees")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "sampled_branch_lengths.png"), dpi=150)
    plt.close()

    # ── Category 4 Plot 3: edge count sanity check ───────────────────────────
    edge_counts = [len(t.edges()) for t in sampled_trees]
    count_counter = Counter(edge_counts)
    bad = sum(1 for c in edge_counts if c != expected_edges)

    plt.figure(figsize=(6, 3))
    plt.bar([str(k) for k in sorted(count_counter)],
            [count_counter[k] for k in sorted(count_counter)],
            color=['red' if k != expected_edges else 'steelblue'
                   for k in sorted(count_counter)])
    plt.axvline(x=str(expected_edges), color='green', linestyle='--',
                linewidth=2, label=f'Expected: {expected_edges}')
    plt.xlabel("Number of edges in sampled tree")
    plt.ylabel("Count of trees")
    plt.title(
        f"Edge Count Check (all should be {expected_edges} = 2×{n_taxa}-3)\n"
        f"{'✓ All correct' if bad == 0 else f'⚠ {bad} trees have wrong edge count!'}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "edge_count_check.png"), dpi=150)
    plt.close()

    if bad > 0:
        print(f"  ⚠ WARNING: {bad} sampled trees have wrong edge count (expected {expected_edges})")
    else:
        print(f"  ✓ All {len(sampled_trees)} sampled trees have correct edge count ({expected_edges})")

    # ── Category 5 Plot 1: log-scale health per edge on the final trained tree ─
    # Re-run cavity computation on the final sample of the last taxon's step
    # using the last step_net, to get a representative cav_cache
    last_taxon_idx = max(step_nets.keys())
    net_last = step_nets[last_taxon_idx]

    # Build a fresh tree using the trained nets (deterministic, using mode)
    allocator_tmp = InternalNodeAllocator(start=n_taxa)
    center_tmp = allocator_tmp.alloc()
    tree_tmp = build_t3_star(
        args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=center_tmp
    )
    with torch.no_grad():
        for tidx in range(3, n_taxa):
            net_t = step_nets[tidx]
            edges_t, q_t, _, mu_b_t, lsb_t, mu_r_t, lsr_t, _, _, _, _, _ = evaluate_edges(
                tree_tmp, leaf_lik_all[tidx], leaf_lik_all, net_t,
                device, k_samples=10, jc_rate=args.jc_rate, temperature=1.0
            )
            e_chosen, rho_chosen, b_chosen = choose_action(
                edges_t, q_t, mu_b_t, lsb_t, mu_r_t, lsr_t, sample_continuous=False
            )
            insert_taxon_on_edge(tree_tmp, e_chosen, tidx, rho_chosen, b_chosen, allocator_tmp)

    # Now get cav_cache for this representative tree
    leaf_lik_current = {
        k: v for k, v in leaf_lik_all.items()
        if k in tree_tmp.adj and len(tree_tmp.adj[k]) == 1
    }
    _, cav_cache_final = compute_messages_and_cavities(tree_tmp, leaf_lik_current)

    scale_means, scale_mins, scale_maxs = [], [], []
    has_nan, has_neginf = False, False
    for key, (_, log_scale) in cav_cache_final.items():
        s = log_scale.detach().cpu()
        scale_means.append(s.mean().item())
        scale_mins.append(s.min().item())
        scale_maxs.append(s.max().item())
        if torch.isnan(s).any():
            has_nan = True
        if torch.isinf(s).any():
            has_neginf = True

    plt.figure(figsize=(8, 4))
    plt.hist(scale_means, bins=30, color='teal', alpha=0.75,
             label='Mean log_scale per cavity')
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5,
                label='0 (unnormalized = bad)')
    plt.xlabel("Mean log_scale value")
    plt.ylabel("Count of cavity directions")
    status = []
    if has_nan:    status.append("⚠ NaN detected!")
    if has_neginf: status.append("⚠ -Inf detected!")
    if not status: status.append("✓ No NaN / -Inf")
    plt.title(
        "Cat 5: Log-Scale Health (cavity normalization check)\n"
        + "  |  ".join(status)
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "log_scale_health.png"), dpi=150)
    plt.close()

    print(f"  Log-scale health: min={min(scale_mins):.2f}  "
          f"mean={np.mean(scale_means):.2f}  max={max(scale_maxs):.2f}  "
          + ("⚠ NaN!" if has_nan else "") + ("⚠ -Inf!" if has_neginf else ""))

    # ── Category 5 Plot 2: per-site log-likelihood distribution ──────────────
    # Shows whether any sites are producing -inf likelihoods (numerical collapse)
    all_site_logliks = []
    pi_dna = PI_DNA.to(device)
    edges_final = tree_tmp.edges()

    with torch.no_grad():
        for u, v in edges_final[:min(5, len(edges_final))]:  # sample 5 edges max
            psi_u, ls_u = cav_cache_final[(u, v)]
            psi_v, ls_v = cav_cache_final[(v, u)]
            t_uv = torch.tensor(tree_tmp.length(u, v), dtype=torch.float32)
            # Use prior mean branch length for Exp(lambda) = 1/lambda
            rho_s = torch.tensor([0.5], dtype=torch.float32)
            b_s   = torch.tensor([1.0 / PRIOR_LAM_B], dtype=torch.float32)
            ll = insertion_loglik_from_edge_batched(
                tree_tmp, (u, v), rho_s, b_s,
                leaf_lik_all[last_taxon_idx].to(device),
                cav_cache_final, pi_dna, rate=args.jc_rate
            )
            all_site_logliks.append(ll.item())

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(all_site_logliks)), all_site_logliks, color='teal', alpha=0.8)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Edge index (first 5 edges sampled)")
    plt.ylabel("Total log-likelihood (sum over sites)")
    plt.title(
        "Cat 5: Per-Edge Total Log-Likelihood Check\n"
        "(very large negative = underflow; near 0 = uninformative)"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "site_loglik_check.png"), dpi=150)
    plt.close()

    print(f"[Category 4 & 5 plots saved to {run_dir}]")

    
def sample_tree_with_log_q(
    args: argparse.Namespace,
    names: List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    step_nets: Dict[int, "CNNPosteriorNet"],
    device: torch.device,
) -> Tuple["UnrootedTree", float, float, float]:
    """
    Sample ONE complete tree from the trained posterior and compute all
    terms needed for marginal log-likelihood (MLL) importance weighting.

    Returns
    -------
    tree        : the sampled UnrootedTree with all N taxa
    log_q       : log q(τ, b, ρ) — sum of log-probs of every insertion decision
    log_p_Y     : log p(Y | τ, b) — full Felsenstein likelihood on the complete tree
    log_p_b     : log p(b) under Exp(λ) prior, summed over all 2N-3 edges

    MLL importance weight for this sample = log_p_Y + log_p_b - log_q
    (topology prior p(τ) is uniform and cancels in IS estimator)
    """
    n_use = len(names)
    allocator = InternalNodeAllocator(start=n_use)
    center_id  = allocator.alloc()
    tree = build_t3_star(args.edge_len_0, args.edge_len_1, args.edge_len_2,
                         center_idx=center_id)

    log_q   = 0.0
    log_p_Y = None   # set at the final insertion step

    with torch.no_grad():
        for taxon_idx in range(3, n_use):
            net = step_nets[taxon_idx]
            is_last = (taxon_idx == n_use - 1)

            # ── build edge features + get variational params ──────────────
            leaf_lik_current = {
                k: v for k, v in leaf_lik_all.items()
                if k in tree.adj and len(tree.adj[k]) == 1
            }
            _, cav_cache = compute_messages_and_cavities(tree, leaf_lik_current)

            edges      = tree.edges()
            n_nodes    = len(tree.adj)
            new_lik    = leaf_lik_all[taxon_idx].to(device)
            edge_feats = build_edge_features(
                edges=edges, cav_cache=cav_cache,
                new_leaf_lik=new_lik, device=device, n_nodes=n_nodes
            )
            mu_b, log_sigma_b, mu_r, log_sigma_r = net(edge_feats)

            # ── compute ELBO per edge at temperature=1 for proper IS ─────
            pi_dna = PI_DNA.to(device)
            loglik_mc, _, _ = mc_expected_loglik_and_samples(
                tree=tree, edges=edges,
                mu_b=mu_b, log_sigma_b=log_sigma_b,
                mu_r=mu_r, log_sigma_r=log_sigma_r,
                new_leaf_lik=new_lik, cav_cache=cav_cache,
                pi=pi_dna, k_samples=args.k_eval, jc_rate=args.jc_rate,
            )
            kl_b_e = exponential_kl_branch(mu_b, log_sigma_b)
            kl_r_e = gaussian_kl_standard(mu_r, log_sigma_r)
            elbo_e = loglik_mc - kl_b_e - kl_r_e
            # temperature=1 for MLL (no annealing distortion)
            q_e = torch.softmax(elbo_e, dim=0)

            # ── sample action (edge + continuous params) ─────────────────
            q_e_cpu = q_e.detach().cpu()
            q_e_cpu = torch.clamp(q_e_cpu, min=0.0)
            q_e_cpu = q_e_cpu / (q_e_cpu.sum() + EPS)

            chosen_idx = torch.multinomial(q_e_cpu, 1).item()

            mu_b_i  = mu_b[chosen_idx].item()
            lsb_i   = log_sigma_b[chosen_idx].item()
            mu_r_i  = mu_r[chosen_idx].item()
            lsr_i   = log_sigma_r[chosen_idx].item()

            sigma_b_i = math.exp(lsb_i)
            z_b       = random.gauss(mu_b_i, sigma_b_i)
            chosen_b  = math.exp(z_b)

            sigma_r_i  = math.exp(lsr_i)
            z_r        = random.gauss(mu_r_i, sigma_r_i)
            chosen_rho = 1.0 / (1.0 + math.exp(-z_r))   # sigmoid

            # ── accumulate log q(e_k, b_k, rho_k) ────────────────────────
            log_q_edge = math.log(float(q_e_cpu[chosen_idx]) + 1e-30)
            log_q_b    = log_lognormal_density(chosen_b, mu_b_i, lsb_i)
            log_q_rho  = log_logistic_normal_density(chosen_rho, mu_r_i, lsr_i)
            log_q      += log_q_edge + log_q_b + log_q_rho

            # ── at the LAST step: compute exact log p(Y | τ, b) ──────────
            # insertion_loglik_from_edge_batched returns sum over ALL sites
            # with log_scale_u + log_scale_v from cavity — this IS the full
            # Felsenstein log-likelihood of the complete N-taxon tree.
            if is_last:
                rho_t = torch.tensor([chosen_rho], dtype=torch.float32, device=device)
                b_t   = torch.tensor([chosen_b],   dtype=torch.float32, device=device)
                log_p_Y_tensor = insertion_loglik_from_edge_batched(
                    tree=tree,
                    edge=edges[chosen_idx],
                    rho=rho_t,
                    b_len=b_t,
                    new_leaf_lik=new_lik,
                    cav_cache=cav_cache,
                    pi=pi_dna,
                    rate=args.jc_rate,
                )
                log_p_Y = log_p_Y_tensor.item()

            insert_taxon_on_edge(
                tree, edges[chosen_idx], taxon_idx, chosen_rho, chosen_b, allocator
            )

    # ── log p(b) under Exp(λ) prior, all 2N-3 edges ──────────────────────────
    # Includes the 3 initial star edges (fixed by JC MLE, not variational).
    lam    = PRIOR_LAM_B
    log_p_b = sum(
        math.log(lam) - lam * tree.length(u, v)
        for u, v in tree.edges()
    )

    return tree, log_q, log_p_Y, log_p_b


def estimate_mll(
    args: argparse.Namespace,
    names: List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    step_nets: Dict[int, "CNNPosteriorNet"],
    device: torch.device,
    k_mll: int = 1000,
) -> Dict[str, float]:
    """
    Estimate the marginal log-likelihood via self-normalized importance sampling:

        MLL ≈ log [ (1/K) Σ_i  p(Y|τ_i,b_i) · p(b_i) / q(τ_i,b_i,ρ_i) ]
            = log_mean_exp( log_p_Y_i + log_p_b_i - log_q_i )

    This is an unbiased estimator of p(Y) before the log, so it is a lower
    bound on log p(Y) that tightens as K → ∞.

    Note: the uniform topology prior p(τ) is a constant (1 / (2N-5)!!) that
    is the same for all methods and is excluded here, as is standard in the
    literature (PhyloGFN / ARTree Table 1 numbers exclude it too).

    Note: the 3 initial star branch lengths are fixed by JC MLE and are not
    part of the variational posterior. Their prior contribution IS included
    in log_p_b. This makes the estimator conservative (lower) relative to
    methods that also place a variational distribution over all branch lengths.
    """
    print(f"\n  Estimating MLL with K={k_mll} importance samples...")
    log_weights = []

    for i in tqdm(range(k_mll), desc="MLL samples", leave=False, dynamic_ncols=True):
        _, log_q, log_p_Y, log_p_b = sample_tree_with_log_q(
            args, names, leaf_lik_all, step_nets, device
        )
        if log_p_Y is None:
            continue    # shouldn't happen but guard against edge case
        log_w = log_p_Y + log_p_b - log_q
        log_weights.append(log_w)

    log_weights = np.array(log_weights)

    # Numerically stable log-mean-exp
    max_lw    = log_weights.max()
    mll       = float(max_lw + math.log(np.exp(log_weights - max_lw).mean()))
    mll_std   = float(np.std(log_weights))       # std of raw weights (for diagnostics)

    # Effective sample size: ESS = (Σ w_i)² / Σ w_i²
    # In log space: log ESS = 2·log(Σ exp(lw_i)) - log(Σ exp(2·lw_i))
    log_sum_w  = max_lw + math.log(np.exp(log_weights - max_lw).sum())
    log_sum_w2 = np.log(np.exp(2.0 * (log_weights - max_lw)).sum()) + 2.0 * max_lw
    ess        = float(math.exp(2.0 * log_sum_w - log_sum_w2))
    ess_pct    = 100.0 * ess / k_mll

    print(f"\n{'='*55}")
    print(f"  MARGINAL LOG-LIKELIHOOD (MLL) ESTIMATE")
    print(f"{'='*55}")
    print(f"  MLL        = {mll:.4f}  nats")
    print(f"  K          = {k_mll}")
    print(f"  ESS        = {ess:.1f} / {k_mll}  ({ess_pct:.1f}%)")
    print(f"  log-weight mean  = {log_weights.mean():.4f}")
    print(f"  log-weight std   = {mll_std:.4f}")
    print(f"  log-weight range = [{log_weights.min():.2f}, {log_weights.max():.2f}]")
    print(f"{'='*55}")
    print(f"  Interpretation:")
    print(f"    PhyloGFN DS1 reference: -7108.4")
    print(f"    MrBayes DS1 reference:  ~-7108.3")
    print(f"  (gap from reference = {mll - (-7108.4):+.2f} nats)")
    print(f"{'='*55}\n")

    return {
        "mll":              mll,
        "k_mll":            k_mll,
        "ess":              ess,
        "ess_pct":          ess_pct,
        "log_weight_mean":  float(log_weights.mean()),
        "log_weight_std":   mll_std,
        "log_weight_min":   float(log_weights.min()),
        "log_weight_max":   float(log_weights.max()),
        "n_valid_samples":  len(log_weights),
        "note_topology_prior": "uniform p(tau) excluded (same convention as PhyloGFN/ARTree)",
        "note_initial_branches": "3 initial star branches are fixed (JC MLE), not variational",
    }


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    names_all, seqs_all = parse_fasta_in_order(args.fasta)
    n_total = len(names_all)
    n_use = n_total if args.n_taxa_use is None else min(args.n_taxa_use, n_total)
    if n_use < 4:
        raise ValueError("Need at least 4 taxa.")

    names = names_all[:n_use]
    seqs = seqs_all[:n_use]
    leaf_lik_all = {i: seq_to_likelihood_matrix(seqs[i]) for i in range(n_use)}

    e0, e1, e2 = compute_t3_branch_lengths(seqs[0], seqs[1], seqs[2])
    args.edge_len_0 = e0
    args.edge_len_1 = e1
    args.edge_len_2 = e2
    print(f"Calculated initial T3 branch lengths: {e0:.4f}, {e1:.4f}, {e2:.4f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_root, f"run_{stamp}_ntaxa{n_use}")
    os.makedirs(run_dir, exist_ok=False)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Output dir: {os.path.abspath(run_dir)}")
    print(f"Using device: {device}")
    print(f"Taxa used: {n_use}/{n_total}")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    with open(os.path.join(run_dir, "taxa_used.txt"), "w", encoding="utf-8") as f:
        for i, nm in enumerate(names):
            f.write(f"{i}\t{nm}\n")

    trained_tree, step_models, step_meta = run_training_pass(args, names, leaf_lik_all, device, run_dir)

    save_training_tree_json(trained_tree, names, os.path.join(run_dir, "trained_tree_edges.json"))
    save_nexus_trees([trained_tree], names, os.path.join(run_dir, "trained_tree.nex"))

    with open(os.path.join(run_dir, "step_meta.json"), "w", encoding="utf-8") as f:
        json.dump(step_meta, f, indent=2)

    step_nets = preload_step_nets(step_models, args.hidden_dim, device)
    set_seed(args.seed + 1000) 
    sampled_trees = []
    print(f"\nSampling {args.n_tree_samples} trees from trained posterior...")
    sample_iter = tqdm(range(args.n_tree_samples), desc="Sample trees", leave=True, dynamic_ncols=True)
    for i in sample_iter:
        tree_i = sample_tree_from_trained_steps(args, names, leaf_lik_all, step_nets, device)
        sampled_trees.append(tree_i)

    save_nexus_trees(sampled_trees, names, os.path.join(run_dir, "sampled_trees.nex"))

    # ── Marginal Log-Likelihood estimation via importance sampling ────────────
    print("\nEstimating Marginal Log-Likelihood (MLL)...")
    mll_results = estimate_mll(
        args=args,
        names=names,
        leaf_lik_all=leaf_lik_all,
        step_nets=step_nets,
        device=device,
        k_mll=args.k_mll,
    )
    with open(os.path.join(run_dir, "mll_results.json"), "w", encoding="utf-8") as f:
        json.dump(mll_results, f, indent=2)
    print(f"  MLL results saved to {run_dir}/mll_results.json")

    # Category 4 & 5: sampled tree diagnostics + numerical health
    save_sampled_tree_plots(
        sampled_trees=sampled_trees,
        names=names,
        run_dir=run_dir,
        leaf_lik_all=leaf_lik_all,
        device=device,
        args=args,
        step_nets=step_nets,
    )
    print("\nDone.")

if __name__ == "__main__":
    parser = build_args()
    main(parser.parse_args())