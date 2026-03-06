"""
Phylogenetic Variational Inference via Sequential Taxon Insertion
with Per-Edge Coordinate Ascent VI (CA-VI).

Key change from the previous version:
  q(rho, b | e) = q(rho | e)  ×  q(b | rho, e)
  
  Two separate network heads:
    RhoNet(cavities_e)        → (mu_r_e, log_sigma_r_e)
    BNet(cavities_e, rho_k)   → (mu_b_e^k, log_sigma_b_e^k)   per MC sample k

  Two-phase coordinate ascent per outer training step:
    Phase 1 – optimise RhoNet (BNet frozen, rho detached from BNet grad path)
    Phase 2 – fix rho, optimise BNet

  The full ELBO per edge is:
    ELBO_e = E_{rho}[E_b[log p(Y|tau,e,rho,b)]]
           - KL(q(rho) || p(rho))          ← closed form (logistic-normal vs N(0,1))
           - E_{rho}[KL(q(b|rho) || p(b))] ← MC estimate (varies per rho sample)
"""

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

EPS        = 1e-12
PI_DNA     = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
# 4 (anc_u) + 4 (anc_v) + 4 (DNA_new_leaf)
EDGE_FEATURE_DIM = 12
PRIOR_LAM_B      = 5.0   # Exp(10) prior on branch lengths → prior mean = 0.1 sub/site


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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
                curr_seq  = []
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
            raise ValueError(
                f"All sequences must have equal length. "
                f"Record {i} has length {len(s)} != {L}"
            )
    return names, seqs


def base_like(ch: str) -> List[float]:
    tbl = {
        "A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1],
        "U": [0,0,0,1], "R": [1,0,1,0], "Y": [0,1,0,1], "S": [0,1,1,0],
        "W": [1,0,0,1], "K": [0,0,1,1], "M": [1,1,0,0], "B": [0,1,1,1],
        "D": [1,0,1,1], "H": [1,1,0,1], "V": [1,1,1,0], "N": [1,1,1,1],
        "?": [1,1,1,1], "-": [1,1,1,1], ".": [1,1,1,1],
    }
    return [float(x) for x in tbl.get(ch, [1,1,1,1])]


def seq_to_likelihood_matrix(seq: str) -> torch.Tensor:
    return torch.tensor([base_like(ch) for ch in seq], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Tree Data Structure
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnrootedTree:
    adj:      Dict[int, List[int]]
    edge_len: Dict[frozenset, float]

    def neighbors(self, node: int) -> List[int]:
        return self.adj.get(node, [])

    def length(self, u: int, v: int) -> float:
        return self.edge_len[frozenset((u, v))]

    def edges(self) -> List[Tuple[int, int]]:
        out = []
        for key in self.edge_len:
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


def build_t3_star(
    edge_len_0: float, edge_len_1: float, edge_len_2: float, center_idx: int
) -> UnrootedTree:
    adj = {
        0: [center_idx], 1: [center_idx], 2: [center_idx],
        center_idx: [0, 1, 2],
    }
    edge_len = {
        frozenset((0, center_idx)): edge_len_0,
        frozenset((1, center_idx)): edge_len_1,
        frozenset((2, center_idx)): edge_len_2,
    }
    return UnrootedTree(adj=adj, edge_len=edge_len)


def compute_t3_branch_lengths(
    seq1: str, seq2: str, seq3: str
) -> Tuple[float, float, float]:
    """Exact 3-taxon JC branch lengths via the additive tree formula."""
    def p_dist(sA: str, sB: str) -> float:
        mm, cmp = 0, 0
        valid = set("ACGT")
        for a, b in zip(sA, sB):
            if a in valid and b in valid:
                cmp += 1
                if a != b:
                    mm += 1
        return mm / cmp if cmp > 0 else 0.75

    def jc_dist(p: float) -> float:
        p = min(p, 0.7499)
        return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)

    d12 = jc_dist(p_dist(seq1, seq2))
    d13 = jc_dist(p_dist(seq1, seq3))
    d23 = jc_dist(p_dist(seq2, seq3))

    e1 = (d12 + d13 - d23) / 2.0
    e2 = (d12 + d23 - d13) / 2.0
    e3 = (d13 + d23 - d12) / 2.0
    return max(1e-4, e1), max(1e-4, e2), max(1e-4, e3)


def insert_taxon_on_edge(
    tree:       UnrootedTree,
    edge:       Tuple[int, int],
    new_taxon:  int,
    rho:        float,
    b_len:      float,
    allocator:  InternalNodeAllocator,
) -> int:
    u, v   = edge
    t_uv   = tree.remove_edge(u, v)
    w      = allocator.alloc()
    tree.add_edge(u, w,         max(1e-8, rho        * t_uv))
    tree.add_edge(w, v,         max(1e-8, (1 - rho)  * t_uv))
    tree.add_edge(w, new_taxon, max(1e-8, b_len))
    return w


# ══════════════════════════════════════════════════════════════════════════════
# Felsenstein / JC Likelihood
# ══════════════════════════════════════════════════════════════════════════════

def jc_transition(t: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    e    = torch.exp(-4.0 * rate * t / 3.0)
    same = 0.25 + 0.75 * e
    diff = 0.25 - 0.25 * e
    p    = torch.ones((4, 4), dtype=t.dtype, device=t.device) * diff
    idx  = torch.arange(4, device=t.device)
    p[idx, idx] = same
    return p


def jc_transition_batched(t: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    """t: (K,)  →  (K, 4, 4) JC transition matrices."""
    e    = torch.exp(-4.0 * rate * t / 3.0)
    same = 0.25 + 0.75 * e
    diff = 0.25 - 0.25 * e
    p    = diff.view(-1, 1, 1).expand(-1, 4, 4).clone()
    idx  = torch.arange(4, device=t.device)
    p[:, idx, idx] = same.unsqueeze(1).expand(-1, 4)
    return p


def compute_messages_and_cavities(
    tree:     UnrootedTree,
    leaf_lik: Dict[int, torch.Tensor],
) -> Tuple[
    Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
    Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
]:
    any_leaf = next(iter(leaf_lik.values()))
    n_sites  = any_leaf.shape[0]
    device   = any_leaf.device

    msg_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    cav_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def cavity(a: int, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (a, b)
        if key in cav_cache:
            return cav_cache[key]
        psi           = leaf_lik[a].clone() if a in leaf_lik else \
                        torch.ones(n_sites, 4, dtype=torch.float32, device=device)
        acc_log_scale = torch.zeros(n_sites, dtype=torch.float32, device=device)
        for k in tree.neighbors(a):
            if k == b:
                continue
            m_k_a, m_ls = message(k, a)
            psi          = psi * m_k_a
            acc_log_scale = acc_log_scale + m_ls
            scaler       = psi.max(dim=-1, keepdim=True)[0].clamp_min(1e-30)
            psi          = psi / scaler
            acc_log_scale = acc_log_scale + torch.log(scaler).squeeze(-1)
        cav_cache[key] = (psi, acc_log_scale)
        return psi, acc_log_scale

    def message(a: int, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (a, b)
        if key in msg_cache:
            return msg_cache[key]
        psi, acc_log_scale = cavity(a, b)
        t_ab = torch.tensor(tree.length(a, b), dtype=torch.float32, device=device)
        p    = jc_transition(t_ab)
        m    = psi @ p.T
        scaler       = m.max(dim=-1, keepdim=True)[0].clamp_min(1e-30)
        m            = m / scaler
        acc_log_scale = acc_log_scale + torch.log(scaler).squeeze(-1)
        msg_cache[key] = (m, acc_log_scale)
        return m, acc_log_scale

    for u, v in tree.edges():
        _ = message(u, v)
        _ = message(v, u)

    return msg_cache, cav_cache


def insertion_loglik_from_edge_batched(
    tree:        UnrootedTree,
    edge:        Tuple[int, int],
    rho:         torch.Tensor,          # (K,)
    b_len:       torch.Tensor,          # (K,)
    new_leaf_lik: torch.Tensor,         # (L, 4)
    cav_cache:   Dict,
    pi:          torch.Tensor,
    rate:        float,
) -> torch.Tensor:                      # (K,) total log-likelihood per sample
    u, v   = edge
    t_uv   = torch.tensor(tree.length(u, v), dtype=torch.float32, device=rho.device)
    t_uw   = rho * t_uv
    t_wv   = (1.0 - rho) * t_uv

    psi_u, log_scale_u = cav_cache[(u, v)]
    psi_v, log_scale_v = cav_cache[(v, u)]

    psi_u        = psi_u.to(rho.device)
    psi_v        = psi_v.to(rho.device)
    log_scale_u  = log_scale_u.to(rho.device)
    log_scale_v  = log_scale_v.to(rho.device)
    new_leaf_lik = new_leaf_lik.to(rho.device)
    pi           = pi.to(rho.device)

    p_uw = jc_transition_batched(t_uw, rate=rate)   # (K,4,4)
    p_wv = jc_transition_batched(t_wv, rate=rate)
    p_yw = jc_transition_batched(b_len, rate=rate)

    m_u = torch.matmul(psi_u.unsqueeze(0), p_uw.transpose(1, 2))   # (K,L,4)
    m_v = torch.matmul(psi_v.unsqueeze(0), p_wv.transpose(1, 2))
    m_y = torch.matmul(new_leaf_lik.unsqueeze(0), p_yw.transpose(1, 2))

    root_partial = m_u * m_v * m_y
    site_like    = (root_partial * pi.view(1, 1, 4)).sum(dim=-1).clamp_min(1e-30)
    log_sl       = torch.log(site_like) + log_scale_u.unsqueeze(0) + log_scale_v.unsqueeze(0)
    return log_sl.sum(dim=-1)           # (K,)


# ══════════════════════════════════════════════════════════════════════════════
# Edge Features (Probabilistic Ancestral Sequences)
# ══════════════════════════════════════════════════════════════════════════════

def build_edge_features(
    edges, cav_cache, new_leaf_lik, device, n_nodes: int, tree, jc_rate: float
) -> torch.Tensor:
    new_leaf_lik = new_leaf_lik.to(device)
    pi_dna = PI_DNA.to(device)
    feats = []
    
    for u, v in edges:
        cav_uv, _ = cav_cache[(u, v)]
        cav_vu, _ = cav_cache[(v, u)]
        
        cav_uv = cav_uv.to(device)
        cav_vu = cav_vu.to(device)
        
        # 1. Get transition matrix for the specific branch length
        t_uv = torch.tensor(tree.length(u, v), dtype=torch.float32, device=device)
        P_uv = jc_transition(t_uv, rate=jc_rate)
        
        # 2. Pass messages across the branch to get full tree context
        msg_v_to_u = torch.matmul(cav_vu, P_uv.T)
        msg_u_to_v = torch.matmul(cav_uv, P_uv.T)
        
        # 3. Compute joint probabilities (Prior * Local Data * Cross-Branch Message)
        joint_u = pi_dna * cav_uv * msg_v_to_u
        joint_v = pi_dna * cav_vu * msg_u_to_v
        
        # 4. Normalize to get exact Probabilistic Ancestral Sequences (sum to 1.0)
        anc_u = joint_u / (joint_u.sum(dim=-1, keepdim=True) + EPS)
        anc_v = joint_v / (joint_v.sum(dim=-1, keepdim=True) + EPS)
        
        # 5. Normalize new leaf sequence
        y_n = new_leaf_lik / (new_leaf_lik.sum(dim=-1, keepdim=True) + EPS)
        
        feats.append(torch.cat([anc_u, anc_v, y_n], dim=-1))
        
    return torch.stack(feats, dim=0)   # (E, L, 12)


# ══════════════════════════════════════════════════════════════════════════════
# Neural Network Heads
# ══════════════════════════════════════════════════════════════════════════════

class RhoNet(nn.Module):
    """
    Sequence-to-rho-params network.
    
    Input : edge_feats  (E, L, EDGE_FEATURE_DIM)
    Output: mu_r        (E,)
            log_sigma_r (E,)
            h           (E, hidden)   ← pooled cavity embedding, reused by BNet
    """
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
        )
        self.rho_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),   # → (mu_r, log_sigma_r)
        )

    def get_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """
        Average pooling over all sites (allows CNN to naturally learn to ignore gaps).
        x : (E, L, C)  →  h : (E, hidden)
        """
        xt    = x.transpose(1, 2)     # (E, C, L)
        h_seq = self.cnn(xt)          # (E, hidden, L)
        h     = h_seq.mean(dim=2)     # (E, hidden)
        return h

    def forward(self, x: torch.Tensor):
        h   = self.get_pooled(x)
        out = self.rho_head(h)                          # (E, 2)
        mu_r        = out[:, 0]
        log_sigma_r = out[:, 1].clamp(-5.0, 3.0)
        return mu_r, log_sigma_r, h


class BNet(nn.Module):
    """
    Branch-length head conditioned on sampled rho.
    
    Input : h   (E, hidden)   pooled cavity features from RhoNet.get_pooled()
            rho (K, E)        sampled split-position values
    Output: mu_b        (K, E)
            log_sigma_b (K, E)
    
    Each MC sample of rho gets its own (mu_b, log_sigma_b), so the variational
    distribution for b truly depends on where on the edge the new taxon attaches.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.rho_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
        )
        self.b_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),   # → (mu_b, log_sigma_b)
        )

    def forward(self, h: torch.Tensor, rho: torch.Tensor):
        """
        h   : (E, hidden)
        rho : (K, E)
        """
        K, E    = rho.shape
        h_exp   = h.unsqueeze(0).expand(K, E, -1)           # (K, E, hidden)
        rho_emb = self.rho_embed(rho.unsqueeze(-1))          # (K, E, hidden)
        combined = torch.cat([h_exp, rho_emb], dim=-1)       # (K, E, 2*hidden)
        out      = self.b_head(combined)                      # (K, E, 2)
        mu_b        = out[..., 0] - 3.0
        log_sigma_b = out[..., 1].clamp(-5.0, 0.5)
        return mu_b, log_sigma_b


# ══════════════════════════════════════════════════════════════════════════════
# KL Divergences
# ══════════════════════════════════════════════════════════════════════════════

def exponential_kl_branch(
    mu: torch.Tensor, log_sigma: torch.Tensor, lam: float = PRIOR_LAM_B
) -> torch.Tensor:
    """
    KL( LogNormal(mu, sigma²) || Exponential(lam) )

    KL = -log_sigma - mu - ½(1+log2π) - log(lam) + lam·exp(mu + sigma²/2)

    Works for any shape of (mu, log_sigma); result has the same shape.
    """
    sigma2 = torch.exp(2.0 * log_sigma)
    e_b    = torch.exp(mu + 0.5 * sigma2)
    return (- log_sigma
            - mu
            - 0.5 * (1.0 + math.log(2.0 * math.pi))
            - math.log(lam)
            + lam * e_b)


def gaussian_kl_standard(
    mu: torch.Tensor, log_sigma: torch.Tensor
) -> torch.Tensor:
    """KL( N(mu, sigma²) || N(0, 1) )."""
    sigma2 = torch.exp(2.0 * log_sigma)
    return 0.5 * (mu.pow(2) + sigma2 - 1.0 - 2.0 * log_sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Log-density helpers for MLL importance weights
# ══════════════════════════════════════════════════════════════════════════════

def log_lognormal_density(b: float, mu: float, log_sigma: float) -> float:
    sigma = math.exp(log_sigma)
    log_b = math.log(max(b, 1e-30))
    return (- log_b
            - log_sigma
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((log_b - mu) / sigma) ** 2)


def log_logistic_normal_density(rho: float, mu_r: float, log_sigma_r: float) -> float:
    """Log density of the logit-normal distribution at rho ∈ (0,1)."""
    sigma_r    = math.exp(log_sigma_r)
    rho_c      = min(max(rho, 1e-6), 1.0 - 1e-6)
    z          = math.log(rho_c / (1.0 - rho_c))
    log_normal = (- log_sigma_r
                  - 0.5 * math.log(2.0 * math.pi)
                  - 0.5 * ((z - mu_r) / sigma_r) ** 2)
    log_jac    = - math.log(rho_c) - math.log(1.0 - rho_c)
    return log_normal + log_jac


# ══════════════════════════════════════════════════════════════════════════════
# Core CA-VI forward pass
# ══════════════════════════════════════════════════════════════════════════════


def _stack_cavities(
    edges:     List[Tuple[int, int]],
    tree:      "UnrootedTree",
    cav_cache: Dict,
    device:    torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack per-edge cavity tensors into batched form for vectorized loglik."""
    psi_u_l, psi_v_l, ls_u_l, ls_v_l, t_l = [], [], [], [], []
    for u, v in edges:
        psi_u, ls_u = cav_cache[(u, v)]
        psi_v, ls_v = cav_cache[(v, u)]
        psi_u_l.append(psi_u.to(device));  psi_v_l.append(psi_v.to(device))
        ls_u_l.append(ls_u.to(device));    ls_v_l.append(ls_v.to(device))
        t_l.append(tree.length(u, v))
    return (
        torch.stack(psi_u_l),
        torch.stack(psi_v_l),
        torch.stack(ls_u_l),
        torch.stack(ls_v_l),
        torch.tensor(t_l, dtype=torch.float32, device=device),
    )


def compute_loglik_all_edges(
    psi_u:        torch.Tensor,   # (E, L, 4)
    psi_v:        torch.Tensor,   # (E, L, 4)
    ls_u:         torch.Tensor,   # (E, L)
    ls_v:         torch.Tensor,   # (E, L)
    t_uv_all:     torch.Tensor,   # (E,)
    rho:          torch.Tensor,   # (K, E)
    b:            torch.Tensor,   # (K, E)
    new_leaf_lik: torch.Tensor,   # (L, 4)
    pi:           torch.Tensor,   # (4,)
    jc_rate:      float,
) -> torch.Tensor:                # (E,) mean log-lik over K samples
    """
    Fully vectorized Felsenstein likelihood for all E edges simultaneously.

    Replaces the old Python for-loop over edges (O(E) Python function calls)
    with a single batched einsum (one BLAS call). On CPU this is 10-40x faster
    at the late insertion steps where E is large (up to 49 edges for DS1).

    Memory: (K*E, L, 4) float32. For DS1 step 26: (980, 1300, 4) ≈ 20 MB.
    """
    K, E = rho.shape[0], rho.shape[1]
    t_uv = t_uv_all.unsqueeze(0)           # (1, E)
    t_uw = rho * t_uv                      # (K, E)
    t_wv = (1.0 - rho) * t_uv             # (K, E)

    def _jc_ke(t_ke: torch.Tensor) -> torch.Tensor:
        """Compute (K, E, 4, 4) JC matrices from (K, E) branch lengths."""
        p = jc_transition_batched(t_ke.reshape(-1), rate=jc_rate)  # (K*E, 4, 4)
        return p.reshape(K, E, 4, 4)

    P_uw = _jc_ke(t_uw)
    P_wv = _jc_ke(t_wv)
    P_yw = _jc_ke(b)

    # m[k,e,l,d] = Σ_c psi[e,l,c] * P[k,e,c,d]
    m_u = torch.einsum("elc,kedc->keld", psi_u, P_uw)              # (K,E,L,4)
    m_v = torch.einsum("elc,kedc->keld", psi_v, P_wv)
    m_y = torch.einsum("lc,kedc->keld",  new_leaf_lik, P_yw)

    root_partial = m_u * m_v * m_y                                  # (K,E,L,4)
    site_like = (root_partial * pi.view(1, 1, 1, 4)).sum(-1).clamp_min(1e-30)  # (K,E,L)
    log_sl = torch.log(site_like) + ls_u.unsqueeze(0) + ls_v.unsqueeze(0)     # (K,E,L)
    
    # Log of Expectation over K samples (IWAE bound / Marginal Likelihood)
    sum_over_sites = log_sl.sum(-1) # (K, E)
    return torch.logsumexp(sum_over_sites, dim=0) - math.log(K)   # (E,)


def compute_loglik_per_edge(
    tree:         UnrootedTree,
    edges:        List[Tuple[int, int]],
    rho:          torch.Tensor,          # (K, E)
    b:            torch.Tensor,          # (K, E)
    new_leaf_lik: torch.Tensor,
    cav_cache:    Dict,
    pi:           torch.Tensor,
    jc_rate:      float,
) -> torch.Tensor:                       # (E,) mean log-lik over K samples
    loglik_list = []
    for i, e in enumerate(edges):
        ll_k = insertion_loglik_from_edge_batched(
            tree=tree, edge=e,
            rho=rho[:, i], b_len=b[:, i],
            new_leaf_lik=new_leaf_lik,
            cav_cache=cav_cache,
            pi=pi, rate=jc_rate,
        )  # (K,)
        loglik_list.append(ll_k.mean())
    return torch.stack(loglik_list)      # (E,)


def forward_ca(
    tree:         UnrootedTree,
    edges:        List[Tuple[int, int]],
    edge_feats:   torch.Tensor,          # (E, L, C)   – already detached
    rho_net:      RhoNet,
    b_net:        BNet,
    new_leaf_lik: torch.Tensor,
    cav_cache:    Dict,
    pi:           torch.Tensor,
    k_samples:    int,
    jc_rate:      float,
    device:       torch.device,
    phase:        int,                   # 1 = optimise RhoNet, 2 = optimise BNet
    rho_fixed:    Optional[torch.Tensor] = None,   # (K,E), only for phase 2
    h_fixed:      Optional[torch.Tensor] = None,   # (E,hidden), only for phase 2
    stacked_cavs: Optional[Tuple]        = None,   # pre-stacked cavities
):
    """
    Single forward pass under the CA-VI factorisation.

    Phase 1 – RhoNet active, BNet frozen:
        • rho sampled via reparameterisation (grad → RhoNet)
        • rho.detach() passed to BNet to break cross-term
        • h.detach() passed to BNet to prevent cavity CNN gradient leaking
        • only optimizer_rho.step() is called by the caller

    Phase 2 – BNet active, RhoNet frozen:
        • rho_fixed supplied (already sampled & detached from Phase 1)
        • BNet receives rho_fixed and h_fixed (both detached from RhoNet)
        • kl_rho excluded from loss (constant wrt BNet)
        • only optimizer_b.step() is called by the caller

    Returns dict with all components needed for loss and diagnostics.
    """
    n_edges = len(edges)

    if phase == 1:
        # ── Phase 1: sample rho, freeze b ─────────────────────────────────────
        mu_r, log_sigma_r, h = rho_net(edge_feats)                # grad → rho_net
        kl_rho = gaussian_kl_standard(mu_r, log_sigma_r)          # (E,)

        std_r  = torch.exp(log_sigma_r)
        eps_r  = torch.randn(k_samples, n_edges, device=device)
        z_r    = mu_r.unsqueeze(0) + std_r.unsqueeze(0) * eps_r   # (K,E)
        rho    = torch.sigmoid(z_r)                                # (K,E)

        # mu_b, log_sigma_b = b_net(h.detach(), rho.detach())       # (K,E)
        # Keep h detached (so BNet doesn't alter the CNN features), 
        # but KEEP rho attached so likelihood gradients flow through BNet back to RhoNet!
        mu_b, log_sigma_b = b_net(h.detach(), rho)

    else:
        # ── Phase 2: rho fixed, optimise b ────────────────────────────────────
        rho        = rho_fixed                                     # (K,E) detached
        h_for_b    = h_fixed                                       # (E,hidden) detached
        mu_b, log_sigma_b = b_net(h_for_b, rho)                   # (K,E), grad → b_net

        # Still compute rho stats for monitoring (no grad needed)
        with torch.no_grad():
            mu_r, log_sigma_r, _ = rho_net(edge_feats)
            kl_rho = gaussian_kl_standard(mu_r, log_sigma_r)
        h = h_fixed   # for return value

    # ── Sample b via reparameterisation ───────────────────────────────────────
    std_b  = torch.exp(log_sigma_b)
    eps_b  = torch.randn(k_samples, n_edges, device=device)
    z_b = mu_b + std_b * eps_b
    # Clamp z_b before exp() to prevent overflow, then clamp the bottom to 1e-8
    b   = torch.exp(z_b.clamp(max=math.log(2.0))).clamp(min=1e-8)                                 # (K,E)

    # ── KL for b – MC estimate (varies per rho sample) ───────────────────────
    kl_b_per_k = exponential_kl_branch(mu_b, log_sigma_b)        # (K,E)
    kl_b_mc    = kl_b_per_k.mean(dim=0)                          # (E,)

    # ── Felsenstein likelihood (vectorized over all E edges) ──────────────────
    if stacked_cavs is None:
        stacked_cavs = _stack_cavities(edges, tree, cav_cache, device)
    loglik_mc = compute_loglik_all_edges(
        *stacked_cavs, rho, b, new_leaf_lik, pi, jc_rate
    )  # (E,)

    # ── Per-edge ELBO ─────────────────────────────────────────────────────────
    if phase == 1:
        elbo_e = loglik_mc - kl_rho - kl_b_mc
    else:
        elbo_e = loglik_mc - kl_rho.detach() - kl_b_mc

    return {
        "elbo_e":       elbo_e,        # (E,) used for loss
        "loglik_mc":    loglik_mc,     # (E,)
        "kl_rho":       kl_rho,       # (E,)
        "kl_b_mc":      kl_b_mc,      # (E,)
        "rho":          rho,          # (K,E)
        "b":            b,            # (K,E)
        "mu_r":         mu_r,         # (E,)
        "log_sigma_r":  log_sigma_r,  # (E,)
        "mu_b":         mu_b,         # (K,E)
        "log_sigma_b":  log_sigma_b,  # (K,E)
        "h":            h,            # (E,hidden)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Edge evaluation (for sampling + context tree construction)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_edges_ca(
    tree:         UnrootedTree,
    new_leaf_lik: torch.Tensor,
    leaf_lik_all: Dict[int, torch.Tensor],
    rho_net:      RhoNet,
    b_net:        BNet,
    device:       torch.device,
    k_samples:    int,
    jc_rate:      float,
    temperature:  float = 1.0,
):
    """
    Full ELBO evaluation for edge scoring.
    Safe to call inside torch.no_grad() (for frozen prior steps).
    Returns all parameters needed for choose_action_ca and diagnostics.
    """
    leaf_lik_current = {
        k: v for k, v in leaf_lik_all.items()
        if k in tree.adj and len(tree.adj[k]) == 1
    }
    expected_leaves = [k for k in tree.adj if k < len(leaf_lik_all)]
    assert len(leaf_lik_current) == len(expected_leaves), (
        f"Tree topology drift: expected {len(expected_leaves)} leaves, "
        f"got {len(leaf_lik_current)}."
    )

    with torch.no_grad():
        _, cav_cache    = compute_messages_and_cavities(tree, leaf_lik_current)
        edges           = tree.edges()
        n_nodes         = len(tree.adj)
        edge_feats_raw  = build_edge_features(
            edges=edges, cav_cache=cav_cache,
            new_leaf_lik=new_leaf_lik, device=device, n_nodes=n_nodes,
            tree=tree, jc_rate=jc_rate,
        )

    edge_feats   = edge_feats_raw.detach()
    new_leaf_lik = new_leaf_lik.to(device)
    pi_dna       = PI_DNA.to(device)
    n_edges      = len(edges)

    # Sample rho
    mu_r, log_sigma_r, h = rho_net(edge_feats)
    kl_rho = gaussian_kl_standard(mu_r, log_sigma_r)

    std_r  = torch.exp(log_sigma_r)
    eps_r  = torch.randn(k_samples, n_edges, device=device)
    z_r    = mu_r.unsqueeze(0) + std_r.unsqueeze(0) * eps_r
    rho    = torch.sigmoid(z_r)           # (K,E)

    # b params conditioned on rho (always detached for clean eval)
    mu_b, log_sigma_b = b_net(h.detach(), rho.detach())   # (K,E)

    std_b  = torch.exp(log_sigma_b)
    eps_b  = torch.randn(k_samples, n_edges, device=device)
    z_b    = mu_b + std_b * eps_b
    b      = torch.exp(z_b.clamp(max=math.log(2.0))).clamp(min=1e-8)

    kl_b_per_k = exponential_kl_branch(mu_b, log_sigma_b)
    kl_b_mc    = kl_b_per_k.mean(0)

    _sc = _stack_cavities(edges, tree, cav_cache, device)
    loglik_mc = compute_loglik_all_edges(
        *_sc, rho, b, new_leaf_lik, pi_dna, jc_rate
    )

    elbo_e = loglik_mc - kl_rho - kl_b_mc
    
    # SCORE EDGES STRICTLY BY FELSENSTEIN LIKELIHOOD (SCALED BY N_SITES)
    n_sites = new_leaf_lik.shape[0]
    q_e     = torch.softmax((loglik_mc / n_sites) / temperature, dim=0)

    rho_np = rho.detach().cpu().numpy()
    b_np   = b.detach().cpu().numpy()

    return (
        edges, q_e, elbo_e,
        mu_r, log_sigma_r,
        mu_b, log_sigma_b,
        h,
        rho_np, b_np,
        kl_rho, kl_b_mc, loglik_mc,
        cav_cache,       # returned for MLL computation
        edge_feats,      # returned for direct net calls
    )



def choose_action_ca(
    edges:        List[Tuple[int, int]],
    q_e:          torch.Tensor,
    rho_samples:  np.ndarray,    # (K, E) from evaluate_edges_ca
    b_samples:    np.ndarray,    # (K, E) from evaluate_edges_ca
    mu_r:         torch.Tensor,  # (E,) needed for MAP
    h:            torch.Tensor,  # (E, hidden) needed for MAP
    b_net:        BNet,          # needed for MAP
    device:       torch.device,  # needed for MAP
    sample_continuous: bool = True,
) -> Tuple[Tuple[int, int], float, float]:
    """
    Sample (edge, rho, b) for one insertion step.
    """
    # 1. Sample the edge using the computed scores
    q_e = torch.nan_to_num(q_e, nan=0.0)
    q_e = q_e / (q_e.sum() + EPS)
    chosen_idx = torch.multinomial(q_e, 1).item()

    if sample_continuous:
        # 2. Grab the EXACT rho and b that were used to evaluate this edge.
        # We just take the first sample (index 0) from the K samples drawn.
        chosen_rho = float(rho_samples[0, chosen_idx])
        chosen_b   = float(b_samples[0, chosen_idx])
    else:
        # For deterministic MAP inference, we use the modes.
        chosen_rho = torch.sigmoid(mu_r[chosen_idx]).item()
        h_e        = h[chosen_idx].detach().unsqueeze(0)
        rho_in     = torch.tensor([[chosen_rho]], dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_b_i, _ = b_net(h_e, rho_in)
        chosen_b   = math.exp(min(mu_b_i.item(), 2.0))

    return edges[chosen_idx], chosen_rho, chosen_b

# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def node_label(node_id: int, taxon_names: List[str]) -> str:
    return taxon_names[node_id] if node_id < len(taxon_names) else f"i{node_id}"


def _ema(values: List[float], alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average."""
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def save_step_plots(
    out_dir:          str,
    edge_labels:      List[str],
    elbo_hist:        List[float],
    q_final:          np.ndarray,
    elbo_final:       np.ndarray,
    loglik_final:     np.ndarray,
    rho_samples_final: np.ndarray,   # (K, E)
    b_samples_final:  np.ndarray,    # (K, E)
    entropy_hist:     List[float],
    max_qe_hist:      List[float],
    kl_b_hist:        List[float],
    kl_r_hist:        List[float],
    loglik_hist:      List[float],
    grad_norm_rho_hist: List[float],
    grad_norm_b_hist:   List[float],
    temp_start:       float,
    temp_end:         float,
    max_steps:        int,
    min_steps:        int,    
    mu_r_top_hist:    List[float],
    sigma_r_top_hist: List[float],
    mu_b_top_hist:    List[float],   # mean of mu_b over K samples for top edge
    sigma_b_top_hist: List[float],
    mu_b_final_mean:  np.ndarray,    # (E,) mean mu_b for bar chart
    log_sigma_b_final_mean: np.ndarray,  # (E,)
    taxon_idx:        int,
    n_sites:          int,           # ADD THIS LINE
    # ── Early-stopping diagnostics (optional) ─────────────────────────
    ema_hist:         Optional[List[float]] = None,
    stopped_early:    bool = False,
    stop_step:        Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    steps = np.arange(1, len(elbo_hist) + 1)
    # Use the online EMA from training if provided, else recompute
    ema = np.array(ema_hist) if ema_hist is not None else _ema(elbo_hist)

    # ── ELBO convergence ─────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(steps, elbo_hist, alpha=0.35, label="ELBO (raw)")
    plt.plot(steps, ema,       linewidth=2, label=f"ELBO (EMA α={0.1:.2f})")
    if stopped_early and stop_step is not None:
        plt.axvline(x=stop_step, color="red", linestyle="--", linewidth=1.5,
                    label=f"Early stop (step {stop_step})")
    plt.xlabel("Outer iteration"); plt.ylabel("ELBO / n_sites")
    plt.title("Convergence Monitoring"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbo_convergence.png"), dpi=150)
    plt.close()

    # ── Edge posterior q(e) ───────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, q_final, color='mediumseagreen')
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge"); plt.ylabel("q(e) = softmax(Log-Likelihood / T)")
    plt.title("Edge Posterior (Pure Felsenstein Likelihood)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_posterior_qe.png"), dpi=150)
    plt.close()

    # ── Relative Log-Likelihood per edge (The Actual Scoring Metric) ──────────
    rel_loglik = loglik_final - np.max(loglik_final)
    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, rel_loglik, color='purple', alpha=0.8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge"); plt.ylabel("LogLik - max(LogLik)")
    plt.title("Relative Per-Edge Felsenstein Likelihood Scores"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_loglik_scores.png"), dpi=150)
    plt.close()
    
    # ── Relative ELBO per edge (For debugging the Neural Network) ─────────────
    rel_elbo = elbo_final - np.max(elbo_final)
    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, rel_elbo, color='steelblue', alpha=0.6)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge"); plt.ylabel("ELBO - max(ELBO)")
    plt.title("Relative Per-Edge ELBO (Neural Network Loss)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_elbo_scores.png"), dpi=150)
    plt.close()

    # ── Rho & b posteriors for top-5 edges ───────────────────────────────────
    top_k    = min(5, len(edge_labels))
    top_idxs = np.argsort(q_final)[-top_k:]

    for arr, fname, xlabel in [
        (rho_samples_final, "rho_posterior.png", "rho"),
        (b_samples_final,   "b_posterior.png",   "b"),
    ]:
        plt.figure(figsize=(8, 4))
        for i in top_idxs:
            plt.hist(arr[:, i], bins=30, alpha=0.30, density=True,
                     label=f"{xlabel}|{edge_labels[i]}")
        plt.xlabel(xlabel); plt.ylabel("density")
        plt.title(f"Posterior for {xlabel} (Top-{top_k} edges)")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    # ── ELBO decomposition ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Fix Panel 0: Divide by n_sites to match the ELBO scale
    axes[0].plot(steps, [x / n_sites for x in loglik_hist], color='blue', alpha=0.7)
    axes[0].set_title("E[log p(data)] / n_sites"); axes[0].set_ylabel("Nats / site"); axes[0].grid(alpha=0.3)
    
    # Fix Panel 1: Add the missing y-axis label
    axes[1].plot(steps, kl_b_hist, color='orange', label="KL_b (branch)")
    axes[1].plot(steps, kl_r_hist, color='red',    label="KL_r (rho)")
    axes[1].set_title("KL Terms"); axes[1].set_ylabel("KL (nats)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(steps, elbo_hist, color='green', alpha=0.7)
    axes[2].set_title("ELBO / n_sites"); axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("Outer iteration")
    plt.suptitle("ELBO Decomposition"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbo_decomposition.png"), dpi=150)
    plt.close()

    # ── Gradient norms ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hist, title in [
        (axes[0], grad_norm_rho_hist, "RhoNet grad norm (phase 1)"),
        (axes[1], grad_norm_b_hist,   "BNet grad norm (phase 2)"),
    ]:
        ax.plot(steps, hist, alpha=0.6, color='red', linewidth=1)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='clip=1.0')
        clip_frac = np.mean(np.array(hist) > 1.0) * 100
        ax.set_title(f"{title}\n({clip_frac:.1f}% clipped)")
        ax.set_xlabel("Outer iteration"); ax.set_yscale('log')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradient_norms.png"), dpi=150)
    plt.close()

    # ── Entropy & confidence ──────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(steps, entropy_hist, color='green', linewidth=2)
    plt.xlabel("Outer iteration"); plt.ylabel("Shannon Entropy H(q)")
    plt.title("Edge Distribution Entropy"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_trajectory.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, max_qe_hist, color='purple', linewidth=2)
    plt.xlabel("Outer iteration"); plt.ylabel("Max q(e)")
    plt.title("Confidence in Top Edge"); plt.ylim(0, 1.05)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_trajectory.png"), dpi=150)
    plt.close()

    # ── Entropy vs temperature ────────────────────────────────────────────────
    temps = [
            temp_start * (temp_end / temp_start) ** min(1.0, i / max(1, min_steps - 1))
            for i in range(len(steps))
        ]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, entropy_hist, color='green', linewidth=2, label='H(q)')
    ax1.set_xlabel("Outer iteration"); ax1.set_ylabel("Entropy", color='green')
    ax2 = ax1.twinx()
    ax2.plot(steps, temps, color='orange', linestyle='--', linewidth=1.5, label='Temp')
    ax2.set_ylabel("Temperature", color='orange')
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc='upper right')
    plt.title("Entropy vs Temperature"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_vs_temperature.png"), dpi=150)
    plt.close()

    # ── Variational parameter trajectories ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, mu_r_top_hist, color='darkorange')
    axes[0, 0].axhline(0.0, color='red', linestyle='--', label='Prior mean (0→ρ=0.5)')
    axes[0, 0].set_title("mu_r of top edge (logit space)")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, sigma_r_top_hist, color='darkorange')
    axes[0, 1].axhline(1.0, color='red', linestyle='--', label='Prior std')
    axes[0, 1].set_title("sigma_r of top edge"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(steps, mu_b_top_hist, color='steelblue')
    axes[1, 0].axhline(math.log(1.0 / PRIOR_LAM_B), color='red', linestyle='--',
                        label=f'Prior mode (log(1/λ)={math.log(1/PRIOR_LAM_B):.2f})')
    axes[1, 0].set_title("mu_b of top edge (log space, mean over K)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(steps, sigma_b_top_hist, color='steelblue')
    axes[1, 1].axhline(1.0, color='red', linestyle='--', label='Prior std')
    axes[1, 1].set_title("sigma_b of top edge (mean over K)"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Outer iteration")
    plt.suptitle(f"Variational Parameter Trajectories — Taxon {taxon_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variational_params_trajectory.png"), dpi=150)
    plt.close()

    # ── Implied branch lengths per edge at convergence ────────────────────────
    # FIX: Calculate the mean and std directly from the actual samples 
    # that were drawn and evaluated, instead of the network's latent parameters.
    b_means = np.mean(b_samples_final, axis=0)
    b_stds  = np.std(b_samples_final, axis=0)
    
    x       = np.arange(len(edge_labels))
    plt.figure(figsize=(max(10, len(edge_labels) * 0.4), 5))
    plt.bar(x, b_means, yerr=b_stds, capsize=3, color='steelblue', alpha=0.7, label='mean±std')
    plt.axhline(0.05, color='green',  linestyle='--', label='Short (0.05)')
    plt.axhline(0.3,  color='orange', linestyle='--', label='Long (0.3)')
    plt.axhline(1.0 / PRIOR_LAM_B, color='red', linestyle=':',
                label=f'Prior mean (1/λ={1/PRIOR_LAM_B:.3f})')
    plt.xticks(x, edge_labels, rotation=45, ha='right', fontsize=7)
    plt.ylabel("Branch length (sub/site)")
    plt.title(f"Inferred Branch Lengths — Taxon {taxon_idx}")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "branch_lengths_per_edge.png"), dpi=150)
    plt.close()


def save_crosstep_plots(
    run_dir: str, step_meta: List[dict], names: List[str]
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    tidxs  = [m["taxon_idx"]  for m in step_meta]
    tlabels = [names[i]       for m in step_meta for i in [m["taxon_idx"]]]
    x      = np.arange(len(tidxs))
    ts     = step_meta

    def _get(key):
        return [m["train_summary"][key] for m in ts]

    final_elbos  = _get("final_scaled_elbo")
    final_ent    = _get("final_entropy")
    final_maxqe  = _get("final_max_qe")
    final_kl_b   = _get("final_kl_b")
    final_kl_r   = _get("final_kl_r")
    final_loglik  = _get("final_loglik")
    n_edges_list  = _get("n_edges")
    actual_steps  = _get("iterations")
    stopped_flags = [m["train_summary"].get("stopped_early", False) for m in ts]

    fig, ax1 = plt.subplots(figsize=(max(8, len(tidxs) * 0.5), 4))
    ax1.plot(x, final_elbos, marker='o', color='steelblue', linewidth=2)
    for xi, yi, st, ns in zip(x, final_elbos, stopped_flags, actual_steps):
        label = f"{yi:.2f}\n({ns}st)"
        color = 'green' if st else 'gray'
        ax1.annotate(label, (xi, yi), textcoords="offset points",
                     xytext=(0, 6), ha='center', fontsize=6, color=color)
    ax2 = ax1.twinx()
    bar_colors = ['#2ca02c' if s else '#aec7e8' for s in stopped_flags]
    ax2.bar(x, actual_steps, color=bar_colors, alpha=0.3, width=0.4,
            label='Steps used (green=early stop)')
    ax2.set_ylabel("Steps used", color='gray')
    ax1.set_xticks(x); ax1.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel("Taxon"); ax1.set_ylabel("Final ELBO / n_sites")
    ax1.set_title("ELBO at Convergence + Steps Used per Taxon\n"
                  "(green annotation = early stopped; gray = ran to max_steps)")
    ax1.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_elbo.png"), dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(max(8, len(tidxs) * 0.5), 4))
    ax1.plot(x, final_ent, marker='o', color='green', linewidth=2, label='H(q)')
    ax1.set_ylabel("Shannon Entropy", color='green')
    ax2 = ax1.twinx()
    ax2.plot(x, final_maxqe, marker='s', color='purple', linewidth=2,
             linestyle='--', label='max q(e)')
    ax2.set_ylabel("Max q(e)", color='purple')
    ax2.set_ylim(0, 1.05)
    ax1.set_xticks(x); ax1.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel("Taxon")
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc='upper left', fontsize=8)
    plt.title("Posterior Confidence per Taxon"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_confidence.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(tidxs) * 1.2), 4))
    axes[0].bar(x, final_loglik, color='steelblue', alpha=0.8)
    axes[0].set_title("Final E[log p(data)] per Taxon"); axes[0].set_ylabel("Nats")
    axes[1].bar(x, final_kl_b, color='orange', alpha=0.8, label='KL_b')
    axes[1].bar(x, final_kl_r, color='red',    alpha=0.8, label='KL_r', bottom=final_kl_b)
    axes[1].set_title("Final KL per Taxon"); axes[1].legend()
    axes[2].plot(x, n_edges_list, marker='o', color='gray', linewidth=2)
    axes[2].set_title("Edges at Each Step (search space)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle("Cross-Taxon Summary"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_decomposition.png"), dpi=150)
    plt.close()

    print(f"\n[Category 3 plots saved to {run_dir}]")


# ══════════════════════════════════════════════════════════════════════════════
# Nexus / Newick I/O
# ══════════════════════════════════════════════════════════════════════════════

def escape_nexus_name(name: str) -> str:
    return f"'{name}'" if any(c in name for c in " \t\n(),:;[]=") else name


def to_newick(
    tree: UnrootedTree,
    taxon_names: List[str],
    label_mapper: Optional[Callable[[int], str]] = None,
) -> str:
    start_node = next(
        (u for u in tree.adj if u >= len(taxon_names)), None
    ) or next(iter(tree.adj), None)
    if start_node is None:
        return "();"

    visited: set = set()

    def build_sub(u: int) -> str:
        visited.add(u)
        children = [v for v in tree.neighbors(u) if v not in visited]
        if not children:
            if u >= len(taxon_names):
                raise RuntimeError(
                    f"Internal node {u} became terminal in Newick traversal."
                )
            return label_mapper(u) if label_mapper else taxon_names[u]
        parts = []
        for v in children:
            parts.append(f"{build_sub(v)}:{tree.length(u,v):.6f}")
        return "(" + ",".join(parts) + ")"

    visited.add(start_node)
    roots = [
        f"{build_sub(v)}:{tree.length(start_node,v):.6f}"
        for v in tree.neighbors(start_node)
    ]
    return "(" + ",".join(roots) + ");"


def to_newick_topology_only(
    tree: UnrootedTree,
    taxon_names: List[str],
) -> str:
    """
    Canonical topology string with NO branch lengths.
    Rooted deterministically at the internal node attached to Taxon 0 
    to guarantee invariance to insertion history and internal node IDs.
    """
    if not tree.adj:
        return "();"

    start_node = 0  # Always use Taxon 0 as the deterministic anchor
    
    # In an unrooted tree, a leaf has exactly one neighbor (an internal node).
    # We treat this connecting internal node as the temporary "root".
    root_internal = tree.neighbors(start_node)[0]
    visited = {start_node, root_internal}

    def build_sub(u: int) -> str:
        visited.add(u)
        children = [v for v in tree.neighbors(u) if v not in visited]
        
        # If it's a leaf, just return its name
        if not children:
            return taxon_names[u]
        
        # If it's an internal node, recursively build and SORT the subtrees
        subtrees = sorted(build_sub(v) for v in children)
        return "(" + ",".join(subtrees) + ")"

    # Traverse the other branches connected to our root_internal
    children = [v for v in tree.neighbors(root_internal) if v not in visited]
    subtrees = sorted(build_sub(v) for v in children)

    # Combine Taxon 0 with the other subtrees at the top level and sort them
    top_level = sorted([taxon_names[start_node]] + subtrees)
    
    return "(" + ",".join(top_level) + ");"

def save_nexus_trees(
    trees: List[UnrootedTree], taxon_names: List[str], path: str
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n\nBEGIN TAXA;\n")
        f.write(f"\tDIMENSIONS NTAX={len(taxon_names)};\n\tTAXLABELS\n")
        for name in taxon_names:
            f.write(f"\t\t{escape_nexus_name(name)}\n")
        f.write("\t;\nEND;\n\nBEGIN TREES;\n\tTRANSLATE\n")
        for i, name in enumerate(taxon_names):
            sep = "," if i < len(taxon_names) - 1 else ""
            f.write(f"\t\t{i+1} {escape_nexus_name(name)}{sep}\n")
        f.write("\t;\n")
        for idx, t in enumerate(trees):
            nwk = to_newick(t, taxon_names, label_mapper=lambda i: str(i + 1))
            f.write(f"\tTREE tree_{idx+1} = {nwk}\n")
        f.write("END;\n")


def save_training_tree_json(
    tree: UnrootedTree, names: List[str], path: str
) -> None:
    rec = [
        {"u_id": int(u), "v_id": int(v),
         "u_label": node_label(u, names), "v_label": node_label(v, names),
         "length": float(tree.length(u, v))}
        for u, v in tree.edges()
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_single_insertion_step(
    taxon_idx:    int,
    step_nets:    Dict[int, Tuple[RhoNet, BNet]],   # frozen prior-step networks
    new_leaf_lik: torch.Tensor,
    leaf_lik_all: Dict[int, torch.Tensor],
    taxon_names:  List[str],
    device:       torch.device,
    args:         argparse.Namespace,
    step_out_dir: Optional[str] = None,
) -> Tuple[RhoNet, BNet, Dict]:
    """
    Train the CA-VI networks for the insertion of taxon `taxon_idx`.

    Coordinate ascent structure per outer step:
        for ca_round in n_ca_rounds:
            Phase 1: n_phase1_steps gradient updates on rho_net only
            Phase 2: n_phase2_steps gradient updates on b_net only
    """
    n_use       = len(taxon_names)
    n_sites     = new_leaf_lik.shape[0]
    new_leaf_lik = new_leaf_lik.to(device)
    pi_dna      = PI_DNA.to(device)

    rho_net = RhoNet(in_channels=EDGE_FEATURE_DIM, hidden=args.hidden_dim).to(device)
    b_net   = BNet(hidden=args.hidden_dim).to(device)
    rho_net.train(); b_net.train()

    optimizer_rho = torch.optim.Adam(rho_net.parameters(), lr=args.lr)
    optimizer_b   = torch.optim.Adam(b_net.parameters(),   lr=args.lr)

    # ── History buffers ────────────────────────────────────────────────────────
    elbo_hist          = []
    entropy_hist       = []
    max_qe_hist        = []
    kl_b_hist          = []
    kl_r_hist          = []
    loglik_hist        = []
    grad_norm_rho_hist = []
    grad_norm_b_hist   = []
    mu_r_top_hist      = []
    sigma_r_top_hist   = []
    mu_b_top_hist      = []
    sigma_b_top_hist   = []

    # ── Early-stopping state ───────────────────────────────────────────────
    # EMA of ELBO/n_sites, updated every outer step.
    # Temperature anneals from temp_start → temp_end over min_steps, then holds.
    # Convergence: after min_steps, if EMA improvement over last `patience`
    # steps < conv_tol, we stop early.
    ema_val        = None          # running EMA of ELBO/n_sites
    ema_at_patience_ago = None     # EMA value `patience` steps ago
    ema_hist_full  = []            # all EMA values (for plotting)
    stopped_early    = False
    stop_reason      = "max_steps reached"
    early_stop_step  = None    # step_iter value when break fires (≠ actual_steps)

    pbar       = tqdm(total=args.max_steps, desc=f"Train Taxon {taxon_idx}",
                      leave=False, dynamic_ncols=True)
    final_tree = None

    for step_iter in range(1, args.max_steps + 1):
        # Temperature: anneal over min_steps then hold at temp_end.
        # This guarantees the model always sees full annealing regardless
        # of when early stopping fires.
        anneal_prog  = min(1.0, (step_iter - 1) / max(1, args.min_steps - 1))
        current_temp = args.temp_start * (args.temp_end / args.temp_start) ** anneal_prog

        # ── Sample a context tree using frozen prior-step nets ─────────────────
        allocator = InternalNodeAllocator(start=n_use)
        center_id = allocator.alloc()
        tree      = build_t3_star(
            args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=center_id
        )
        with torch.no_grad():
            for step in range(3, taxon_idx):
                rho_net_s, b_net_s = step_nets[step]
                (edges_s, q_e_s, _, mu_r_s, lsr_s, _, _, h_s,
                 rho_np_s, b_np_s, _, _, _, _, _) = evaluate_edges_ca(
                    tree, leaf_lik_all[step], leaf_lik_all,
                    rho_net_s, b_net_s, device,
                    k_samples=args.k_train, jc_rate=args.jc_rate,
                    temperature=args.sample_temp,
                )
                chosen_edge, chosen_rho, chosen_b = choose_action_ca(
                    edges_s, q_e_s, rho_np_s, b_np_s, mu_r_s, h_s, b_net_s, device,
                    sample_continuous=True,
                )
                insert_taxon_on_edge(tree, chosen_edge, step,
                                     chosen_rho, chosen_b, allocator)

        # ── Build edge features for THIS taxon (static within outer step) ──────
        leaf_lik_current = {
            k: v for k, v in leaf_lik_all.items()
            if k in tree.adj and len(tree.adj[k]) == 1
        }
        with torch.no_grad():
            _, cav_cache   = compute_messages_and_cavities(tree, leaf_lik_current)
            edges          = tree.edges()
            n_nodes        = len(tree.adj)
            n_edges        = len(edges)
            edge_feats_raw = build_edge_features(
                edges=edges, cav_cache=cav_cache,
                new_leaf_lik=new_leaf_lik, device=device, n_nodes=n_nodes,
                tree=tree, jc_rate=args.jc_rate,
            )
        edge_feats = edge_feats_raw.detach()

        # Pre-stack cavities ONCE per outer step and reuse across all inner CA steps.
        # Avoids re-extracting (E, L, 4) tensors from the dict on every gradient step.
        stacked_cavs = _stack_cavities(edges, tree, cav_cache, device)

        # ══ Coordinate Ascent Rounds ══════════════════════════════════════════
        for ca_round in range(args.n_ca_rounds):

            # ── Phase 1: optimise RhoNet ───────────────────────────────────────
            for _ in range(args.n_phase1_steps):
                rho_net.train(); b_net.eval()
                result = forward_ca(
                    tree, edges, edge_feats,
                    rho_net, b_net,
                    new_leaf_lik, cav_cache, pi_dna,
                    args.k_train, args.jc_rate, device,
                    phase=1,
                    stacked_cavs=stacked_cavs,
                )
                elbo_e      = result["elbo_e"]
                tree_elbo   = current_temp * torch.logsumexp(
                    elbo_e / current_temp, dim=0
                ) - math.log(n_edges)
                loss = -tree_elbo / n_sites

                optimizer_rho.zero_grad()
                loss.backward()
                gnorm = sum(
                    p.grad.norm().item() ** 2
                    for p in rho_net.parameters() if p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(rho_net.parameters(), 1.0)
                optimizer_rho.step()
                # store grad norm of last phase-1 inner step per outer iter
                _gn_rho = gnorm



            # ── Phase 2: optimise BNet ─────────────────────────────────────────
            for _ in range(args.n_phase2_steps):

                # Resample fresh rho noise on EVERY gradient step to prevent BNet overfitting
                with torch.no_grad():
                    mu_r_snap, lsr_snap, h_snap = rho_net(edge_feats)
                    std_r_snap = torch.exp(lsr_snap)
                    eps_r_snap = torch.randn(args.k_train, n_edges, device=device)
                    rho_fixed  = torch.sigmoid(
                        mu_r_snap.unsqueeze(0) + std_r_snap.unsqueeze(0) * eps_r_snap
                    )  # (K, E) detached
                    h_fixed    = h_snap.detach()
                
                rho_net.eval(); b_net.train()
                result = forward_ca(
                    tree, edges, edge_feats,
                    rho_net, b_net,
                    new_leaf_lik, cav_cache, pi_dna,
                    args.k_train, args.jc_rate, device,
                    phase=2,
                    rho_fixed=rho_fixed,
                    h_fixed=h_fixed,
                    stacked_cavs=stacked_cavs,
                )
                elbo_e    = result["elbo_e"]
                tree_elbo = current_temp * torch.logsumexp(
                    elbo_e / current_temp, dim=0
                ) - math.log(n_edges)
                loss = -tree_elbo / n_sites

                optimizer_b.zero_grad()
                loss.backward()
                gnorm = sum(
                    p.grad.norm().item() ** 2
                    for p in b_net.parameters() if p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(b_net.parameters(), 1.0)
                optimizer_b.step()
                _gn_b = gnorm

        # ══ End of CA rounds — evaluate full ELBO for logging ═════════════════
        rho_net.eval(); b_net.eval()
        with torch.no_grad():
            result_eval = forward_ca(
                tree, edges, edge_feats,
                rho_net, b_net,
                new_leaf_lik, cav_cache, pi_dna,
                args.k_train, args.jc_rate, device,
                phase=1,   # full ELBO (both KL terms)
                stacked_cavs=stacked_cavs,
            )

        elbo_e_eval  = result_eval["elbo_e"]
        kl_rho_eval  = result_eval["kl_rho"]
        kl_b_eval    = result_eval["kl_b_mc"]
        loglik_eval  = result_eval["loglik_mc"]
        mu_r_eval    = result_eval["mu_r"]
        lsr_eval     = result_eval["log_sigma_r"]
        mu_b_eval    = result_eval["mu_b"]    # (K, E)
        lsb_eval     = result_eval["log_sigma_b"]

        tree_elbo_eval = (
            current_temp * torch.logsumexp(elbo_e_eval / current_temp, dim=0)
            - math.log(n_edges)
        )
        val = (tree_elbo_eval / n_sites).item()

        # Score the top edge for logging using strictly the Log-Likelihood!
        q_e_eval = torch.softmax(loglik_eval / current_temp, dim=0)
        entropy  = -torch.sum(q_e_eval * torch.log(q_e_eval + EPS)).item()
        top_idx  = q_e_eval.argmax().item()

        elbo_hist.append(val)
        entropy_hist.append(entropy)
        max_qe_hist.append(q_e_eval.max().item())
        kl_b_hist.append(kl_b_eval.mean().item())
        kl_r_hist.append(kl_rho_eval.mean().item())
        loglik_hist.append(loglik_eval.mean().item())
        grad_norm_rho_hist.append(_gn_rho)
        grad_norm_b_hist.append(_gn_b)
        mu_r_top_hist.append(mu_r_eval[top_idx].item())
        sigma_r_top_hist.append(torch.exp(lsr_eval[top_idx]).item())
        mu_b_top_hist.append(mu_b_eval[:, top_idx].mean().item())
        sigma_b_top_hist.append(torch.exp(lsb_eval[:, top_idx]).mean().item())

        # ── EMA update ────────────────────────────────────────────────────
        alpha = args.ema_alpha
        ema_val = val if ema_val is None else alpha * val + (1 - alpha) * ema_val
        ema_hist_full.append(ema_val)

        # Remember EMA from `patience` steps ago for the improvement check
        if len(ema_hist_full) > args.patience:
            ema_at_patience_ago = ema_hist_full[-args.patience - 1]

        pbar.update(1)
        pbar.set_postfix(
            elbo=f"{val:.4f}",
            ema=f"{ema_val:.4f}",
            T=f"{current_temp:.1f}",
        )

        final_tree = tree  # always keep last tree

        # ── Early stopping check ──────────────────────────────────────────
        # Only fire after min_steps AND after we have a full patience window.
        # Earliest possible stop: step min_steps + patience.
        # (step_iter >= min_steps is implied by the third condition)
        if (ema_at_patience_ago is not None
                and step_iter >= args.min_steps + args.patience):
            improvement = ema_val - ema_at_patience_ago
            if improvement < args.conv_tol:
                stop_reason      = (f"converged at step {step_iter} "
                                    f"(EMA Δ={improvement:.2e} < tol={args.conv_tol:.1e})")
                stopped_early    = True
                early_stop_step  = step_iter   # exact step convergence detected
                break

    pbar.close()
    actual_steps = len(elbo_hist)
    if stopped_early:
        print(f"  ✓ Early stop: {stop_reason}")
    else:
        print(f"  → Ran full {actual_steps} steps (did not converge within tolerance)")

    summary = {
        "final_scaled_elbo": elbo_hist[-1],
        "iterations":        actual_steps,
        "max_steps":         args.max_steps,
        "stopped_early":     stopped_early,
        "stop_reason":       stop_reason,
        "final_ema_elbo":    ema_val,
        "final_entropy":     entropy_hist[-1],
        "final_max_qe":      max_qe_hist[-1],
        "final_kl_b":        kl_b_hist[-1],
        "final_kl_r":        kl_r_hist[-1],
        "final_loglik":      loglik_hist[-1],
        "n_edges":           n_edges,
    }

    # ── Save per-step plots ────────────────────────────────────────────────────
    if step_out_dir and final_tree is not None:
        rho_net.eval(); b_net.eval()
        with torch.no_grad():
            (edges_f, q_e_f, elbo_e_f,
             mu_r_f, lsr_f, mu_b_f, lsb_f, h_f,
             rho_np_f, b_np_f,
             _, _, loglik_mc_f, _, edge_feats_f) = evaluate_edges_ca(
                final_tree, new_leaf_lik, leaf_lik_all,
                rho_net, b_net, device,
                k_samples=args.k_eval, jc_rate=args.jc_rate,
                temperature=current_temp,
            )
        edge_labels = [
            f"({node_label(u, taxon_names)}, {node_label(v, taxon_names)})"
            for u, v in edges_f
        ]
        save_step_plots(
            out_dir=step_out_dir,
            edge_labels=edge_labels,
            elbo_hist=elbo_hist,
            ema_hist=ema_hist_full,
            stopped_early=stopped_early,
            stop_step=early_stop_step,   # None if not early-stopped; else the exact step convergence fired
            q_final=q_e_f.detach().cpu().numpy(),
            elbo_final=elbo_e_f.detach().cpu().numpy(),
            loglik_final=loglik_mc_f.detach().cpu().numpy(),
            rho_samples_final=rho_np_f,
            b_samples_final=b_np_f,
            entropy_hist=entropy_hist,
            max_qe_hist=max_qe_hist,
            kl_b_hist=kl_b_hist,
            kl_r_hist=kl_r_hist,
            loglik_hist=loglik_hist,
            grad_norm_rho_hist=grad_norm_rho_hist,
            grad_norm_b_hist=grad_norm_b_hist,
            temp_start=args.temp_start,
            temp_end=args.temp_end,
            max_steps=args.max_steps,
            min_steps=args.min_steps,
            mu_r_top_hist=mu_r_top_hist,
            sigma_r_top_hist=sigma_r_top_hist,
            mu_b_top_hist=mu_b_top_hist,
            sigma_b_top_hist=sigma_b_top_hist,
            mu_b_final_mean=mu_b_f.detach().cpu().numpy().mean(0),
            log_sigma_b_final_mean=lsb_f.detach().cpu().numpy().mean(0),
            taxon_idx=taxon_idx,
            n_sites=n_sites,
        )
        torch.save(rho_net.state_dict(), os.path.join(step_out_dir, "rho_net.pt"))
        torch.save(b_net.state_dict(),   os.path.join(step_out_dir, "b_net.pt"))

    return rho_net, b_net, summary


# ══════════════════════════════════════════════════════════════════════════════
# Training orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_training_pass(
    args:         argparse.Namespace,
    names:        List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    device:       torch.device,
    run_dir:      str,
):
    n_use         = len(names)
    step_models:  Dict[int, Tuple[str, str]] = {}    # (rho_path, b_path)
    step_nets:    Dict[int, Tuple[RhoNet, BNet]] = {}
    step_meta     = []

    for taxon_idx in range(3, n_use):
        print(f"\n=== Train Step  taxon_idx={taxon_idx}  name={names[taxon_idx]} ===")
        step_dir = os.path.join(
            run_dir, f"train_step_{taxon_idx:03d}_{names[taxon_idx]}"
        )
        os.makedirs(step_dir, exist_ok=True)

        rho_net, b_net, train_summary = train_single_insertion_step(
            taxon_idx=taxon_idx,
            step_nets=step_nets,
            new_leaf_lik=leaf_lik_all[taxon_idx],
            leaf_lik_all=leaf_lik_all,
            taxon_names=names,
            device=device,
            args=args,
            step_out_dir=step_dir,
        )

        rho_path = os.path.join(step_dir, "rho_net.pt")
        b_path   = os.path.join(step_dir, "b_net.pt")
        # (already saved inside train_single_insertion_step, but record paths)
        step_models[taxon_idx] = (rho_path, b_path)

        rho_net.eval(); b_net.eval()
        step_nets[taxon_idx] = (rho_net, b_net)

        step_meta.append({
            "taxon_idx":    taxon_idx,
            "taxon_name":   names[taxon_idx],
            "rho_path":     rho_path,
            "b_path":       b_path,
            "train_summary": train_summary,
        })

    trained_tree = sample_tree_from_trained_steps(
        args, names, leaf_lik_all, step_nets, device
    )
    save_crosstep_plots(run_dir, step_meta, names)
    return trained_tree, step_models, step_meta


def preload_step_nets_ca(
    step_models: Dict[int, Tuple[str, str]],
    hidden_dim:  int,
    device:      torch.device,
) -> Dict[int, Tuple[RhoNet, BNet]]:
    step_nets = {}
    for taxon_idx, (rho_path, b_path) in step_models.items():
        rho_net = RhoNet(in_channels=EDGE_FEATURE_DIM, hidden=hidden_dim).to(device)
        rho_net.load_state_dict(
            torch.load(rho_path, map_location=device, weights_only=True)
        )
        rho_net.eval()

        b_net = BNet(hidden=hidden_dim).to(device)
        b_net.load_state_dict(
            torch.load(b_path, map_location=device, weights_only=True)
        )
        b_net.eval()
        step_nets[taxon_idx] = (rho_net, b_net)
    return step_nets


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════

def sample_tree_from_trained_steps(
    args:         argparse.Namespace,
    names:        List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    step_nets:    Dict[int, Tuple[RhoNet, BNet]],
    device:       torch.device,
) -> UnrootedTree:
    n_use     = len(names)
    allocator = InternalNodeAllocator(start=n_use)
    center_id = allocator.alloc()
    tree      = build_t3_star(
        args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=center_id
    )

    with torch.no_grad():
        for taxon_idx in range(3, n_use):
            rho_net, b_net = step_nets[taxon_idx]
            (edges, q_e, _, mu_r, lsr, _, _, h,
             rho_np, b_np, _, _, _, _, _) = evaluate_edges_ca(
                tree, leaf_lik_all[taxon_idx], leaf_lik_all,
                rho_net, b_net, device,
                k_samples=args.k_eval, jc_rate=args.jc_rate,
                temperature=args.sample_temp,
            )
            chosen_edge, chosen_rho, chosen_b = choose_action_ca(
                edges, q_e, rho_np, b_np, mu_r, h, b_net, device, 
                sample_continuous=True
            )
            insert_taxon_on_edge(tree, chosen_edge, taxon_idx,
                                 chosen_rho, chosen_b, allocator)

    return tree


def sample_tree_with_log_q(
    args:         argparse.Namespace,
    names:        List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    step_nets:    Dict[int, Tuple[RhoNet, BNet]],
    device:       torch.device,
) -> Tuple[UnrootedTree, float, float, float]:
    """
    Sample ONE complete tree and return terms for MLL importance weighting.

    Weight: log_p_Y + log_p_b - log_q
    where log_q = sum_steps [ log q(e) + log q(rho|e) + log q(b|rho,e) ]

    NOTE: the 3 fixed initial star branches are included in log_p_b but not
    in log_q. This makes the MLL estimate conservative (lower bound).  To
    remove this bias, either (a) also place a variational distribution over
    the initial 3 branches, or (b) exclude those 3 edges from log_p_b.
    """
    n_use     = len(names)
    allocator = InternalNodeAllocator(start=n_use)
    center_id = allocator.alloc()
    tree      = build_t3_star(
        args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=center_id
    )

    log_q   = 0.0
    log_p_Y = None

    with torch.no_grad():
        for taxon_idx in range(3, n_use):
            rho_net, b_net = step_nets[taxon_idx]
            is_last        = (taxon_idx == n_use - 1)

            # Build edge features
            leaf_lik_current = {
                k: v for k, v in leaf_lik_all.items()
                if k in tree.adj and len(tree.adj[k]) == 1
            }
            _, cav_cache = compute_messages_and_cavities(tree, leaf_lik_current)
            edges        = tree.edges()
            n_nodes      = len(tree.adj)
            n_edges      = len(edges)
            new_lik      = leaf_lik_all[taxon_idx].to(device)
            edge_feats   = build_edge_features(
                edges=edges, cav_cache=cav_cache,
                new_leaf_lik=new_lik, device=device, n_nodes=n_nodes,
                tree=tree, jc_rate=args.jc_rate,
            )
            pi_dna = PI_DNA.to(device)

            # ── Get rho variational params ────────────────────────────────────
            mu_r, log_sigma_r, h = rho_net(edge_feats)   # (E,), (E,), (E,H)

            # ── Sample rho for ALL edges ──────────────────────────────────────
            std_r    = torch.exp(log_sigma_r)
            eps_r    = torch.randn_like(mu_r)
            z_r_samp = mu_r + std_r * eps_r
            rho_samp = torch.sigmoid(z_r_samp).unsqueeze(0)       # (1, E)

            # ── Sample b for ALL edges conditioned on sampled rho ─────────────
            mu_b, log_sigma_b = b_net(h.detach(), rho_samp)       # (1, E)
            std_b    = torch.exp(log_sigma_b)
            eps_b    = torch.randn_like(mu_b)
            z_b_samp = mu_b + std_b * eps_b
            # Clamp z_b_samp before exp() to prevent overflow
            b_samp   = torch.exp(z_b_samp.clamp(max=math.log(2.0))).clamp(min=1e-8)

            # ── Compute likelihood using the actual samples ───────────────────
            _sc = _stack_cavities(edges, tree, cav_cache, device)
            loglik_samp = compute_loglik_all_edges(
                *_sc, rho_samp, b_samp, new_lik, pi_dna, args.jc_rate
            ) # (E,)
            
            # SCORE EDGES STRICTLY BY FELSENSTEIN LIKELIHOOD 
            # (No n_sites scaling here! We want the proposal to match the true peaked posterior for IS)
            q_e = torch.softmax(loglik_samp, dim=0)

            # ── Sample edge ───────────────────────────────────────────────────
            q_e_cpu     = q_e.detach().cpu().clamp(min=0.0)
            q_e_cpu     = q_e_cpu / (q_e_cpu.sum() + EPS)
            chosen_idx  = torch.multinomial(q_e_cpu, 1).item()

            # Retrieve the exact rho and b that were already sampled & evaluated
            chosen_rho = float(rho_samp[0, chosen_idx])
            chosen_b   = float(b_samp[0, chosen_idx])
            
            # Retrieve parameters for the log_q calculation
            mu_r_i = mu_r[chosen_idx].item()
            lsr_i  = log_sigma_r[chosen_idx].item()
            mu_b_i = mu_b[0, chosen_idx].item()
            lsb_i  = log_sigma_b[0, chosen_idx].item()

            # ── Accumulate log q ──────────────────────────────────────────────
            log_q += (math.log(float(q_e_cpu[chosen_idx]) + 1e-30)
                      + log_logistic_normal_density(chosen_rho, mu_r_i, lsr_i)
                      + log_lognormal_density(chosen_b, mu_b_i, lsb_i))


            # <--- NEW: Accumulate the exact prior aligned with the ELBO
            log_p_prior += (math.log(PRIOR_LAM_B) - PRIOR_LAM_B * chosen_b)
            # Prior for rho corresponds to z ~ N(0,1), which is LogitNormal with mu=0, log_sigma=0
            log_p_prior += log_logistic_normal_density(chosen_rho, 0.0, 0.0)
            # ── At last step: compute exact log p(Y|τ,b) ─────────────────────
            if is_last:
                rho_t = torch.tensor([chosen_rho], dtype=torch.float32, device=device)
                b_t   = torch.tensor([chosen_b],   dtype=torch.float32, device=device)
                log_p_Y = insertion_loglik_from_edge_batched(
                    tree, edges[chosen_idx], rho_t, b_t,
                    new_lik, cav_cache, pi_dna, rate=args.jc_rate,
                ).item()

            insert_taxon_on_edge(
                tree, edges[chosen_idx], taxon_idx,
                chosen_rho, chosen_b, allocator,
            )


    return tree, log_q, log_p_Y, log_p_prior


# ══════════════════════════════════════════════════════════════════════════════
# MLL Estimation
# ══════════════════════════════════════════════════════════════════════════════

def estimate_mll(
    args:         argparse.Namespace,
    names:        List[str],
    leaf_lik_all: Dict[int, torch.Tensor],
    step_nets:    Dict[int, Tuple[RhoNet, BNet]],
    device:       torch.device,
    k_mll:        int = 1000,
) -> Dict:
    """
    MLL via self-normalised importance sampling:
        MLL ≈ log [ (1/K) Σ_i  exp(log_p_Y_i + log_p_b_i - log_q_i) ]

    Excludes the uniform topology prior p(τ) — same convention as PhyloGFN/ARTree.
    """
    print(f"\n  Estimating MLL with K={k_mll} importance samples...")
    log_weights = []

    for _ in tqdm(range(k_mll), desc="MLL samples", leave=False, dynamic_ncols=True):
        _, log_q, log_p_Y, log_p_prior = sample_tree_with_log_q(
            args, names, leaf_lik_all, step_nets, device
        )
        if log_p_Y is None:
            continue
        log_weights.append(log_p_Y + log_p_prior - log_q)

    log_weights = np.array(log_weights)
    max_lw  = log_weights.max()
    mll     = float(max_lw + math.log(np.exp(log_weights - max_lw).mean()))
    mll_std = float(np.std(log_weights))

    log_sum_w  = max_lw + math.log(np.exp(log_weights - max_lw).sum())
    log_sum_w2 = (np.log(np.exp(2.0 * (log_weights - max_lw)).sum())
                  + 2.0 * max_lw)
    ess        = float(math.exp(2.0 * log_sum_w - log_sum_w2))
    ess_pct    = 100.0 * ess / k_mll

    print(f"\n{'='*55}")
    print(f"  MARGINAL LOG-LIKELIHOOD (MLL) ESTIMATE")
    print(f"{'='*55}")
    print(f"  MLL              = {mll:.4f}  nats")
    print(f"  K                = {k_mll}")
    print(f"  ESS              = {ess:.1f} / {k_mll}  ({ess_pct:.1f}%)")
    print(f"  log-weight mean  = {log_weights.mean():.4f}")
    print(f"  log-weight std   = {mll_std:.4f}")
    print(f"  log-weight range = [{log_weights.min():.2f}, {log_weights.max():.2f}]")
    print(f"{'='*55}")
    print(f"  Reference: PhyloGFN DS1 = -7108.4  |  MrBayes DS1 ≈ -7108.3")
    print(f"  Gap from PhyloGFN ref   = {mll - (-7108.4):+.2f} nats")
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
        "note_topology_prior": "uniform p(tau) excluded (same as PhyloGFN/ARTree)",
        "note_initial_branches": (
            "3 initial star branches fixed by JC MLE — prior included in log_p_b "
            "but not in log_q, making MLL conservative."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Post-training diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def save_sampled_tree_plots(
    sampled_trees: List[UnrootedTree],
    names:         List[str],
    run_dir:       str,
    leaf_lik_all:  Dict[int, torch.Tensor],
    device:        torch.device,
    args:          argparse.Namespace,
    step_nets:     Dict[int, Tuple[RhoNet, BNet]],
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    n_taxa         = len(names)
    expected_edges = 2 * n_taxa - 3

    # ── Topology frequency ────────────────────────────────────────────────────
    # Use topology-only strings (no branch lengths) for counting.
    # to_newick() includes :.6f branch lengths, making every sample
    # appear unique even if topologically identical. Fixed here.
    topo_strings    = [to_newick_topology_only(t, names) for t in sampled_trees]
    newick_strings  = [to_newick(t, names) for t in sampled_trees]  # kept for log only
    topology_counts = Counter(topo_strings)
    n_unique        = len(topology_counts)
    top_topos       = topology_counts.most_common(min(20, n_unique))
    top_freqs       = [c / len(sampled_trees) for _, c in top_topos]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top_freqs)), top_freqs, color='steelblue', alpha=0.8)
    plt.xlabel("Topology rank (0 = most common)")
    plt.ylabel("Posterior probability estimate")
    plt.title(
        f"Top-{len(top_freqs)} Topology Frequencies\n"
        f"{n_unique} unique / {len(sampled_trees)} samples  |  "
        f"Top freq: {top_freqs[0]:.3f}"
    )
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "topology_frequencies.png"), dpi=150)
    plt.close()

    print(f"\n[Cat 4] Topology diversity: {n_unique} unique / {len(sampled_trees)} samples")
    for rank, (topo, cnt) in enumerate(top_topos[:5]):
        print(f"  Rank {rank+1}: freq={cnt/len(sampled_trees):.4f}  {topo[:80]}")

    # ── Branch length distribution ────────────────────────────────────────────
    all_bl = np.array([
        t.length(u, v) for t in sampled_trees for u, v in t.edges()
    ])
    plt.figure(figsize=(8, 4))
    plt.hist(all_bl, bins=60, density=True, color='steelblue', alpha=0.75)
    plt.axvline(np.median(all_bl), color='red',    linestyle='--', linewidth=1.5,
                label=f'Median: {np.median(all_bl):.4f}')
    plt.axvline(np.mean(all_bl),   color='orange', linestyle='--', linewidth=1.5,
                label=f'Mean:   {np.mean(all_bl):.4f}')
    plt.axvline(1.0 / PRIOR_LAM_B, color='black',  linestyle=':',  linewidth=1.5,
                label=f'Prior mean: {1/PRIOR_LAM_B:.3f}')
    plt.xlabel("Branch length (sub/site)"); plt.ylabel("Density")
    plt.title("Branch Length Distribution Across All Sampled Trees")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "sampled_branch_lengths.png"), dpi=150)
    plt.close()

    # ── Edge count sanity check ───────────────────────────────────────────────
    ec  = Counter(len(t.edges()) for t in sampled_trees)
    bad = sum(1 for t in sampled_trees if len(t.edges()) != expected_edges)
    plt.figure(figsize=(6, 3))
    plt.bar([str(k) for k in sorted(ec)],
            [ec[k] for k in sorted(ec)],
            color=['red' if k != expected_edges else 'steelblue' for k in sorted(ec)])
    plt.title(
        f"Edge Count Check (expected {expected_edges} = 2×{n_taxa}-3)\n"
        f"{'✓ All correct' if bad == 0 else f'⚠ {bad} trees wrong!'}"
    )
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "edge_count_check.png"), dpi=150)
    plt.close()
    if bad > 0:
        print(f"  ⚠ WARNING: {bad} sampled trees have wrong edge count!")
    else:
        print(f"  ✓ All {len(sampled_trees)} sampled trees have correct edge count ({expected_edges})")

    # ── Numerical health: log-scale cavity check ──────────────────────────────
    last_tidx = max(step_nets.keys())
    rho_net_l, b_net_l = step_nets[last_tidx]

    allocator_tmp = InternalNodeAllocator(start=n_taxa)
    tree_tmp      = build_t3_star(
        args.edge_len_0, args.edge_len_1, args.edge_len_2,
        center_idx=allocator_tmp.alloc(),
    )
    with torch.no_grad():
        for tidx in range(3, n_taxa):
            rn, bn = step_nets[tidx]
            # 1. Update the unpacking to capture rho_np_t and b_np_t 
            (edges_t, q_t, _, mu_r_t, lsr_t, _, _, h_t,
             rho_np_t, b_np_t, _, _, _, _, _) = evaluate_edges_ca(
                tree_tmp, leaf_lik_all[tidx], leaf_lik_all,
                rn, bn, device, k_samples=10, jc_rate=args.jc_rate, temperature=1.0,
            )
            
            # 2. Update the function call to use the new signature
            e_c, r_c, b_c = choose_action_ca(
                edges_t, q_t, rho_np_t, b_np_t, mu_r_t, h_t, bn, device, 
                sample_continuous=False  # (This one naturally uses the MAP logic we preserved!)
            )
            insert_taxon_on_edge(tree_tmp, e_c, tidx, r_c, b_c, allocator_tmp)

    leaf_lik_curr = {
        k: v for k, v in leaf_lik_all.items()
        if k in tree_tmp.adj and len(tree_tmp.adj[k]) == 1
    }
    _, cav_final = compute_messages_and_cavities(tree_tmp, leaf_lik_curr)

    scale_means, scale_mins, scale_maxs = [], [], []
    has_nan = has_inf = False
    for _, (_, ls) in cav_final.items():
        s = ls.detach().cpu()
        scale_means.append(s.mean().item())
        scale_mins.append(s.min().item())
        scale_maxs.append(s.max().item())
        if torch.isnan(s).any(): has_nan = True
        if torch.isinf(s).any(): has_inf = True

    plt.figure(figsize=(8, 4))
    plt.hist(scale_means, bins=30, color='teal', alpha=0.75)
    plt.axvline(0.0, color='red', linestyle='--', linewidth=1.5, label='0 (bad)')
    status = (["⚠ NaN!"] if has_nan else []) + (["⚠ -Inf!"] if has_inf else []) \
             or ["✓ No NaN/-Inf"]
    plt.title("Cat 5: Log-Scale Health\n" + "  ".join(status))
    plt.xlabel("Mean log_scale"); plt.ylabel("Count"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "log_scale_health.png"), dpi=150)
    plt.close()
    print(f"  Log-scale: min={min(scale_mins):.2f}  mean={np.mean(scale_means):.2f}  "
          f"max={max(scale_maxs):.2f}  " + ("⚠ NaN!" if has_nan else ""))

    # ── Actual Model Loglik Check (Top 5 edges) ──────────────────────────────
    pi_dna = PI_DNA.to(device)
    with torch.no_grad():
        # Actually evaluate the temporary tree using the network and samples
        (edges_f, q_t, _, _, _, _, _, _,
         _, _, _, _, loglik_mc, _, _) = evaluate_edges_ca(
            tree_tmp, leaf_lik_all[last_tidx], leaf_lik_all,
            rho_net_l, b_net_l, device, k_samples=args.k_eval, 
            jc_rate=args.jc_rate, temperature=1.0,
        )
    
    # Sort edges by likelihood to show the top 5 BEST edges
    sorted_indices = torch.argsort(loglik_mc, descending=True)[:min(5, len(edges_f))]
    site_lls = loglik_mc[sorted_indices].cpu().numpy()
    edge_labels = [f"({edges_f[i][0]},{edges_f[i][1]})" for i in sorted_indices]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(site_lls)), site_lls, color='teal', alpha=0.8)
    plt.xticks(range(len(site_lls)), edge_labels, rotation=45)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Top 5 Edges"); plt.ylabel("True Sampled Log-lik")
    plt.title("Cat 5: Actual Per-Edge Log-Likelihoods (from NN samples)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "site_loglik_check.png"), dpi=150)
    plt.close()

    print(f"[plots saved to {run_dir}]")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phylogenetic VI with Per-Edge Coordinate Ascent (CA-VI)"
    )
    p.add_argument("--fasta",         type=str,   required=True)
    p.add_argument("--n_taxa_use",    type=int,   default=None)
    p.add_argument("--max_steps",     type=int,   default=500,
                   help="Outer iterations (tree samples) per taxon")
    p.add_argument("--n_ca_rounds",   type=int,   default=2,
                   help="Coordinate ascent rounds per outer step")
    p.add_argument("--n_phase1_steps", type=int,  default=5,
                   help="RhoNet gradient steps per CA round")
    p.add_argument("--n_phase2_steps", type=int,  default=5,
                   help="BNet gradient steps per CA round")
    p.add_argument("--lr",            type=float, default=3e-3)
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--k_train",       type=int,   default=20,
                   help="MC samples (rho/b) during training")
    p.add_argument("--k_eval",        type=int,   default=100,
                   help="MC samples for sampling & MLL")
    p.add_argument("--n_tree_samples", type=int,  default=1000)
    p.add_argument("--k_mll",         type=int,   default=1000)
    p.add_argument("--temp_start",    type=float, default=50.0)
    p.add_argument("--temp_end",      type=float, default=1.0)
    p.add_argument("--sample_temp",   type=float, default=2.0,
                   help="Temperature for posterior sampling (>1 = more diverse)")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--jc_rate",       type=float, default=1.0)
    p.add_argument("--cuda",          action="store_true")
    p.add_argument("--out_root",      type=str,   default="results_ca_vi")
    p.add_argument("--num_threads",   type=int,   default=4,
                   help="PyTorch CPU threads. Set <= cores to dedicate on shared server.")
    # ── Early stopping ──────────────────────────────────────────────────
    p.add_argument("--min_steps",  type=int,   default=80,
                   help="Minimum outer iterations before checking convergence. "
                        "Also controls how long temperature anneals: temp reaches "
                        "temp_end at step min_steps and holds there.")
    p.add_argument("--patience",   type=int,   default=50,
                   help="Stop if EMA-ELBO has not improved by more than conv_tol "
                        "over this many consecutive steps.")
    p.add_argument("--conv_tol",   type=float, default=2e-4,
                   help="Minimum absolute improvement in EMA-ELBO/n_sites over "
                        "the patience window required to keep training. "
                        "Smaller = stricter convergence (e.g. 1e-4), "
                        "Larger = earlier stopping (e.g. 5e-4).")
    p.add_argument("--ema_alpha",  type=float, default=0.1,
                   help="EMA smoothing for convergence detection. "
                        "Smaller = smoother but slower to react (default 0.1).")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    # ── CPU thread control (prevent starving shared server) ───────────────
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(max(1, args.num_threads // 2))
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS",    str(args.num_threads))
    _os.environ.setdefault("MKL_NUM_THREADS",    str(args.num_threads))
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.num_threads))
    print(f"CPU threads: intra-op={args.num_threads}, "
          f"inter-op={max(1, args.num_threads // 2)}")
    set_seed(args.seed)

    names_all, seqs_all = parse_fasta_in_order(args.fasta)
    n_total = len(names_all)
    n_use   = n_total if args.n_taxa_use is None else min(args.n_taxa_use, n_total)
    if n_use < 4:
        raise ValueError("Need at least 4 taxa.")

    names        = names_all[:n_use]
    seqs         = seqs_all[:n_use]
    leaf_lik_all = {i: seq_to_likelihood_matrix(seqs[i]) for i in range(n_use)}

    e0, e1, e2 = compute_t3_branch_lengths(seqs[0], seqs[1], seqs[2])
    args.edge_len_0 = e0
    args.edge_len_1 = e1
    args.edge_len_2 = e2
    print(f"Initial T3 branch lengths: {e0:.4f}, {e1:.4f}, {e2:.4f}")

    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_root, f"run_{stamp}_ntaxa{n_use}")
    os.makedirs(run_dir, exist_ok=False)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Output dir : {os.path.abspath(run_dir)}")
    print(f"Device     : {device}")
    print(f"Taxa       : {n_use}/{n_total}")
    print(f"CA config  : {args.n_ca_rounds} rounds × "
          f"({args.n_phase1_steps} phase-1 + {args.n_phase2_steps} phase-2) steps/round")
    print(f"Total grad steps/outer iter: "
          f"{args.n_ca_rounds * (args.n_phase1_steps + args.n_phase2_steps)}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(run_dir, "taxa_used.txt"), "w") as f:
        for i, nm in enumerate(names):
            f.write(f"{i}\t{nm}\n")

    trained_tree, step_models, step_meta = run_training_pass(
        args, names, leaf_lik_all, device, run_dir
    )

    save_training_tree_json(trained_tree, names,
                            os.path.join(run_dir, "trained_tree_edges.json"))
    save_nexus_trees([trained_tree], names,
                     os.path.join(run_dir, "trained_tree.nex"))

    with open(os.path.join(run_dir, "step_meta.json"), "w") as f:
        json.dump(step_meta, f, indent=2)

    # ── Reload from disk (mirrors production use) ─────────────────────────────
    step_nets = preload_step_nets_ca(step_models, args.hidden_dim, device)
    set_seed(args.seed + 1000)

    print(f"\nSampling {args.n_tree_samples} trees from trained posterior...")
    sampled_trees = []
    for _ in tqdm(range(args.n_tree_samples), desc="Sample trees",
                  leave=True, dynamic_ncols=True):
        sampled_trees.append(
            sample_tree_from_trained_steps(args, names, leaf_lik_all, step_nets, device)
        )
    save_nexus_trees(sampled_trees, names, os.path.join(run_dir, "sampled_trees.nex"))

    # ── MLL estimation ────────────────────────────────────────────────────────
    print("\nEstimating Marginal Log-Likelihood (MLL)...")
    mll_results = estimate_mll(
        args=args, names=names, leaf_lik_all=leaf_lik_all,
        step_nets=step_nets, device=device, k_mll=args.k_mll,
    )
    with open(os.path.join(run_dir, "mll_results.json"), "w") as f:
        json.dump(mll_results, f, indent=2)
    print(f"  MLL results saved to {run_dir}/mll_results.json")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    save_sampled_tree_plots(
        sampled_trees=sampled_trees, names=names, run_dir=run_dir,
        leaf_lik_all=leaf_lik_all, device=device, args=args, step_nets=step_nets,
    )
    print("\nDone.")


if __name__ == "__main__":
    main(build_args().parse_args())