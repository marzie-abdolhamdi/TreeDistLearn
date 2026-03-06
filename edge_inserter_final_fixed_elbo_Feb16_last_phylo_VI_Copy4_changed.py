"""
Phylogenetic Variational Inference via Sequential Taxon Insertion
with Per-Edge Coordinate Ascent VI (CA-VI).

Key changes:
  q(rho, b | e) = q(rho | e)  ×  q(b | rho, e)
  
  Two separate network heads:
    RhoNet(cavities_e)        → (mu_r_e, log_sigma_r_e)
    BNet(cavities_e, rho_k)   → (mu_b_e^k, log_sigma_b_e^k)   per MC sample k

  Two-phase coordinate ascent per outer training step:
    Phase 1 – optimise RhoNet (BNet frozen, rho detached from BNet grad path)
    Phase 2 – fix rho, optimise BNet
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

EPS              = 1e-12
PI_DNA           = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
EDGE_FEATURE_DIM = 12
PRIOR_LAM_B      = 10.0   # Exp(10) prior on branch lengths → prior mean = 0.1 sub/site


# ══════════════════════════════════════════════════════════════════════════════
# Utilities & Tree Data Structure
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
            if not line: continue
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
    
    L = len(seqs[0])
    for i, s in enumerate(seqs):
        if len(s) != L:
            raise ValueError(f"Record {i} length {len(s)} != {L}")
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

@dataclass
class UnrootedTree:
    adj:      Dict[int, List[int]]
    edge_len: Dict[frozenset, float]

    def neighbors(self, node: int) -> List[int]:
        return self.adj.get(node, [])

    def length(self, u: int, v: int) -> float:
        return self.edge_len[frozenset((u, v))]

    def edges(self) -> List[Tuple[int, int]]:
        return sorted([(u, v) for u, v in [sorted(tuple(key)) for key in self.edge_len]])

    def add_edge(self, u: int, v: int, t: float) -> None:
        self.adj.setdefault(u, []); self.adj.setdefault(v, [])
        if v not in self.adj[u]: self.adj[u].append(v)
        if u not in self.adj[v]: self.adj[v].append(u)
        self.edge_len[frozenset((u, v))] = float(t)

    def remove_edge(self, u: int, v: int) -> float:
        t = self.edge_len.pop(frozenset((u, v)))
        if u in self.adj: self.adj[u] = [x for x in self.adj[u] if x != v]
        if v in self.adj: self.adj[v] = [x for x in self.adj[v] if x != u]
        return t

    def copy(self) -> "UnrootedTree":
        return UnrootedTree(
            adj={k: list(v) for k, v in self.adj.items()},
            edge_len={k: float(v) for k, v in self.edge_len.items()},
        )

class InternalNodeAllocator:
    def __init__(self, start: int): self.next_id = start
    def alloc(self) -> int:
        out = self.next_id
        self.next_id += 1
        return out

def build_t3_star(e0: float, e1: float, e2: float, center_idx: int) -> UnrootedTree:
    adj = {0: [center_idx], 1: [center_idx], 2: [center_idx], center_idx: [0, 1, 2]}
    edge_len = {
        frozenset((0, center_idx)): e0,
        frozenset((1, center_idx)): e1,
        frozenset((2, center_idx)): e2,
    }
    return UnrootedTree(adj=adj, edge_len=edge_len)

def compute_t3_branch_lengths(seq1: str, seq2: str, seq3: str) -> Tuple[float, float, float]:
    def p_dist(sA: str, sB: str) -> float:
        mm, cmp = 0, 0
        valid = set("ACGT")
        for a, b in zip(sA, sB):
            if a in valid and b in valid:
                cmp += 1
                if a != b: mm += 1
        return mm / cmp if cmp > 0 else 0.75

    def jc_dist(p: float) -> float:
        p = min(p, 0.7499)
        return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)

    d12 = jc_dist(p_dist(seq1, seq2)); d13 = jc_dist(p_dist(seq1, seq3)); d23 = jc_dist(p_dist(seq2, seq3))
    return max(1e-4, (d12 + d13 - d23) / 2.0), max(1e-4, (d12 + d23 - d13) / 2.0), max(1e-4, (d13 + d23 - d12) / 2.0)

def insert_taxon_on_edge(
    tree: UnrootedTree, edge: Tuple[int, int], new_taxon: int, rho: float, b_len: float, allocator: InternalNodeAllocator
) -> int:
    u, v = edge
    t_uv = tree.remove_edge(u, v)
    w = allocator.alloc()
    tree.add_edge(u, w, max(1e-8, rho * t_uv))
    tree.add_edge(w, v, max(1e-8, (1 - rho) * t_uv))
    tree.add_edge(w, new_taxon, max(1e-8, b_len))
    return w


# ══════════════════════════════════════════════════════════════════════════════
# Felsenstein Likelihood & Message Passing
# ══════════════════════════════════════════════════════════════════════════════

def jc_transition(t: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    e = torch.exp(-4.0 * rate * t / 3.0)
    same, diff = 0.25 + 0.75 * e, 0.25 - 0.25 * e
    p = torch.ones((4, 4), dtype=t.dtype, device=t.device) * diff
    idx = torch.arange(4, device=t.device)
    p[idx, idx] = same
    return p

def jc_transition_batched(t: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    e = torch.exp(-4.0 * rate * t / 3.0)
    same, diff = 0.25 + 0.75 * e, 0.25 - 0.25 * e
    p = diff.view(-1, 1, 1).expand(-1, 4, 4).clone()
    idx = torch.arange(4, device=t.device)
    p[:, idx, idx] = same.unsqueeze(1).expand(-1, 4)
    return p

def compute_messages_and_cavities(tree: UnrootedTree, leaf_lik: Dict[int, torch.Tensor]):
    device = next(iter(leaf_lik.values())).device
    n_sites = next(iter(leaf_lik.values())).shape[0]
    msg_cache, cav_cache = {}, {}

    def cavity(a: int, b: int):
        if (a, b) in cav_cache: return cav_cache[(a, b)]
        psi = leaf_lik[a].clone() if a in leaf_lik else torch.ones(n_sites, 4, dtype=torch.float32, device=device)
        acc_log_scale = torch.zeros(n_sites, dtype=torch.float32, device=device)
        for k in tree.neighbors(a):
            if k == b: continue
            m_k_a, m_ls = message(k, a)
            psi = psi * m_k_a
            acc_log_scale += m_ls
            scaler = psi.max(dim=-1, keepdim=True)[0].clamp_min(1e-30)
            psi = psi / scaler
            acc_log_scale += torch.log(scaler).squeeze(-1)
        cav_cache[(a, b)] = (psi, acc_log_scale)
        return psi, acc_log_scale

    def message(a: int, b: int):
        if (a, b) in msg_cache: return msg_cache[(a, b)]
        psi, acc_log_scale = cavity(a, b)
        t_ab = torch.tensor(tree.length(a, b), dtype=torch.float32, device=device)
        m = psi @ jc_transition(t_ab).T
        scaler = m.max(dim=-1, keepdim=True)[0].clamp_min(1e-30)
        m = m / scaler
        acc_log_scale += torch.log(scaler).squeeze(-1)
        msg_cache[(a, b)] = (m, acc_log_scale)
        return m, acc_log_scale

    for u, v in tree.edges():
        message(u, v); message(v, u)
    return msg_cache, cav_cache

def insertion_loglik_from_edge_batched(
    tree: UnrootedTree, edge: Tuple[int, int], rho: torch.Tensor, b_len: torch.Tensor,
    new_leaf_lik: torch.Tensor, cav_cache: Dict, pi: torch.Tensor, rate: float
) -> torch.Tensor:
    u, v = edge
    t_uv = torch.tensor(tree.length(u, v), dtype=torch.float32, device=rho.device)
    t_uw, t_wv = rho * t_uv, (1.0 - rho) * t_uv

    psi_u, log_scale_u = cav_cache[(u, v)]
    psi_v, log_scale_v = cav_cache[(v, u)]

    psi_u, psi_v = psi_u.to(rho.device), psi_v.to(rho.device)
    log_scale_u, log_scale_v = log_scale_u.to(rho.device), log_scale_v.to(rho.device)
    new_leaf_lik, pi = new_leaf_lik.to(rho.device), pi.to(rho.device)

    p_uw = jc_transition_batched(t_uw, rate=rate)
    p_wv = jc_transition_batched(t_wv, rate=rate)
    p_yw = jc_transition_batched(b_len, rate=rate)

    m_u = torch.matmul(psi_u.unsqueeze(0), p_uw.transpose(1, 2))
    m_v = torch.matmul(psi_v.unsqueeze(0), p_wv.transpose(1, 2))
    m_y = torch.matmul(new_leaf_lik.unsqueeze(0), p_yw.transpose(1, 2))

    site_like = (m_u * m_v * m_y * pi.view(1, 1, 4)).sum(dim=-1).clamp_min(1e-30)
    log_sl = torch.log(site_like) + log_scale_u.unsqueeze(0) + log_scale_v.unsqueeze(0)
    return log_sl.sum(dim=-1)

def build_edge_features(edges, cav_cache, new_leaf_lik, device, n_nodes, tree, jc_rate: float) -> torch.Tensor:
    new_leaf_lik = new_leaf_lik.to(device)
    pi_dna = PI_DNA.to(device)
    feats = []
    
    for u, v in edges:
        cav_uv, _ = cav_cache[(u, v)]
        cav_vu, _ = cav_cache[(v, u)]
        
        t_uv = torch.tensor(tree.length(u, v), dtype=torch.float32, device=device)
        P_uv = jc_transition(t_uv, rate=jc_rate)
        
        msg_v_to_u = torch.matmul(cav_vu.to(device), P_uv.T)
        msg_u_to_v = torch.matmul(cav_uv.to(device), P_uv.T)
        
        joint_u = pi_dna * cav_uv.to(device) * msg_v_to_u
        joint_v = pi_dna * cav_vu.to(device) * msg_u_to_v
        
        anc_u = joint_u / (joint_u.sum(dim=-1, keepdim=True) + EPS)
        anc_v = joint_v / (joint_v.sum(dim=-1, keepdim=True) + EPS)
        y_n = new_leaf_lik / (new_leaf_lik.sum(dim=-1, keepdim=True) + EPS)
        
        feats.append(torch.cat([anc_u, anc_v, y_n], dim=-1))
        
    return torch.stack(feats, dim=0)   # (E, L, 12)


# ══════════════════════════════════════════════════════════════════════════════
# Neural Network Architecture
# ══════════════════════════════════════════════════════════════════════════════

class RhoNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1), nn.ReLU(),
        )
        self.rho_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor):
        xt = x.transpose(1, 2)                         # (E, C, L)
        y_n = xt[:, 8:12, :]                           
        mask = (y_n.max(dim=1)[0] > 0.26).float()      # (E, L)
        h_seq = self.cnn(xt)                           # (E, hidden, L)
        
        # Masked average pooling
        mask = mask.unsqueeze(1)                       
        valid = mask.sum(dim=2).clamp_min(1.0)         
        h = (h_seq * mask).sum(dim=2) / valid          # (E, hidden)
        
        out = self.rho_head(h)                         
        mu_r = out[:, 0]
        log_sigma_r = out[:, 1].clamp(-5.0, 3.0)
        return mu_r, log_sigma_r, h

class BNet(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.rho_embed = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
        self.b_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, h: torch.Tensor, rho: torch.Tensor):
        K, E = rho.shape
        h_exp = h.unsqueeze(0).expand(K, E, -1)              
        rho_emb = self.rho_embed(rho.unsqueeze(-1))          
        combined = torch.cat([h_exp, rho_emb], dim=-1)       
        
        out = self.b_head(combined)                          
        # Shift initialization to ~0.05 to prevent catastrophic likelihood on real data
        mu_b = out[..., 0] - 3.0                             
        log_sigma_b = out[..., 1].clamp(-5.0, 0.5)
        return mu_b, log_sigma_b


# ══════════════════════════════════════════════════════════════════════════════
# Mathematical Divergences
# ══════════════════════════════════════════════════════════════════════════════

def exponential_kl_branch(mu: torch.Tensor, log_sigma: torch.Tensor, lam: float = PRIOR_LAM_B) -> torch.Tensor:
    sigma2 = torch.exp(2.0 * log_sigma)
    e_b    = torch.exp(mu + 0.5 * sigma2)
    return -log_sigma - mu - 0.5*(1.0 + math.log(2.0 * math.pi)) - math.log(lam) + lam * e_b

def gaussian_kl_standard(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    sigma2 = torch.exp(2.0 * log_sigma)
    return 0.5 * (mu.pow(2) + sigma2 - 1.0 - 2.0 * log_sigma)

def log_lognormal_density(b: float, mu: float, log_sigma: float) -> float:
    sigma = math.exp(log_sigma)
    log_b = math.log(max(b, 1e-30))
    return -log_b - log_sigma - 0.5 * math.log(2.0 * math.pi) - 0.5 * ((log_b - mu) / sigma) ** 2

def log_logistic_normal_density(rho: float, mu_r: float, log_sigma_r: float) -> float:
    sigma_r = math.exp(log_sigma_r)
    rho_c   = min(max(rho, 1e-6), 1.0 - 1e-6)
    z       = math.log(rho_c / (1.0 - rho_c))
    log_normal = -log_sigma_r - 0.5 * math.log(2.0 * math.pi) - 0.5 * ((z - mu_r) / sigma_r) ** 2
    return log_normal - math.log(rho_c) - math.log(1.0 - rho_c)


# ══════════════════════════════════════════════════════════════════════════════
# Core CA-VI Forward Passes
# ══════════════════════════════════════════════════════════════════════════════

def _stack_cavities(edges, tree, cav_cache, device):
    psi_u_l, psi_v_l, ls_u_l, ls_v_l, t_l = [], [], [], [], []
    for u, v in edges:
        psi_u, ls_u = cav_cache[(u, v)]; psi_v, ls_v = cav_cache[(v, u)]
        psi_u_l.append(psi_u.to(device)); psi_v_l.append(psi_v.to(device))
        ls_u_l.append(ls_u.to(device));   ls_v_l.append(ls_v.to(device))
        t_l.append(tree.length(u, v))
    return (torch.stack(psi_u_l), torch.stack(psi_v_l), torch.stack(ls_u_l), 
            torch.stack(ls_v_l), torch.tensor(t_l, dtype=torch.float32, device=device))

def compute_loglik_all_edges(psi_u, psi_v, ls_u, ls_v, t_uv_all, rho, b, new_leaf_lik, pi, jc_rate):
    K, E = rho.shape[0], rho.shape[1]
    t_uv = t_uv_all.unsqueeze(0)
    t_uw, t_wv = rho * t_uv, (1.0 - rho) * t_uv

    def _jc_ke(t_ke):
        return jc_transition_batched(t_ke.reshape(-1), rate=jc_rate).reshape(K, E, 4, 4)

    P_uw, P_wv, P_yw = _jc_ke(t_uw), _jc_ke(t_wv), _jc_ke(b)

    m_u = torch.einsum("elc,kedc->keld", psi_u, P_uw)
    m_v = torch.einsum("elc,kedc->keld", psi_v, P_wv)
    m_y = torch.einsum("lc,kedc->keld",  new_leaf_lik, P_yw)

    site_like = (m_u * m_v * m_y * pi.view(1, 1, 1, 4)).sum(-1).clamp_min(1e-30)
    log_sl = torch.log(site_like) + ls_u.unsqueeze(0) + ls_v.unsqueeze(0)
    return log_sl.sum(-1).mean(0)   # (E,)

def compute_loglik_per_edge(tree, edges, rho, b, new_leaf_lik, cav_cache, pi, jc_rate):
    loglik_list = []
    for i, e in enumerate(edges):
        ll_k = insertion_loglik_from_edge_batched(
            tree=tree, edge=e, rho=rho[:, i], b_len=b[:, i], new_leaf_lik=new_leaf_lik,
            cav_cache=cav_cache, pi=pi, rate=jc_rate,
        )
        loglik_list.append(ll_k.mean())
    return torch.stack(loglik_list)

def forward_ca(tree, edges, edge_feats, rho_net, b_net, new_leaf_lik, cav_cache, pi, k_samples, jc_rate, device, phase, rho_fixed=None, h_fixed=None, stacked_cavs=None):
    n_edges = len(edges)
    n_sites = new_leaf_lik.shape[0]

    if phase == 1:
        mu_r, log_sigma_r, h = rho_net(edge_feats)
        kl_rho = gaussian_kl_standard(mu_r, log_sigma_r)
        
        std_r = torch.exp(log_sigma_r)
        eps_r = torch.randn(k_samples, n_edges, device=device)
        rho   = torch.sigmoid(mu_r.unsqueeze(0) + std_r.unsqueeze(0) * eps_r)
        
        mu_b, log_sigma_b = b_net(h.detach(), rho)
    else:
        rho = rho_fixed; h_for_b = h_fixed
        mu_b, log_sigma_b = b_net(h_for_b, rho)
        with torch.no_grad():
            mu_r, log_sigma_r, _ = rho_net(edge_feats)
            kl_rho = gaussian_kl_standard(mu_r, log_sigma_r)
        h = h_fixed

    std_b = torch.exp(log_sigma_b)
    eps_b = torch.randn(k_samples, n_edges, device=device)
    b     = torch.exp(mu_b + std_b * eps_b).clamp(min=1e-8, max=2.0)

    kl_b_mc = exponential_kl_branch(mu_b, log_sigma_b).mean(dim=0)
    if stacked_cavs is None: stacked_cavs = _stack_cavities(edges, tree, cav_cache, device)
    
    loglik_mc = compute_loglik_all_edges(*stacked_cavs, rho, b, new_leaf_lik, pi, jc_rate)

    # KL Scaling ensures prior applies correctly vs massive sequence log-likelihoods
    if phase == 1: elbo_e = loglik_mc - (kl_rho * n_sites) - (kl_b_mc * n_sites)
    else:          elbo_e = loglik_mc - (kl_rho.detach() * n_sites) - (kl_b_mc * n_sites)

    return {"elbo_e": elbo_e, "loglik_mc": loglik_mc, "kl_rho": kl_rho, "kl_b_mc": kl_b_mc, "rho": rho, "b": b, "mu_r": mu_r, "log_sigma_r": log_sigma_r, "mu_b": mu_b, "log_sigma_b": log_sigma_b, "h": h}


# ══════════════════════════════════════════════════════════════════════════════
# Inference / Sampling
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_edges_ca(tree, new_leaf_lik, leaf_lik_all, rho_net, b_net, device, k_samples, jc_rate, temperature=1.0):
    leaf_lik_current = {k: v for k, v in leaf_lik_all.items() if k in tree.adj and len(tree.adj[k]) == 1}
    with torch.no_grad():
        _, cav_cache   = compute_messages_and_cavities(tree, leaf_lik_current)
        edges          = tree.edges()
        edge_feats_raw = build_edge_features(edges, cav_cache, new_leaf_lik, device, len(tree.adj), tree, jc_rate)

    edge_feats, new_leaf_lik, pi_dna = edge_feats_raw.detach(), new_leaf_lik.to(device), PI_DNA.to(device)
    n_edges, n_sites = len(edges), new_leaf_lik.shape[0]

    mu_r, log_sigma_r, h = rho_net(edge_feats)
    kl_rho = gaussian_kl_standard(mu_r, log_sigma_r)

    rho = torch.sigmoid(mu_r.unsqueeze(0) + torch.exp(log_sigma_r).unsqueeze(0) * torch.randn(k_samples, n_edges, device=device))
    mu_b, log_sigma_b = b_net(h.detach(), rho.detach())
    b = torch.exp(mu_b + torch.exp(log_sigma_b) * torch.randn(k_samples, n_edges, device=device)).clamp(min=1e-8, max=2.0)

    kl_b_mc = exponential_kl_branch(mu_b, log_sigma_b).mean(0)
    loglik_mc = compute_loglik_all_edges(*_stack_cavities(edges, tree, cav_cache, device), rho, b, new_leaf_lik, pi_dna, jc_rate)

    elbo_e = loglik_mc - (kl_rho * n_sites) - (kl_b_mc * n_sites)
    q_e    = torch.softmax(loglik_mc / temperature, dim=0)

    return (edges, q_e, elbo_e, mu_r, log_sigma_r, mu_b, log_sigma_b, h, rho.detach().cpu().numpy(), b.detach().cpu().numpy(), kl_rho, kl_b_mc, loglik_mc, cav_cache, edge_feats)


def choose_action_ca(edges, q_e, rho_samples, b_samples, mu_r, h, b_net, device, sample_continuous=True):
    q_e = torch.nan_to_num(q_e, nan=0.0)
    q_e = q_e / (q_e.sum() + EPS)
    chosen_idx = torch.multinomial(q_e, 1).item()

    if sample_continuous:
        chosen_rho, chosen_b = float(rho_samples[0, chosen_idx]), float(b_samples[0, chosen_idx])
    else:
        chosen_rho = torch.sigmoid(mu_r[chosen_idx]).item()
        with torch.no_grad():
            mu_b_i, _ = b_net(h[chosen_idx].detach().unsqueeze(0), torch.tensor([[chosen_rho]], dtype=torch.float32, device=device))
        chosen_b = math.exp(min(mu_b_i.item(), 2.0))

    return edges[chosen_idx], chosen_rho, chosen_b


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def node_label(node_id: int, taxon_names: List[str]) -> str:
    return taxon_names[node_id] if node_id < len(taxon_names) else f"i{node_id}"

def _ema(values: List[float], alpha: float = 0.05) -> np.ndarray:
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

def save_step_plots(
    out_dir: str, edge_labels: List[str], elbo_hist: List[float], q_final: np.ndarray,
    elbo_final: np.ndarray, loglik_final: np.ndarray, rho_samples_final: np.ndarray,
    b_samples_final: np.ndarray, entropy_hist: List[float], max_qe_hist: List[float],
    kl_b_hist: List[float], kl_r_hist: List[float], loglik_hist: List[float],
    grad_norm_rho_hist: List[float], grad_norm_b_hist: List[float], temp_start: float,
    temp_end: float, max_steps: int, min_steps: int, mu_r_top_hist: List[float],
    sigma_r_top_hist: List[float], mu_b_top_hist: List[float], sigma_b_top_hist: List[float],
    mu_b_final_mean: np.ndarray, log_sigma_b_final_mean: np.ndarray, taxon_idx: int,
    n_sites: int, ema_hist: Optional[List[float]] = None, stopped_early: bool = False,
    stop_step: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    steps = np.arange(1, len(elbo_hist) + 1)
    ema = np.array(ema_hist) if ema_hist is not None else _ema(elbo_hist)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, elbo_hist, alpha=0.35, label="ELBO (raw)")
    plt.plot(steps, ema, linewidth=2, label=f"ELBO (EMA α={0.1:.2f})")
    if stopped_early and stop_step is not None:
        plt.axvline(x=stop_step, color="red", linestyle="--", linewidth=1.5, label=f"Early stop (step {stop_step})")
    plt.xlabel("Outer iteration"); plt.ylabel("ELBO / n_sites")
    plt.title("Convergence Monitoring"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbo_convergence.png"), dpi=150); plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, q_final, color='mediumseagreen')
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge"); plt.ylabel("q(e) = softmax(Log-Likelihood / T)")
    plt.title("Edge Posterior (Pure Felsenstein Likelihood)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_posterior_qe.png"), dpi=150); plt.close()

    rel_loglik = loglik_final - np.max(loglik_final)
    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, rel_loglik, color='purple', alpha=0.8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge"); plt.ylabel("LogLik - max(LogLik)")
    plt.title("Relative Per-Edge Felsenstein Likelihood Scores"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_loglik_scores.png"), dpi=150); plt.close()
    
    rel_elbo = elbo_final - np.max(elbo_final)
    plt.figure(figsize=(10, 5))
    plt.bar(edge_labels, rel_elbo, color='steelblue', alpha=0.6)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Edge"); plt.ylabel("ELBO - max(ELBO)")
    plt.title("Relative Per-Edge ELBO (Neural Network Loss)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_elbo_scores.png"), dpi=150); plt.close()

    top_k = min(5, len(edge_labels))
    top_idxs = np.argsort(q_final)[-top_k:]
    for arr, fname, xlabel in [(rho_samples_final, "rho_posterior.png", "rho"), (b_samples_final, "b_posterior.png", "b")]:
        plt.figure(figsize=(8, 4))
        for i in top_idxs:
            plt.hist(arr[:, i], bins=30, alpha=0.30, density=True, label=f"{xlabel}|{edge_labels[i]}")
        plt.xlabel(xlabel); plt.ylabel("density")
        plt.title(f"Posterior for {xlabel} (Top-{top_k} edges)")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, [x / n_sites for x in loglik_hist], color='blue', alpha=0.7)
    axes[0].set_title("E[log p(data)] / n_sites"); axes[0].set_ylabel("Nats / site"); axes[0].grid(alpha=0.3)
    axes[1].plot(steps, kl_b_hist, color='orange', label="KL_b (branch)")
    axes[1].plot(steps, kl_r_hist, color='red', label="KL_r (rho)")
    axes[1].set_title("KL Terms"); axes[1].set_ylabel("KL (nats)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(steps, elbo_hist, color='green', alpha=0.7)
    axes[2].set_title("ELBO / n_sites"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Outer iteration")
    plt.suptitle("ELBO Decomposition"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbo_decomposition.png"), dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hist, title in [(axes[0], grad_norm_rho_hist, "RhoNet grad norm (phase 1)"), (axes[1], grad_norm_b_hist, "BNet grad norm (phase 2)")]:
        ax.plot(steps, hist, alpha=0.6, color='red', linewidth=1)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='clip=1.0')
        ax.set_title(f"{title}\n({np.mean(np.array(hist) > 1.0) * 100:.1f}% clipped)")
        ax.set_xlabel("Outer iteration"); ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradient_norms.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, entropy_hist, color='green', linewidth=2)
    plt.xlabel("Outer iteration"); plt.ylabel("Shannon Entropy H(q)")
    plt.title("Edge Distribution Entropy"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_trajectory.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, max_qe_hist, color='purple', linewidth=2)
    plt.xlabel("Outer iteration"); plt.ylabel("Max q(e)")
    plt.title("Confidence in Top Edge"); plt.ylim(0, 1.05); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_trajectory.png"), dpi=150); plt.close()

    temps = [temp_start * (temp_end / temp_start) ** min(1.0, i / max(1, min_steps - 1)) for i in range(len(steps))]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, entropy_hist, color='green', linewidth=2, label='H(q)')
    ax1.set_xlabel("Outer iteration"); ax1.set_ylabel("Entropy", color='green')
    ax2 = ax1.twinx()
    ax2.plot(steps, temps, color='orange', linestyle='--', linewidth=1.5, label='Temp')
    ax2.set_ylabel("Temperature", color='orange')
    ax1.legend(ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1], loc='upper right')
    plt.title("Entropy vs Temperature"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_vs_temperature.png"), dpi=150); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, mu_r_top_hist, color='darkorange'); axes[0, 0].axhline(0.0, color='red', linestyle='--', label='Prior mean'); axes[0, 0].set_title("mu_r of top edge"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(steps, sigma_r_top_hist, color='darkorange'); axes[0, 1].axhline(1.0, color='red', linestyle='--', label='Prior std'); axes[0, 1].set_title("sigma_r of top edge"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(steps, mu_b_top_hist, color='steelblue'); axes[1, 0].axhline(math.log(1.0 / PRIOR_LAM_B), color='red', linestyle='--', label=f'Prior mode'); axes[1, 0].set_title("mu_b of top edge"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(steps, sigma_b_top_hist, color='steelblue'); axes[1, 1].axhline(1.0, color='red', linestyle='--', label='Prior std'); axes[1, 1].set_title("sigma_b of top edge"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    for ax in axes.flat: ax.set_xlabel("Outer iteration")
    plt.suptitle(f"Variational Parameter Trajectories — Taxon {taxon_idx}"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variational_params_trajectory.png"), dpi=150); plt.close()

    b_means, b_stds = np.mean(b_samples_final, axis=0), np.std(b_samples_final, axis=0)
    x = np.arange(len(edge_labels))
    plt.figure(figsize=(max(10, len(edge_labels) * 0.4), 5))
    plt.bar(x, b_means, yerr=b_stds, capsize=3, color='steelblue', alpha=0.7, label='mean±std')
    plt.axhline(0.05, color='green', linestyle='--', label='Short (0.05)'); plt.axhline(0.3, color='orange', linestyle='--', label='Long (0.3)'); plt.axhline(1.0 / PRIOR_LAM_B, color='red', linestyle=':', label=f'Prior mean')
    plt.xticks(x, edge_labels, rotation=45, ha='right', fontsize=7)
    plt.ylabel("Branch length (sub/site)"); plt.title(f"Inferred Branch Lengths — Taxon {taxon_idx}")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "branch_lengths_per_edge.png"), dpi=150); plt.close()

def save_crosstep_plots(run_dir: str, step_meta: List[dict], names: List[str]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    tidxs = [m["taxon_idx"] for m in step_meta]; tlabels = [names[i] for m in step_meta for i in [m["taxon_idx"]]]
    x, ts = np.arange(len(tidxs)), step_meta
    def _get(key): return [m["train_summary"][key] for m in ts]

    final_elbos, final_ent, final_maxqe, final_kl_b, final_kl_r, final_loglik, n_edges_list, actual_steps = _get("final_scaled_elbo"), _get("final_entropy"), _get("final_max_qe"), _get("final_kl_b"), _get("final_kl_r"), _get("final_loglik"), _get("n_edges"), _get("iterations")
    stopped_flags = [m["train_summary"].get("stopped_early", False) for m in ts]

    fig, ax1 = plt.subplots(figsize=(max(8, len(tidxs) * 0.5), 4))
    ax1.plot(x, final_elbos, marker='o', color='steelblue', linewidth=2)
    for xi, yi, st, ns in zip(x, final_elbos, stopped_flags, actual_steps):
        ax1.annotate(f"{yi:.2f}\n({ns}st)", (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=6, color='green' if st else 'gray')
    ax2 = ax1.twinx()
    ax2.bar(x, actual_steps, color=['#2ca02c' if s else '#aec7e8' for s in stopped_flags], alpha=0.3, width=0.4, label='Steps used')
    ax2.set_ylabel("Steps used", color='gray'); ax1.set_xticks(x); ax1.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel("Taxon"); ax1.set_ylabel("Final ELBO / n_sites"); ax1.set_title("ELBO at Convergence + Steps Used per Taxon"); ax1.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_elbo.png"), dpi=150); plt.close()

    fig, ax1 = plt.subplots(figsize=(max(8, len(tidxs) * 0.5), 4))
    ax1.plot(x, final_ent, marker='o', color='green', linewidth=2, label='H(q)')
    ax1.set_ylabel("Shannon Entropy", color='green')
    ax2 = ax1.twinx()
    ax2.plot(x, final_maxqe, marker='s', color='purple', linewidth=2, linestyle='--', label='max q(e)')
    ax2.set_ylabel("Max q(e)", color='purple'); ax2.set_ylim(0, 1.05); ax1.set_xticks(x); ax1.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=8); ax1.set_xlabel("Taxon")
    ax1.legend(ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1], loc='upper left', fontsize=8)
    plt.title("Posterior Confidence per Taxon"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_confidence.png"), dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(tidxs) * 1.2), 4))
    axes[0].bar(x, final_loglik, color='steelblue', alpha=0.8); axes[0].set_title("Final E[log p(data)] per Taxon"); axes[0].set_ylabel("Nats")
    axes[1].bar(x, final_kl_b, color='orange', alpha=0.8, label='KL_b'); axes[1].bar(x, final_kl_r, color='red', alpha=0.8, label='KL_r', bottom=final_kl_b); axes[1].set_title("Final KL per Taxon"); axes[1].legend()
    axes[2].plot(x, n_edges_list, marker='o', color='gray', linewidth=2); axes[2].set_title("Edges at Each Step")
    for ax in axes: ax.set_xticks(x); ax.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle("Cross-Taxon Summary"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "crossstep_decomposition.png"), dpi=150); plt.close()
    print(f"\n[Category 3 plots saved to {run_dir}]")


# ══════════════════════════════════════════════════════════════════════════════
# Nexus / Newick I/O
# ══════════════════════════════════════════════════════════════════════════════

def escape_nexus_name(name: str) -> str:
    return f"'{name}'" if any(c in name for c in " \t\n(),:;[]=") else name

def to_newick(tree: UnrootedTree, taxon_names: List[str], label_mapper: Optional[Callable[[int], str]] = None) -> str:
    start_node = next((u for u in tree.adj if u >= len(taxon_names)), None) or next(iter(tree.adj), None)
    if start_node is None: return "();"
    visited = set()

    def build_sub(u: int) -> str:
        visited.add(u)
        children = [v for v in tree.neighbors(u) if v not in visited]
        if not children: return label_mapper(u) if label_mapper else taxon_names[u]
        return "(" + ",".join([f"{build_sub(v)}:{tree.length(u,v):.6f}" for v in children]) + ")"

    visited.add(start_node)
    return "(" + ",".join([f"{build_sub(v)}:{tree.length(start_node,v):.6f}" for v in tree.neighbors(start_node)]) + ");"

def to_newick_topology_only(tree: UnrootedTree, taxon_names: List[str]) -> str:
    if not tree.adj: return "();"
    start_node = 0
    root_internal = tree.neighbors(start_node)[0]
    visited = {start_node, root_internal}

    def build_sub(u: int) -> str:
        visited.add(u)
        children = [v for v in tree.neighbors(u) if v not in visited]
        if not children: return taxon_names[u]
        return "(" + ",".join(sorted(build_sub(v) for v in children)) + ")"

    children = [v for v in tree.neighbors(root_internal) if v not in visited]
    return "(" + ",".join(sorted([taxon_names[start_node]] + sorted(build_sub(v) for v in children))) + ");"

def save_nexus_trees(trees: List[UnrootedTree], taxon_names: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n\nBEGIN TAXA;\n")
        f.write(f"\tDIMENSIONS NTAX={len(taxon_names)};\n\tTAXLABELS\n")
        for name in taxon_names: f.write(f"\t\t{escape_nexus_name(name)}\n")
        f.write("\t;\nEND;\n\nBEGIN TREES;\n\tTRANSLATE\n")
        for i, name in enumerate(taxon_names): f.write(f"\t\t{i+1} {escape_nexus_name(name)}{',' if i < len(taxon_names) - 1 else ''}\n")
        f.write("\t;\n")
        for idx, t in enumerate(trees): f.write(f"\tTREE tree_{idx+1} = {to_newick(t, taxon_names, label_mapper=lambda i: str(i + 1))}\n")
        f.write("END;\n")

def save_training_tree_json(tree: UnrootedTree, names: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"u_id": int(u), "v_id": int(v), "u_label": node_label(u, names), "v_label": node_label(v, names), "length": float(tree.length(u, v))} for u, v in tree.edges()], f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_single_insertion_step(taxon_idx, step_nets, new_leaf_lik, leaf_lik_all, taxon_names, device, args, step_out_dir=None):
    n_use, n_sites = len(taxon_names), new_leaf_lik.shape[0]
    new_leaf_lik, pi_dna = new_leaf_lik.to(device), PI_DNA.to(device)

    rho_net = RhoNet(in_channels=EDGE_FEATURE_DIM, hidden=args.hidden_dim).to(device)
    b_net   = BNet(hidden=args.hidden_dim).to(device)
    optimizer_rho = torch.optim.Adam(rho_net.parameters(), lr=args.lr)
    optimizer_b   = torch.optim.Adam(b_net.parameters(),   lr=args.lr)

    elbo_hist, entropy_hist, max_qe_hist, kl_b_hist, kl_r_hist, loglik_hist = [], [], [], [], [], []
    grad_norm_rho_hist, grad_norm_b_hist = [], []
    mu_r_top_hist, sigma_r_top_hist, mu_b_top_hist, sigma_b_top_hist = [], [], [], []
    
    ema_val, ema_at_patience_ago, ema_hist_full = None, None, []
    stopped_early, stop_reason, early_stop_step, final_tree = False, "max_steps reached", None, None

    pbar = tqdm(total=args.max_steps, desc=f"Train Taxon {taxon_idx}", leave=False, dynamic_ncols=True)

    for step_iter in range(1, args.max_steps + 1):
        current_temp = args.temp_start * (args.temp_end / args.temp_start) ** min(1.0, (step_iter - 1) / max(1, args.min_steps - 1))

        # Sample Context Tree
        allocator = InternalNodeAllocator(start=n_use)
        tree = build_t3_star(args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=allocator.alloc())
        
        with torch.no_grad():
            for step in range(3, taxon_idx):
                rho_net_s, b_net_s = step_nets[step]
                (edges_s, q_e_s, _, mu_r_s, _, _, _, h_s, rho_np_s, b_np_s, _, _, _, _, _) = evaluate_edges_ca(
                    tree, leaf_lik_all[step], leaf_lik_all, rho_net_s, b_net_s, device, args.k_train, args.jc_rate, args.sample_temp
                )
                e_c, r_c, b_c = choose_action_ca(edges_s, q_e_s, rho_np_s, b_np_s, mu_r_s, h_s, b_net_s, device, True)
                insert_taxon_on_edge(tree, e_c, step, r_c, b_c, allocator)

        # Build Features for Training
        leaf_lik_current = {k: v for k, v in leaf_lik_all.items() if k in tree.adj and len(tree.adj[k]) == 1}
        with torch.no_grad():
            _, cav_cache = compute_messages_and_cavities(tree, leaf_lik_current)
            edges, n_edges = tree.edges(), len(tree.edges())
            edge_feats = build_edge_features(edges, cav_cache, new_leaf_lik, device, len(tree.adj), tree, args.jc_rate).detach()
            stacked_cavs = _stack_cavities(edges, tree, cav_cache, device)

        for ca_round in range(args.n_ca_rounds):
            # Phase 1: RhoNet
            for _ in range(args.n_phase1_steps):
                rho_net.train(); b_net.eval()
                res = forward_ca(tree, edges, edge_feats, rho_net, b_net, new_leaf_lik, cav_cache, pi_dna, args.k_train, args.jc_rate, device, 1, stacked_cavs=stacked_cavs)
                
                scaled_elbo = (res["elbo_e"] / current_temp).clamp(min=-50000.0, max=50000.0)
                tree_elbo = current_temp * torch.logsumexp(scaled_elbo, dim=0) - math.log(n_edges)
                loss = -tree_elbo / 1000.0  # Constant scaling prevents gap dilution

                optimizer_rho.zero_grad(); loss.backward()
                _gn_rho = sum(p.grad.norm().item()**2 for p in rho_net.parameters() if p.grad is not None)**0.5
                torch.nn.utils.clip_grad_norm_(rho_net.parameters(), 1.0); optimizer_rho.step()

            # Phase 2: BNet
            for _ in range(args.n_phase2_steps):
                with torch.no_grad():
                    mu_r_s, lsr_s, h_s = rho_net(edge_feats)
                    rho_f = torch.sigmoid(mu_r_s.unsqueeze(0) + torch.exp(lsr_s).unsqueeze(0) * torch.randn(args.k_train, n_edges, device=device))
                
                rho_net.eval(); b_net.train()
                res = forward_ca(tree, edges, edge_feats, rho_net, b_net, new_leaf_lik, cav_cache, pi_dna, args.k_train, args.jc_rate, device, 2, rho_fixed=rho_f, h_fixed=h_s.detach(), stacked_cavs=stacked_cavs)
                
                scaled_elbo = (res["elbo_e"] / current_temp).clamp(min=-50000.0, max=50000.0)
                tree_elbo = current_temp * torch.logsumexp(scaled_elbo, dim=0) - math.log(n_edges)
                loss = -tree_elbo / 1000.0

                optimizer_b.zero_grad(); loss.backward()
                _gn_b = sum(p.grad.norm().item()**2 for p in b_net.parameters() if p.grad is not None)**0.5
                torch.nn.utils.clip_grad_norm_(b_net.parameters(), 1.0); optimizer_b.step()

        # End of outer step evaluation
        rho_net.eval(); b_net.eval()
        with torch.no_grad():
            r_eval = forward_ca(tree, edges, edge_feats, rho_net, b_net, new_leaf_lik, cav_cache, pi_dna, args.k_train, args.jc_rate, device, 1, stacked_cavs=stacked_cavs)

        scaled_elbo = (r_eval["elbo_e"] / current_temp).clamp(min=-50000.0, max=50000.0)
        val = ((current_temp * torch.logsumexp(scaled_elbo, dim=0) - math.log(n_edges)) / n_sites).item()

        q_e_e = torch.softmax(r_eval["loglik_mc"] / current_temp, dim=0)
        top_idx = q_e_e.argmax().item()

        elbo_hist.append(val)
        entropy_hist.append(-torch.sum(q_e_e * torch.log(q_e_e + EPS)).item())
        max_qe_hist.append(q_e_e.max().item())
        kl_b_hist.append(r_eval["kl_b_mc"].mean().item())
        kl_r_hist.append(r_eval["kl_rho"].mean().item())
        loglik_hist.append(r_eval["loglik_mc"].mean().item())
        grad_norm_rho_hist.append(_gn_rho); grad_norm_b_hist.append(_gn_b)
        mu_r_top_hist.append(r_eval["mu_r"][top_idx].item()); sigma_r_top_hist.append(torch.exp(r_eval["log_sigma_r"][top_idx]).item())
        mu_b_top_hist.append(r_eval["mu_b"][:, top_idx].mean().item()); sigma_b_top_hist.append(torch.exp(r_eval["log_sigma_b"][:, top_idx]).mean().item())

        ema_val = val if ema_val is None else args.ema_alpha * val + (1 - args.ema_alpha) * ema_val
        ema_hist_full.append(ema_val)
        if len(ema_hist_full) > args.patience: ema_at_patience_ago = ema_hist_full[-args.patience - 1]

        pbar.update(1); pbar.set_postfix(elbo=f"{val:.4f}", ema=f"{ema_val:.4f}", T=f"{current_temp:.1f}")
        final_tree = tree

        if ema_at_patience_ago is not None and step_iter >= args.min_steps + args.patience:
            if (ema_val - ema_at_patience_ago) < args.conv_tol:
                stop_reason = f"converged at {step_iter} (EMA Δ < {args.conv_tol:.1e})"
                stopped_early, early_stop_step = True, step_iter
                break

    pbar.close()
    if step_out_dir and final_tree is not None:
        rho_net.eval(); b_net.eval()
        with torch.no_grad():
            (edges_f, q_e_f, elbo_e_f, mu_r_f, lsr_f, mu_b_f, lsb_f, h_f, rho_np_f, b_np_f, _, _, loglik_mc_f, _, _) = evaluate_edges_ca(
                final_tree, new_leaf_lik, leaf_lik_all, rho_net, b_net, device, args.k_eval, args.jc_rate, current_temp)
        edge_labels = [f"({node_label(u, taxon_names)}, {node_label(v, taxon_names)})" for u, v in edges_f]
        save_step_plots(step_out_dir, edge_labels, elbo_hist, q_e_f.detach().cpu().numpy(), elbo_e_f.detach().cpu().numpy(), loglik_mc_f.detach().cpu().numpy(), rho_np_f, b_np_f, entropy_hist, max_qe_hist, kl_b_hist, kl_r_hist, loglik_hist, grad_norm_rho_hist, grad_norm_b_hist, args.temp_start, args.temp_end, args.max_steps, args.min_steps, mu_r_top_hist, sigma_r_top_hist, mu_b_top_hist, sigma_b_top_hist, mu_b_f.detach().cpu().numpy().mean(0), lsb_f.detach().cpu().numpy().mean(0), taxon_idx, n_sites, ema_hist=ema_hist_full, stopped_early=stopped_early, stop_step=early_stop_step)
        torch.save(rho_net.state_dict(), os.path.join(step_out_dir, "rho_net.pt"))
        torch.save(b_net.state_dict(),   os.path.join(step_out_dir, "b_net.pt"))

    return rho_net, b_net, {"final_scaled_elbo": elbo_hist[-1], "iterations": len(elbo_hist), "max_steps": args.max_steps, "stopped_early": stopped_early, "stop_reason": stop_reason, "final_ema_elbo": ema_val, "final_entropy": entropy_hist[-1], "final_max_qe": max_qe_hist[-1], "final_kl_b": kl_b_hist[-1], "final_kl_r": kl_r_hist[-1], "final_loglik": loglik_hist[-1], "n_edges": n_edges}


# ══════════════════════════════════════════════════════════════════════════════
# MLL Estimation
# ══════════════════════════════════════════════════════════════════════════════

def sample_tree_with_log_q(args, names, leaf_lik_all, step_nets, device):
    n_use, allocator = len(names), InternalNodeAllocator(start=len(names))
    tree = build_t3_star(args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=allocator.alloc())
    log_q, log_p_Y = 0.0, None

    with torch.no_grad():
        for taxon_idx in range(3, n_use):
            rho_net, b_net = step_nets[taxon_idx]
            leaf_lik_current = {k: v for k, v in leaf_lik_all.items() if k in tree.adj and len(tree.adj[k]) == 1}
            _, cav_cache = compute_messages_and_cavities(tree, leaf_lik_current)
            edges = tree.edges()
            new_lik = leaf_lik_all[taxon_idx].to(device)
            edge_feats = build_edge_features(edges, cav_cache, new_lik, device, len(tree.adj), tree, args.jc_rate)
            pi_dna = PI_DNA.to(device)

            mu_r, log_sigma_r, h = rho_net(edge_feats)
            
            # Sample for ALL edges first
            std_r = torch.exp(log_sigma_r)
            rho_samp = torch.sigmoid(mu_r + std_r * torch.randn_like(mu_r)).unsqueeze(0)
            
            mu_b, log_sigma_b = b_net(h.detach(), rho_samp)
            b_samp = torch.exp(mu_b + torch.exp(log_sigma_b) * torch.randn_like(mu_b)).clamp(min=1e-8, max=2.0)

            # Evaluate exact samples
            _sc = _stack_cavities(edges, tree, cav_cache, device)
            loglik_samp = compute_loglik_all_edges(*_sc, rho_samp, b_samp, new_lik, pi_dna, args.jc_rate)
            
            q_e = torch.softmax(loglik_samp, dim=0)
            q_e_cpu = (q_e.detach().cpu().clamp(min=0.0) / (q_e.sum() + EPS))
            chosen_idx = torch.multinomial(q_e_cpu, 1).item()

            chosen_rho = float(rho_samp[0, chosen_idx])
            chosen_b   = float(b_samp[0, chosen_idx])
            
            log_q += (math.log(float(q_e_cpu[chosen_idx]) + 1e-30) + 
                      log_logistic_normal_density(chosen_rho, mu_r[chosen_idx].item(), log_sigma_r[chosen_idx].item()) + 
                      log_lognormal_density(chosen_b, mu_b[0, chosen_idx].item(), log_sigma_b[0, chosen_idx].item()))

            if taxon_idx == n_use - 1:
                log_p_Y = insertion_loglik_from_edge_batched(
                    tree, edges[chosen_idx], torch.tensor([chosen_rho], dtype=torch.float32, device=device), 
                    torch.tensor([chosen_b], dtype=torch.float32, device=device), new_lik, cav_cache, pi_dna, args.jc_rate
                ).item()

            insert_taxon_on_edge(tree, edges[chosen_idx], taxon_idx, chosen_rho, chosen_b, allocator)

    log_p_b_all = sum(math.log(PRIOR_LAM_B) - PRIOR_LAM_B * tree.length(u, v) for u, v in tree.edges())
    log_p_b = log_p_b_all - (3 * math.log(PRIOR_LAM_B) - PRIOR_LAM_B * (args.edge_len_0 + args.edge_len_1 + args.edge_len_2))
    return tree, log_q, log_p_Y, log_p_b

def estimate_mll(args, names, leaf_lik_all, step_nets, device, k_mll=1000):
    print(f"\n  Estimating MLL with K={k_mll} importance samples...")
    log_weights = []
    for _ in tqdm(range(k_mll), desc="MLL samples", leave=False, dynamic_ncols=True):
        _, log_q, log_p_Y, log_p_b = sample_tree_with_log_q(args, names, leaf_lik_all, step_nets, device)
        if log_p_Y is not None: log_weights.append(log_p_Y + log_p_b - log_q)

    log_weights = np.array(log_weights); max_lw = log_weights.max()
    mll = float(max_lw + math.log(np.exp(log_weights - max_lw).mean()))
    log_sum_w = max_lw + math.log(np.exp(log_weights - max_lw).sum())
    log_sum_w2 = np.log(np.exp(2.0 * (log_weights - max_lw)).sum()) + 2.0 * max_lw
    ess = float(math.exp(2.0 * log_sum_w - log_sum_w2))

    print(f"\n{'='*55}\n  MARGINAL LOG-LIKELIHOOD (MLL) ESTIMATE\n{'='*55}\n  MLL              = {mll:.4f}  nats\n  K                = {k_mll}\n  ESS              = {ess:.1f} / {k_mll}  ({100.0 * ess / k_mll:.1f}%)\n  log-weight mean  = {log_weights.mean():.4f}\n  log-weight std   = {float(np.std(log_weights)):.4f}\n{'='*55}\n")
    return {"mll": mll, "k_mll": k_mll, "ess": ess, "log_weight_mean": float(log_weights.mean())}


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_training_pass(args, names, leaf_lik_all, device, run_dir):
    n_use, step_models, step_nets, step_meta = len(names), {}, {}, []
    for taxon_idx in range(3, n_use):
        print(f"\n=== Train Step  taxon_idx={taxon_idx}  name={names[taxon_idx]} ===")
        step_dir = os.path.join(run_dir, f"train_step_{taxon_idx:03d}_{names[taxon_idx]}")
        os.makedirs(step_dir, exist_ok=True)
        
        rho_net, b_net, train_summary = train_single_insertion_step(taxon_idx, step_nets, leaf_lik_all[taxon_idx], leaf_lik_all, names, device, args, step_dir)
        rho_net.eval(); b_net.eval()
        step_models[taxon_idx] = (os.path.join(step_dir, "rho_net.pt"), os.path.join(step_dir, "b_net.pt"))
        step_nets[taxon_idx] = (rho_net, b_net)
        step_meta.append({"taxon_idx": taxon_idx, "taxon_name": names[taxon_idx], "train_summary": train_summary})

    trained_tree = sample_tree_from_trained_steps(args, names, leaf_lik_all, step_nets, device)
    save_crosstep_plots(run_dir, step_meta, names)
    return trained_tree, step_models, step_meta

def preload_step_nets_ca(step_models, hidden_dim, device):
    step_nets = {}
    for taxon_idx, (rho_path, b_path) in step_models.items():
        rho_net = RhoNet(in_channels=EDGE_FEATURE_DIM, hidden=hidden_dim).to(device)
        rho_net.load_state_dict(torch.load(rho_path, map_location=device, weights_only=True)); rho_net.eval()
        b_net = BNet(hidden=hidden_dim).to(device)
        b_net.load_state_dict(torch.load(b_path, map_location=device, weights_only=True)); b_net.eval()
        step_nets[taxon_idx] = (rho_net, b_net)
    return step_nets

def save_sampled_tree_plots(sampled_trees, names, run_dir, leaf_lik_all, device, args, step_nets):
    os.makedirs(run_dir, exist_ok=True)
    n_taxa, expected_edges = len(names), 2 * len(names) - 3

    topo_strings = [to_newick_topology_only(t, names) for t in sampled_trees]
    topology_counts = Counter(topo_strings)
    top_topos = topology_counts.most_common(min(20, len(topology_counts)))
    top_freqs = [c / len(sampled_trees) for _, c in top_topos]

    plt.figure(figsize=(10, 4)); plt.bar(range(len(top_freqs)), top_freqs, color='steelblue', alpha=0.8)
    plt.xlabel("Topology rank (0 = most common)"); plt.ylabel("Posterior probability estimate"); plt.title(f"Top-{len(top_freqs)} Topology Frequencies\n{len(topology_counts)} unique / {len(sampled_trees)} samples"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "topology_frequencies.png"), dpi=150); plt.close()

    all_bl = np.array([t.length(u, v) for t in sampled_trees for u, v in t.edges()])
    plt.figure(figsize=(8, 4)); plt.hist(all_bl, bins=60, density=True, color='steelblue', alpha=0.75)
    plt.axvline(np.median(all_bl), color='red', linestyle='--', linewidth=1.5, label=f'Median: {np.median(all_bl):.4f}')
    plt.axvline(1.0 / PRIOR_LAM_B, color='black', linestyle=':', linewidth=1.5, label=f'Prior mean: {1/PRIOR_LAM_B:.3f}')
    plt.xlabel("Branch length (sub/site)"); plt.ylabel("Density"); plt.title("Branch Length Distribution Across All Sampled Trees"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "sampled_branch_lengths.png"), dpi=150); plt.close()

    ec = Counter(len(t.edges()) for t in sampled_trees)
    bad = sum(1 for t in sampled_trees if len(t.edges()) != expected_edges)
    plt.figure(figsize=(6, 3)); plt.bar([str(k) for k in sorted(ec)], [ec[k] for k in sorted(ec)], color=['red' if k != expected_edges else 'steelblue' for k in sorted(ec)])
    plt.title(f"Edge Count Check (expected {expected_edges})\n{'✓ All correct' if bad == 0 else f'⚠ {bad} trees wrong!'}"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "edge_count_check.png"), dpi=150); plt.close()

    last_tidx = max(step_nets.keys())
    rho_net_l, b_net_l = step_nets[last_tidx]
    allocator_tmp, tree_tmp = InternalNodeAllocator(start=n_taxa), build_t3_star(args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=InternalNodeAllocator(start=n_taxa).alloc())
    with torch.no_grad():
        for tidx in range(3, n_taxa):
            rn, bn = step_nets[tidx]
            edges_t, q_t, _, mu_r_t, _, _, _, h_t, rho_np_t, b_np_t, _, _, _, _, _ = evaluate_edges_ca(tree_tmp, leaf_lik_all[tidx], leaf_lik_all, rn, bn, device, 10, args.jc_rate, 1.0)
            e_c, r_c, b_c = choose_action_ca(edges_t, q_t, rho_np_t, b_np_t, mu_r_t, h_t, bn, device, False)
            insert_taxon_on_edge(tree_tmp, e_c, tidx, r_c, b_c, allocator_tmp)

    _, cav_final = compute_messages_and_cavities(tree_tmp, {k: v for k, v in leaf_lik_all.items() if k in tree_tmp.adj and len(tree_tmp.adj[k]) == 1})
    scale_means = [ls.detach().cpu().mean().item() for _, (_, ls) in cav_final.items()]
    
    plt.figure(figsize=(8, 4)); plt.hist(scale_means, bins=30, color='teal', alpha=0.75); plt.axvline(0.0, color='red', linestyle='--', linewidth=1.5, label='0 (bad)')
    plt.title("Log-Scale Health"); plt.xlabel("Mean log_scale"); plt.ylabel("Count"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "log_scale_health.png"), dpi=150); plt.close()

    with torch.no_grad():
        edges_f, _, _, _, _, _, _, _, _, _, _, _, loglik_mc, _, _ = evaluate_edges_ca(tree_tmp, leaf_lik_all[last_tidx], leaf_lik_all, rho_net_l, b_net_l, device, args.k_eval, args.jc_rate, 1.0)
    
    sorted_indices = torch.argsort(loglik_mc, descending=True)[:min(5, len(edges_f))]
    site_lls = loglik_mc[sorted_indices].cpu().numpy()
    edge_labels = [f"({edges_f[i][0]},{edges_f[i][1]})" for i in sorted_indices]

    plt.figure(figsize=(8, 4)); plt.bar(range(len(site_lls)), site_lls, color='teal', alpha=0.8); plt.xticks(range(len(site_lls)), edge_labels, rotation=45); plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Top 5 Edges"); plt.ylabel("True Sampled Log-lik"); plt.title("Actual Per-Edge Log-Likelihoods (from NN samples)"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "site_loglik_check.png"), dpi=150); plt.close()

def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=str, required=True); p.add_argument("--n_taxa_use", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=500); p.add_argument("--n_ca_rounds", type=int, default=2)
    p.add_argument("--n_phase1_steps", type=int, default=5); p.add_argument("--n_phase2_steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--k_train", type=int, default=20); p.add_argument("--k_eval", type=int, default=100)
    p.add_argument("--n_tree_samples", type=int, default=1000); p.add_argument("--k_mll", type=int, default=1000)
    p.add_argument("--temp_start", type=float, default=50.0); p.add_argument("--temp_end", type=float, default=1.0)
    p.add_argument("--sample_temp", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jc_rate", type=float, default=1.0); p.add_argument("--cuda", action="store_true")
    p.add_argument("--out_root", type=str, default="results_ca_vi"); p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--min_steps", type=int, default=80); p.add_argument("--patience", type=int, default=50)
    p.add_argument("--conv_tol", type=float, default=2e-4); p.add_argument("--ema_alpha", type=float, default=0.1)
    return p

def main(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.num_threads); torch.set_num_interop_threads(max(1, args.num_threads // 2))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.num_threads)); os.environ.setdefault("MKL_NUM_THREADS", str(args.num_threads))
    set_seed(args.seed)

    names_all, seqs_all = parse_fasta_in_order(args.fasta)
    n_use = len(names_all) if args.n_taxa_use is None else min(args.n_taxa_use, len(names_all))
    names, seqs = names_all[:n_use], seqs_all[:n_use]
    leaf_lik_all = {i: seq_to_likelihood_matrix(seqs[i]) for i in range(n_use)}

    args.edge_len_0, args.edge_len_1, args.edge_len_2 = compute_t3_branch_lengths(seqs[0], seqs[1], seqs[2])
    run_dir = os.path.join(args.out_root, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ntaxa{n_use}")
    os.makedirs(run_dir, exist_ok=False)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    with open(os.path.join(run_dir, "config.json"), "w") as f: json.dump(vars(args), f, indent=2)
    trained_tree, step_models, step_meta = run_training_pass(args, names, leaf_lik_all, device, run_dir)
    save_training_tree_json(trained_tree, names, os.path.join(run_dir, "trained_tree_edges.json"))
    save_nexus_trees([trained_tree], names, os.path.join(run_dir, "trained_tree.nex"))
    with open(os.path.join(run_dir, "step_meta.json"), "w") as f: json.dump(step_meta, f, indent=2)

    step_nets = preload_step_nets_ca(step_models, args.hidden_dim, device)
    sampled_trees = [sample_tree_from_trained_steps(args, names, leaf_lik_all, step_nets, device) for _ in tqdm(range(args.n_tree_samples), desc="Sample trees")]
    save_nexus_trees(sampled_trees, names, os.path.join(run_dir, "sampled_trees.nex"))

    mll_results = estimate_mll(args, names, leaf_lik_all, step_nets, device, k_mll=args.k_mll)
    with open(os.path.join(run_dir, "mll_results.json"), "w") as f: json.dump(mll_results, f, indent=2)

    save_sampled_tree_plots(sampled_trees, names, run_dir, leaf_lik_all, device, args, step_nets)

if __name__ == "__main__":
    main(build_args().parse_args())