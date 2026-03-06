import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

EPS = 1e-12
PI_DNA = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
EDGE_FEATURE_DIM = 12

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
            if not line: continue
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

    def neighbors(self, node: int) -> List[int]: return self.adj.get(node, []) 
    def length(self, u: int, v: int) -> float: return self.edge_len[frozenset((u, v))]

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
        if v not in self.adj[u]: self.adj[u].append(v)
        if u not in self.adj[v]: self.adj[v].append(u)
        self.edge_len[frozenset((u, v))] = float(t)

    def remove_edge(self, u: int, v: int) -> float:
        key = frozenset((u, v))
        t = self.edge_len.pop(key)
        if u in self.adj: self.adj[u] = [x for x in self.adj[u] if x != v]
        if v in self.adj: self.adj[v] = [x for x in self.adj[v] if x != u]
        return t

class InternalNodeAllocator:
    def __init__(self, start: int): self.next_id = start
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
    def p_dist(sA: str, sB: str) -> float:
        mismatches, comparable = 0, 0
        valid_bases = set("ACGT")
        for a, b in zip(sA, sB):
            if a in valid_bases and b in valid_bases:
                comparable += 1
                if a != b: mismatches += 1
        return mismatches / comparable if comparable > 0 else 0.75

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

def insert_taxon_on_edge(tree: UnrootedTree, edge: Tuple[int, int], new_taxon: int, rho: float, b_len: float, allocator: InternalNodeAllocator) -> int:
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

def compute_messages_and_cavities(tree: UnrootedTree, leaf_lik: Dict[int, torch.Tensor]):
    any_leaf = next(iter(leaf_lik.values()))
    n_sites = any_leaf.shape[0]
    device = any_leaf.device
    msg_cache, cav_cache = {}, {}

    def cavity(a: int, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (a, b)
        if key in cav_cache: return cav_cache[key]
        psi = leaf_lik[a].clone() if a in leaf_lik else torch.ones(n_sites, 4, dtype=torch.float32, device=device)
        acc_log_scale = torch.zeros(n_sites, dtype=torch.float32, device=device)
        
        for k in tree.neighbors(a):
            if k == b: continue
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
        if key in msg_cache: return msg_cache[key]
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

def edge_feature(m_uv: torch.Tensor, m_vu: torch.Tensor, new_leaf_lik: torch.Tensor) -> torch.Tensor:
    m_uv_n = m_uv / (m_uv.sum(dim=-1, keepdim=True) + EPS)
    m_vu_n = m_vu / (m_vu.sum(dim=-1, keepdim=True) + EPS)
    y_n = new_leaf_lik / (new_leaf_lik.sum(dim=-1, keepdim=True) + EPS)
    return torch.cat([m_uv_n, m_vu_n, y_n], dim=-1)

def build_edge_features(edges: List[Tuple[int, int]], msg_cache, new_leaf_lik: torch.Tensor, device: torch.device) -> torch.Tensor:
    new_leaf_lik = new_leaf_lik.to(device)
    feats = [edge_feature(msg_cache[(u, v)][0].to(device), msg_cache[(v, u)][0].to(device), new_leaf_lik) for (u, v) in edges]
    return torch.stack(feats, dim=0)

def jc_transition_batched(t: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    e = torch.exp(-4.0 * rate * t / 3.0)
    same = 0.25 + 0.75 * e
    diff = 0.25 - 0.25 * e
    p = diff.view(-1, 1, 1).expand(-1, 4, 4).clone()
    idx = torch.arange(4, device=t.device)
    p[:, idx, idx] = same.unsqueeze(1).expand(-1, 4)
    return p

def insertion_loglik_from_edge_batched(tree: UnrootedTree, edge: Tuple[int, int], rho: torch.Tensor, b_len: torch.Tensor, new_leaf_lik: torch.Tensor, cav_cache, pi: torch.Tensor, rate: float) -> torch.Tensor:
    u, v = edge
    t_uv = torch.tensor(tree.length(u, v), dtype=torch.float32, device=rho.device)
    t_uw = rho * t_uv
    t_wv = (1.0 - rho) * t_uv

    psi_u_excl_v, log_scale_u = cav_cache[(u, v)]
    psi_v_excl_u, log_scale_v = cav_cache[(v, u)]
    
    psi_u_excl_v, psi_v_excl_u = psi_u_excl_v.to(rho.device), psi_v_excl_u.to(rho.device)  
    log_scale_u, log_scale_v = log_scale_u.to(rho.device), log_scale_v.to(rho.device)
    new_leaf_lik, pi = new_leaf_lik.to(rho.device), pi.to(rho.device)

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

class CNNPosteriorNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4) 
        )
    
    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2) 
        h = self.cnn(x).squeeze(-1)
        out = self.mlp(h)
        mu_b = out[:, 0] 
        log_sigma_b = out[:, 1].clamp(-5.0, 3.0) 
        mu_r = out[:, 2] 
        log_sigma_r = out[:, 3].clamp(-5.0, 3.0) 
        return mu_b, log_sigma_b, mu_r, log_sigma_r

def gaussian_kl_standard(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    sigma2 = torch.exp(2.0 * log_sigma)
    return 0.5 * (mu.pow(2) + sigma2 - 1.0 - 2.0 * log_sigma)

def mc_expected_loglik_and_samples(tree: UnrootedTree, edges: List[Tuple[int, int]], mu_b: torch.Tensor, log_sigma_b: torch.Tensor, mu_r: torch.Tensor, log_sigma_r: torch.Tensor, new_leaf_lik: torch.Tensor, cav_cache, pi: torch.Tensor, k_samples: int, jc_rate: float):
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
        ll_k = insertion_loglik_from_edge_batched(tree=tree, edge=e, rho=rho[:, i], b_len=b[:, i], new_leaf_lik=new_leaf_lik, cav_cache=cav_cache, pi=pi, rate=jc_rate)
        loglik_mc_list.append(ll_k.mean())

    loglik_mc = torch.stack(loglik_mc_list)
    return loglik_mc, rho.detach().cpu().numpy(), b.detach().cpu().numpy()

def evaluate_edges(tree: UnrootedTree, new_leaf_lik: torch.Tensor, leaf_lik_all: Dict[int, torch.Tensor], net: CNNPosteriorNet, device: torch.device, k_samples: int, jc_rate: float, temperature: float = 1.0):
    leaf_lik_current = {k: v for k, v in leaf_lik_all.items() if k in tree.adj and len(tree.adj[k]) == 1}
    msg_cache, cav_cache = compute_messages_and_cavities(tree, leaf_lik_current)
    edges = tree.edges()
    edge_feats = build_edge_features(edges=edges, msg_cache=msg_cache, new_leaf_lik=new_leaf_lik, device=device)

    new_leaf_lik, pi_dna = new_leaf_lik.to(device), PI_DNA.to(device)
    mu_b, log_sigma_b, mu_r, log_sigma_r = net(edge_feats)

    loglik_mc, rho_samples, b_samples = mc_expected_loglik_and_samples(
        tree=tree, edges=edges, mu_b=mu_b, log_sigma_b=log_sigma_b, mu_r=mu_r,
        log_sigma_r=log_sigma_r, new_leaf_lik=new_leaf_lik, cav_cache=cav_cache,
        pi=pi_dna, k_samples=k_samples, jc_rate=jc_rate,
    )

    kl_b = gaussian_kl_standard(mu_b, log_sigma_b)
    kl_r = gaussian_kl_standard(mu_r, log_sigma_r)
    elbo_e = loglik_mc - kl_b - kl_r
    q_e = torch.softmax(elbo_e / temperature, dim=0)
    return edges, q_e, elbo_e, mu_b, log_sigma_b, mu_r, log_sigma_r, rho_samples, b_samples

def choose_action(edges: List[Tuple[int, int]], q_e: torch.Tensor, mu_b: torch.Tensor, log_sigma_b: torch.Tensor, mu_r: torch.Tensor, log_sigma_r: torch.Tensor, sample_continuous: bool = True):
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

def escape_nexus_name(name: str) -> str:
    if any(c in name for c in " \t\n(),:;[]="): return f"'{name}'"
    return name

def to_newick(tree: UnrootedTree, taxon_names: List[str], label_mapper: Optional[Callable[[int], str]] = None) -> str:
    start_node = None
    for u in tree.adj:
        if u >= len(taxon_names):
            start_node = u
            break
    if start_node is None:
        start_node = next(iter(tree.adj.keys()), None)
        if start_node is None: return "();"

    visited = set()
    def build_substring(u: int) -> str:
        visited.add(u)
        children = [v for v in tree.neighbors(u) if v not in visited]
        if not children: return label_mapper(u) if label_mapper else taxon_names[u]

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
        f.write("#NEXUS\n\nBEGIN TAXA;\n")
        f.write(f"\tDIMENSIONS NTAX={len(taxon_names)};\n\tTAXLABELS\n")
        for name in taxon_names: f.write(f"\t\t{escape_nexus_name(name)}\n")
        f.write("\t;\nEND;\n\nBEGIN TREES;\n\tTRANSLATE\n")
        for i, name in enumerate(taxon_names):
            sep = "," if i < len(taxon_names) - 1 else ""
            f.write(f"\t\t{i+1} {escape_nexus_name(name)}{sep}\n")
        f.write("\t;\n")
        def map_idx_to_nexus_id(idx: int) -> str: return str(idx + 1)
        for idx, t in enumerate(trees):
            nwk = to_newick(t, taxon_names, label_mapper=map_idx_to_nexus_id)
            f.write(f"\tTREE tree_{idx+1} = {nwk}\n")
        f.write("END;\n")

def preload_step_nets(step_meta_path: str, run_dir: str, hidden_dim: int, device: torch.device) -> Dict[int, CNNPosteriorNet]:
    with open(step_meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    
    step_nets: Dict[int, CNNPosteriorNet] = {}
    for item in meta_list:
        taxon_idx = item["taxon_idx"]
        # Fallback handling to ensure paths resolve correctly even if you moved the folder
        model_path = item.get("model_path")
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
             model_path = os.path.join(run_dir, f"train_step_{taxon_idx:03d}_{item['taxon_name']}", "cnn_posterior_net.pt")

        net = CNNPosteriorNet(in_channels=EDGE_FEATURE_DIM, hidden=hidden_dim).to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        step_nets[taxon_idx] = net
    return step_nets

def sample_tree_from_trained_steps(args: argparse.Namespace, names: List[str], leaf_lik_all: Dict[int, torch.Tensor], step_nets: Dict[int, CNNPosteriorNet], device: torch.device) -> UnrootedTree:
    n_use = len(names)
    allocator = InternalNodeAllocator(start=n_use)
    center_id = allocator.alloc()
    tree = build_t3_star(args.edge_len_0, args.edge_len_1, args.edge_len_2, center_idx=center_id)

    with torch.no_grad():
        for taxon_idx in range(3, n_use):
            net = step_nets[taxon_idx]
            edges, q_e, _, mu_b, log_sigma_b, mu_r, log_sigma_r, _, _ = evaluate_edges(
                tree, leaf_lik_all[taxon_idx], leaf_lik_all, net, device, k_samples=args.k_eval, jc_rate=args.jc_rate, temperature=20.0
            )
            chosen_edge, chosen_rho, chosen_b = choose_action(
                edges, q_e, mu_b, log_sigma_b, mu_r, log_sigma_r, sample_continuous=True
            )
            insert_taxon_on_edge(tree, chosen_edge, taxon_idx, chosen_rho, chosen_b, allocator)

    return tree

def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone Tree Sampler for Trained Posterior")
    p.add_argument("--fasta", type=str, required=True, help="Input FASTA path used for training")
    p.add_argument("--run_dir", type=str, required=True, help="Directory containing config.json, step_meta.json, and the trained models")
    p.add_argument("--n_tree_samples", type=int, default=1000, help="Number of trees to sample")
    p.add_argument("--k_eval", type=int, default=100, help="MC samples for evaluating ELBO during generation")
    p.add_argument("--seed", type=int, default=42, help="Random seed strictly for generation")
    p.add_argument("--cuda", action="store_true", help="Use GPU for generation if available")
    p.add_argument("--out_file", type=str, default="sampled_trees_standalone.nex", help="Output nexus file name")
    return p

def main() -> None:
    parser = build_args()
    cli_args = parser.parse_args()
    set_seed(cli_args.seed)

    config_path = os.path.join(cli_args.run_dir, "config.json")
    meta_path = os.path.join(cli_args.run_dir, "step_meta.json")
    
    if not os.path.exists(config_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing config.json or step_meta.json in {cli_args.run_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        train_config = json.load(f)

    # Merge configs (CLI overrides training defaults for sampling specific vars)
    args = argparse.Namespace()
    args.jc_rate = train_config.get("jc_rate", 1.0)
    args.hidden_dim = train_config.get("hidden_dim", 64)
    args.n_taxa_use = train_config.get("n_taxa_use", None)
    args.k_eval = cli_args.k_eval
    
    device = torch.device("cuda" if cli_args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Loading environment from: {cli_args.run_dir}")
    print(f"Using device: {device}")

    names_all, seqs_all = parse_fasta_in_order(cli_args.fasta)
    n_total = len(names_all)
    n_use = n_total if args.n_taxa_use is None else min(args.n_taxa_use, n_total)
    
    names = names_all[:n_use]
    seqs = seqs_all[:n_use]
    leaf_lik_all = {i: seq_to_likelihood_matrix(seqs[i]) for i in range(n_use)}

    e0, e1, e2 = compute_t3_branch_lengths(seqs[0], seqs[1], seqs[2])
    args.edge_len_0, args.edge_len_1, args.edge_len_2 = e0, e1, e2

    print("Loading pre-trained networks...")
    step_nets = preload_step_nets(meta_path, cli_args.run_dir, args.hidden_dim, device)

    sampled_trees = []
    print(f"\nSampling {cli_args.n_tree_samples} trees from trained posterior...")
    sample_iter = tqdm(range(cli_args.n_tree_samples), desc="Sample trees", leave=True, dynamic_ncols=True)
    
    for _ in sample_iter:
        tree_i = sample_tree_from_trained_steps(args, names, leaf_lik_all, step_nets, device)
        sampled_trees.append(tree_i)

    out_path = os.path.join(cli_args.run_dir, cli_args.out_file)
    save_nexus_trees(sampled_trees, names, out_path)
    print(f"\nDone. Saved {cli_args.n_tree_samples} trees to {out_path}.")

if __name__ == "__main__":
    main()