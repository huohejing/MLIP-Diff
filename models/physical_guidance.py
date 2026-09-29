#!/usr/bin/env python3
"""
Interaction-aware physical guidance for the DiffGui sampling loop.

During late-stage denoising the predicted ligand state is reconstructed into a
chemistry-aware heavy-atom proxy (project_to_mace_proxy) and evaluated with the
MACE-OFF24 potential under a fixed local protein environment. Two evaluations
separate the two physical contributions:

    F_complex      ligand proxy together with the fixed protein environment
    F_intra        ligand proxy alone
    F_inter        = F_complex - F_intra

The difference is the signal that is used: it carries the change in ligand
forces induced by the protein environment, rather than bulk conformational
relaxation of the ligand itself. The correction is applied as a direction whose
magnitude is set by the current diffusion step's own coordinate update, so the
diffusion trajectory determines the spatial scale of each correction and the
force magnitude is never converted directly into a displacement.

Atom and bond identities keep following the original DiffGui discrete
transition process throughout; only the coordinate trajectory is modified.
"""

import os
import math
import numpy as np
import torch


# DiffGui aromatic atom type index → atomic number
# From utils/transforms.py map_index_to_atom_type_aromatic
AROMATIC_INDEX_TO_ATOMIC_NUMBER = {
    0: 1, 1: 5, 2: 6, 3: 6, 4: 7, 5: 7, 6: 8, 7: 8,
    8: 9, 9: 15, 10: 15, 11: 16, 12: 16, 13: 17, 14: 35, 15: 53,
}


# RDKit sanitize is the final arbiter for exotic charged states.
MAX_VALENCE = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}

# Extreme bond distance thresholds (Å) — very loose, only catch garbage.
# Used by project_to_mace_proxy (keep them out of the deleted legacy gates).
BOND_DMIN, BOND_DMAX = 1.0, 2.5


# ============================================================
#  Deterministic single proxy projection (MACE guidance input)
#  Replaces the hard-skip gates (repair_none / postcheck /
#  repair_ratio) with ONE deterministic reconstruction:
#  DiffGui predicted-clean state → exactly ONE valid, connected,
#  valence-valid, RDKit-sanitizable heavy-atom proxy.
#  Never calls MACE. Never writes back to the diffusion state.
# ============================================================

def _p5_default():
    """Conservative no-bond prior for halfedges missing from the posterior map."""
    return np.array([0.6, 0.2, 0.1, 0.05, 0.05], dtype=np.float32)


def _p5_of(edge, bond_type_probs):
    i, j, _ = edge
    return bond_type_probs.get((min(i, j), max(i, j)), _p5_default())


def _p_dbl_support(p5):
    """Posterior support for a double bond on this edge.

    Categories: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic.
    The true double class is p5[2]; a triple (p5[3]) can be downgraded to
    double; an aromatic edge (p5[4]) resolves to single or double so it
    contributes half.
    """
    return float(p5[2] + p5[3] + 0.5 * p5[4])


def _p_any_bond(p5):
    return float(p5[1] + p5[2] + p5[3] + p5[4])


def _edge_in_cycle(edges, keep_idx):
    """True if edges[keep_idx] lies on any cycle of the graph formed by `edges`."""
    i, j, _ = edges[keep_idx]
    adj = {}
    for k, (a, b, _) in enumerate(edges):
        if k == keep_idx:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = {i}
    stack = [i]
    while stack:
        u = stack.pop()
        for v in adj.get(u, []):
            if v == j:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _covalent_dmax(zi, zj, gamma=1.5):
    """Element-dependent covalent-bond length ceiling (Å) for guessed bonds.

    Used for connectivity bridges: a bond WE invent must be chemically
    plausible, unlike argmax bonds from the model which are tolerated up
    to the global BOND_DMAX. RDKit's covalent radii are conservative
    (C: 0.68 Å), so gamma=1.5 gives ~1.3-1.4x the real single-bond length
    (C-C ceiling 2.04 Å): long-but-plausible early-diffusion bridges pass,
    absurd 2.8 Å fake bonds are rejected.
    """
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    return gamma * (pt.GetRcovalent(int(zi)) + pt.GetRcovalent(int(zj)))


def _lower_tail_mean(values, frac=0.2):
    """Mean of the lowest `frac` fraction of values.

    Used for confidence aggregation: the least-determined atoms/bonds of a
    molecule decide whether its chemical identity is really stable — a
    simple overall mean hides a single 0.20-confidence atom behind 19
    confident ones.
    """
    v = sorted(float(x) for x in values)
    if not v:
        return 1.0
    k = max(1, int(math.ceil(len(v) * frac)))
    return float(np.mean(v[:k]))


def _kekulize_aromatic(edges, elements, bond_type_probs):
    """Deterministic Kekulé assignment for aromatic (order 4) edges in place.

    UNIFIED aromatic-heterocycle handling via Hückel pi-electron counting,
    applied identically to single rings and fused ring systems:
      - pi_target = 4 * ring_count + 2  (ring_count = E - V + 1)
      - each sp2 C contributes 1 pi; O/S contribute 2 pi;
      - N contributes 1 pi (pyridine-type, no H) or 2 pi (pyrrole-type,
        one H after AddHs);
      - the pi equation fixes how many N must be pyrrole-type per system.
    ALL choices of "which N" are enumerated — including the CARTESIAN
    PRODUCT across independent ring systems (two imidazoles are 2x2=4
    joint choices, not 2). Every joint choice is solved by exact
    backtracking Kekulé matching; only the single best proxy is kept
    (no second proxy, no second MACE).

    Tautomer margin: the best vs second-best assignment with a DIFFERENT
    per-atom H distribution is reported as 'proxy_tautomer_conf' (sigmoid
    on the score margin). Pure C=C resonance reshuffles leave H counts
    unchanged and do not count.

    Systems with no pi solution are counted in 'proxy_aromatic_ambiguous';
    if the Kekulé matching ultimately fails everywhere, the last-resort
    all-single proxy is counted in 'proxy_aromatic_fallback' (the caller
    floors confidence at 0.1 for it).

    Returns dict with 'proxy_kekule_changes', 'proxy_aromatic_ambiguous',
    'proxy_aromatic_fallback', 'proxy_tautomer_conf'.
    """
    stats = {'proxy_kekule_changes': 0, 'proxy_aromatic_ambiguous': 0,
             'proxy_aromatic_fallback': 0, 'proxy_tautomer_conf': 1.0}
    arom = [k for k, e in enumerate(edges) if e[2] == 4]
    if not arom:
        return stats

    # ---- 1. non-ring aromatic edges: resolve by posterior ----
    for k in arom:
        if _edge_in_cycle(edges, k):
            continue
        i, j, _ = edges[k]
        p5 = bond_type_probs.get((min(i, j), max(i, j)), _p5_default())
        p_dbl = _p_dbl_support(p5)
        p_sgl = float(p5[1])
        edges[k][2] = 2 if p_dbl > p_sgl + 0.5 * p5[4] else 1
        stats['proxy_kekule_changes'] += 1
    arom = [k for k, e in enumerate(edges) if e[2] == 4]
    if not arom:
        return stats

    # ---- 2. ring systems = connected components of the aromatic subgraph ----
    adj = {}
    for k in arom:
        i, j, _ = edges[k]
        adj.setdefault(i, []).append((j, k))
        adj.setdefault(j, []).append((i, k))
    seen_atoms = set()
    components = []
    for a0 in adj:
        if a0 in seen_atoms:
            continue
        atoms, eids, stack = set(), set(), [a0]
        seen_atoms.add(a0)
        while stack:
            u = stack.pop()
            atoms.add(u)
            for v, k in adj.get(u, []):
                eids.add(k)
                if v not in seen_atoms:
                    seen_atoms.add(v)
                    stack.append(v)
        components.append((atoms, eids))

    # S (16) target lowered to 2 for thiophene-type aromatic S.
    targets = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 2, 17: 1, 35: 1, 53: 1}
    n_edges_total = {}
    for k, e in enumerate(edges):
        n_edges_total[e[0]] = n_edges_total.get(e[0], 0) + 1
        n_edges_total[e[1]] = n_edges_total.get(e[1], 0) + 1

    # ---- 3. per-system pi counting -> N-role choice groups ----
    n_type = {}  # atom -> 'pyridine' | 'pyrrole' | 'os'
    choice_groups = []  # list of list of (n_list, combo) — Cartesian product

    def min_dbl_neighbour(a):
        vals = [_p_dbl_support(bond_type_probs.get((min(a, b), max(a, b)), _p5_default()))
                for b, k in adj.get(a, [])]
        return min(vals) if vals else 1.0

    from itertools import combinations, product
    for atoms, eids in components:
        m = len(atoms)
        ring_count = len(eids) - m + 1  # E - V + 1 (1 for a single ring)
        pi_target = 4 * ring_count + 2  # 6 for 5/6/7-member rings; 10 for
        # naphthalene/indole/purine; pyrene (4n) is handled by the all-C path
        n_c = sum(1 for a in atoms if elements[a] == 6)
        n_n = sum(1 for a in atoms if elements[a] == 7)
        n_os = sum(1 for a in atoms if elements[a] in (8, 16))
        n_list = [a for a in atoms if elements[a] == 7]
        for a in atoms:
            if elements[a] in (8, 16):
                n_type[a] = 'os'
        if n_n == 0 and n_os == 0:
            # all-C system (benzene, naphthalene, pyrene): no protonation
            # choice — the Kekulé matching decides everything
            continue
        need = pi_target - n_c - 2 * n_os  # pi the N's must supply
        n_pyrrole = need - n_n  # N's upgraded from 1 to 2 pi
        if 0 <= n_pyrrole <= n_n:
            choice_groups.append([(n_list, c) for c in combinations(n_list, n_pyrrole)])
        else:
            stats['proxy_aromatic_ambiguous'] += 1
            for a in n_list:
                n_type[a] = 'pyridine'  # conservative fallback

    # ---- 4. exact backtracking Kekulé matching ----
    def solve_all(demand):
        """Yield (given, decided, score) for every valid assignment."""
        given = dict.fromkeys(adj, 0)
        decided = {}
        out = []

        def bt(score):
            for a in sorted(adj, key=lambda x: (-(demand[x] - given[x]), x)):
                if given[a] >= demand[a]:
                    continue
                cands = []
                for k in arom:
                    if k in decided:
                        continue
                    i, j, _ = edges[k]
                    if i != a and j != a:
                        continue
                    b = j if i == a else i
                    if b in adj and given[b] >= demand[b]:
                        continue  # partner saturated — never over-allocate
                    cands.append((k, b))
                cands.sort(key=lambda kb: -_p_dbl_support(bond_type_probs.get(
                    (min(edges[kb[0]][0], edges[kb[0]][1]),
                     max(edges[kb[0]][0], edges[kb[0]][1])), _p5_default())))
                for k, b in cands:
                    p5 = bond_type_probs.get(
                        (min(edges[k][0], edges[k][1]),
                         max(edges[k][0], edges[k][1])))
                    s = _p_dbl_support(p5) if p5 is not None else 0.5
                    decided[k] = b
                    edges[k][2] = 2
                    given[a] += 1
                    if b in adj:
                        given[b] += 1
                    bt(score + s)
                    given[a] -= 1
                    if b in adj:
                        given[b] -= 1
                    del decided[k]
                    edges[k][2] = 4
                return
            out.append((dict(given), dict(decided), score))

        bt(0.0)
        return out

    def h_distribution(decided):
        """Per-atom H count implied by this assignment. Two assignments are
        the SAME tautomer iff this distribution is identical: a pure C=C
        resonance reshuffle (benzene, pyridine's two N=C placements) leaves
        every H count unchanged; a real tautomer moves an H between atoms."""
        h = []
        for a in sorted(adj):
            s = 0
            for k in arom:
                if edges[k][0] == a or edges[k][1] == a:
                    s += (2 if k in decided else 1)
            tgt = targets.get(elements[a], 4)
            h.append(max(0, tgt - s))
        return tuple(h)

    def demand_from_types(nt):
        demand = {}
        for a in adj:
            tgt = targets.get(elements[a], 4)
            d = min(1, max(0, tgt - n_edges_total.get(a, 0)))
            t = nt.get(a)
            if t in ('pyrrole', 'os'):
                d = 0
            demand[a] = d
        return demand

    # ---- 5. solve: Cartesian product over all systems' N-role choices ----
    solutions = []
    if choice_groups:
        for joint in product(*choice_groups):
            nt = dict(n_type)
            for (n_list, combo) in joint:
                for a in n_list:
                    nt[a] = 'pyrrole' if a in combo else 'pyridine'
            demand = demand_from_types(nt)
            for g, dec, sc in solve_all(demand):
                solutions.append((sc, h_distribution(dec), dec))
    else:
        demand = demand_from_types(n_type)
        for g, dec, sc in solve_all(demand):
            solutions.append((sc, h_distribution(dec), dec))

    sol = None
    if solutions:
        solutions.sort(key=lambda s: -s[0])
        best = solutions[0]
        second = next((s for s in solutions[1:] if s[1] != best[1]), None)
        if second is None:
            stats['proxy_tautomer_conf'] = 1.0  # unique tautomer / resonance
        else:
            d_score = best[0] - second[0]
            # sigmoid: ~0.1 margin in summed double posterior is "clear"
            stats['proxy_tautomer_conf'] = 1.0 / (1.0 + math.exp(-(d_score - 0.1) / 0.05))
        sol = best[2]
    else:
        # last resort: all-single proxy (sanitizable, chemistry degraded).
        # Counted separately from pi-ambiguity: the caller floors
        # confidence at 0.1 for this.
        stats['proxy_aromatic_fallback'] += 1
        for k in arom:
            edges[k][2] = 1
        return stats

    # Re-apply the best assignment: the enumerating backtracker restored
    # every edge to aromatic (4) while collecting solutions.
    for k in sol:
        edges[k][2] = 2

    # ---- 6. remaining aromatic edges -> single; count every conversion ----
    for k in arom:
        if edges[k][2] == 4:
            edges[k][2] = 1
        stats['proxy_kekule_changes'] += 1
    return stats


def _charge_adjust_for_rdkit(elements, edges):
    """Formal charges so RDKit sanitize accepts the topology.

    Common charged states within our organic subset:
      - tetravalent N -> [N+] (up to 2+);
      - nitro-like N+ paired with a terminal single-bond O -> that O is
        [O-] (otherwise AddHs protonates it to an OH, giving MACE the
        wrong local environment);
      - oxonium O (rare).
    Charges are proxy-only — never written back to DiffGui.
    """
    charges = np.zeros(len(elements), dtype=np.int32)
    for a, el in enumerate(elements):
        v = sum(o for i, j, o in edges if i == a or j == a)
        if el == 7 and v > 3:
            charges[a] = min(v - 3, 2)  # [N+]: up to 4 heavy bonds
        elif el == 8 and v > 2:
            charges[a] = v - 2          # oxonium (rare)
    # nitro-like pairing: single-bond terminal O next to a charged N -> [O-]
    for a, el in enumerate(elements):
        if el != 7 or charges[a] <= 0:
            continue
        for i, j, o in edges:
            if o != 1:
                continue
            b = j if i == a else (i if j == a else None)
            if b is None or elements[b] != 8 or charges[b] != 0:
                continue
            deg = sum(1 for i2, j2, o2 in edges if i2 == b or j2 == b)
            if deg == 1:
                charges[b] = -1
                break  # one [O-] per charged N is enough
    return charges


def project_to_mace_proxy(pos, elements, bonds, bond_type_probs=None,
                          atom_type_idx=None, atom_conf=None, max_iters=8):
    """Deterministic single chemical proxy for MACE guidance.

    From DiffGui's predicted-clean state — fixed atom identity, full bond
    posterior (all halfedges incl. no-bond), 3D geometry — reconstruct
    EXACTLY ONE valid, connected, valence-valid, RDKit-sanitizable heavy-
    atom proxy. Never calls MACE; never writes back into diffusion state.

    Proxy confidence (stats['proxy_confidence']) measures how much the
    ORIGINAL prediction supports the final proxy:
      C_topo = mean posterior support of each REAL topology edit
               (removed -> P(no); added bridge -> P(single)+P(double)+
                P(triple); order change -> P(new order); Kekulé conversions
                and formal charges are representation fixes, no penalty)
      C_atom = mean atom-type confidence (atom_conf, if provided)
      r_topo = (n_added + n_removed) / n_final_bonds
      C_proxy = min(C_atom, C_topo) * exp(-2 * r_topo)
    This replaces the old edit-count heuristic, which rewarded "few edits"
    regardless of how uncertain the model was about them.

    Args:
        pos: (N, 3) float32 heavy-atom positions in A
        elements: list of int atomic numbers (identity is FIXED)
        bonds: list of (i, j, order) — current argmax bonds (order 1-4)
        bond_type_probs: dict {(min_ij, max_ij): np.array(5)} full posterior
        atom_type_idx: optional per-atom argmax index (diagnostics only)
        atom_conf: optional (N,) array of atom-type argmax probabilities

    Returns:
        (bonds, formal_charges, stats) with bond order in {1, 2, 3} only,
        or None if no sanitizable proxy can be produced.

    Note on the `# ---- 2.x ----` markers below: those are this function's
    OWN step numbers, kept from development. They are unrelated to the
    section numbers of the paper. The paper describes this procedure in its
    Supporting Information; the code is the authoritative detail.
    """
    pos = np.asarray(pos, dtype=np.float32)
    n = len(elements)
    stats = {'proxy_direct': 0, 'proxy_repaired': 0, 'proxy_fail': 0,
             'proxy_added_bonds': 0, 'proxy_removed_bonds': 0,
             'proxy_order_changes': 0, 'proxy_charge_adjustments': 0,
             'proxy_kekule_changes': 0, 'proxy_aromatic_ambiguous': 0,
             'proxy_aromatic_fallback': 0,
             'proxy_confidence': 1.0, 'proxy_conf_atom': 1.0,
             'proxy_conf_topo': 1.0, 'proxy_r_topo': 0.0}
    if bond_type_probs is None:
        bond_type_probs = {}
    edit_supports = []  # posterior support of each REAL topology edit

    # ---- initial edge set: argmax bonds, physics-filtered ----
    edges = []
    for i, j, order in bonds:
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if d < BOND_DMIN or d > BOND_DMAX:
            stats['proxy_removed_bonds'] += 1
            p5 = bond_type_probs.get((min(i, j), max(i, j)))
            if p5 is not None:
                edit_supports.append(float(p5[0]))  # P(no bond)
            continue
        edges.append([i, j, int(order)])

    def incident(a):
        return [e for e in edges if e[0] == a or e[1] == a]

    def vsum(a):
        # Aromatic (4) edges count as single-bond valence here: their pi
        # contribution is decided by the Kekulé pass, not by bond order.
        return sum((1 if e[2] == 4 else e[2]) for e in edges if e[0] == a or e[1] == a)

    def record_remove(a, b):
        stats['proxy_removed_bonds'] += 1
        p5 = bond_type_probs.get((min(a, b), max(a, b)))
        if p5 is not None:
            edit_supports.append(float(p5[0]))  # support of "no bond here"

    def record_order(e):
        stats['proxy_order_changes'] += 1
        p5 = bond_type_probs.get((min(e[0], e[1]), max(e[0], e[1])))
        if p5 is not None:
            edit_supports.append(float(p5[e[2]]))  # support of the new order

    def record_add(a, b, order, support):
        stats['proxy_added_bonds'] += 1
        edit_supports.append(float(support))

    # ---- 2.2/2.3 remove wrong bonds / fix order: over-valence ----
    # N is allowed up to 4 (resolved via formal charge below); others capped
    # at MAX_VALENCE. Posterior support is the tie-breaker for what to touch.
    # Aromatic edges are NEVER touched here — they are resolved by the
    # deterministic Kekulé pass (2.5), otherwise a benzene C (two aromatic
    # edges) would be treated as valence 8 and downgraded to cumulene.
    for _ in range(3):
        changed = False
        for a in range(n):
            mv = 4 if elements[a] == 7 else MAX_VALENCE.get(elements[a], 6)
            while vsum(a) > mv:
                inc = [e for e in incident(a) if e[2] != 4]
                if not inc:
                    break
                inc.sort(key=lambda e: (_p_any_bond(_p5_of(e, bond_type_probs)), -e[2]))
                e = inc[0]
                if e[2] > 1:
                    e[2] -= 1
                    record_order(e)
                else:
                    edges.remove(e)
                    record_remove(e[0], e[1])
                changed = True
        if not changed:
            break

    # ---- 2.5 aromatic -> deterministic Kekulé (representation, not repair) ----
    kstats = _kekulize_aromatic(edges, elements, bond_type_probs)
    stats['proxy_kekule_changes'] += kstats['proxy_kekule_changes']
    stats['proxy_aromatic_ambiguous'] += kstats['proxy_aromatic_ambiguous']
    stats['proxy_aromatic_fallback'] += kstats['proxy_aromatic_fallback']
    c_taut = kstats.get('proxy_tautomer_conf', 1.0)
    # all-single last resort: chemistry is degraded — never push hard
    if kstats.get('proxy_aromatic_fallback', 0) > 0:
        c_taut = min(c_taut, 0.1)
    # write back so guidance_stats.csv reports the REAL tautomer margin
    # (model.py accumulates pstats['proxy_tautomer_conf'])
    stats['proxy_tautomer_conf'] = c_taut

    # ---- 2.4 connectivity: join components with the best legal bridge ----
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(e[0], e[1])
    comps = {}
    for a in range(n):
        comps.setdefault(find(a), []).append(a)
    n_guard = 0
    while len(comps) > 1:
        n_guard += 1
        if n_guard > n * 4:
            stats['proxy_fail'] += 1
            return None
        best = None
        best_score = -1.0
        clist = list(comps)
        for ci in range(len(clist)):
            for cj in range(ci + 1, len(clist)):
                for a in comps[clist[ci]]:
                    for b in comps[clist[cj]]:
                        p5 = bond_type_probs.get((min(a, b), max(a, b)))
                        if p5 is None:
                            continue
                        d = float(np.linalg.norm(pos[a] - pos[b]))
                        # a bridge WE invent must be chemically plausible:
                        # element-dependent covalent ceiling, not a flat 3 A
                        dmax = _covalent_dmax(elements[a], elements[b])
                        if d > dmax:
                            continue
                        # a cross-component bridge can never be aromatic:
                        # only the legal non-aromatic posterior counts
                        p_legal = float(p5[1] + p5[2] + p5[3])
                        if p_legal < 0.05:
                            continue
                        if vsum(a) >= (4 if elements[a] == 7 else MAX_VALENCE.get(elements[a], 6)) \
                                or vsum(b) >= (4 if elements[b] == 7 else MAX_VALENCE.get(elements[b], 6)):
                            continue
                        score = 10.0 * p_legal + 5.0 * (1.0 - d / dmax) + 3.0
                        if score > best_score:
                            best_score = score
                            best = (a, b, p5)
        if best is None:
            stats['proxy_fail'] += 1
            return None
        a, b, p5 = best
        # order: argmax over legal non-aromatic orders, valence-constrained
        mv_a = 4 if elements[a] == 7 else MAX_VALENCE.get(elements[a], 6)
        mv_b = 4 if elements[b] == 7 else MAX_VALENCE.get(elements[b], 6)
        cand_order = [(o, p5[o]) for o in (1, 2, 3)]
        cand_order = [o for o in cand_order if vsum(a) + o[0] <= mv_a and vsum(b) + o[0] <= mv_b]
        if not cand_order:
            stats['proxy_fail'] += 1
            return None
        order = max(cand_order, key=lambda x: x[1])[0]
        edges.append([a, b, order])
        record_add(a, b, order, p5[1] + p5[2] + p5[3])
        union(a, b)
        comps = {}
        for aa in range(n):
            comps.setdefault(find(aa), []).append(aa)

    # ---- 2.6 formal charges from final topology ----
    charges = _charge_adjust_for_rdkit(elements, edges)
    n_charges = int((charges != 0).sum())
    stats['proxy_charge_adjustments'] += n_charges

    # ---- 2.7 RDKit validation loop (deterministic, bounded <= max_iters) ----
    def build_mol():
        from rdkit import Chem
        m = Chem.RWMol()
        for i, el in enumerate(elements):
            a = Chem.Atom(int(el))
            q = int(charges[i])
            if q != 0:
                a.SetFormalCharge(q)
            m.AddAtom(a)
        bmap = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for i, j, order in edges:
            m.AddBond(int(i), int(j), bmap.get(int(order), Chem.BondType.SINGLE))
        mol = m.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    ok = False
    for it in range(max_iters):
        try:
            build_mol()
            ok = True
            break
        except Exception as e:
            emsg = str(e).lower()
            if 'valence' in emsg or 'valency' in emsg:
                fixed = False
                for a in range(n):
                    mv = MAX_VALENCE.get(elements[a], 6)
                    if elements[a] == 7 and charges[a] > 0:
                        mv = 3 + charges[a]
                    if vsum(a) > mv:
                        inc = sorted(incident(a),
                                     key=lambda e: (_p_any_bond(_p5_of(e, bond_type_probs)), -e[2]))
                        for e in inc:
                            if e[2] > 1:
                                e[2] -= 1
                                record_order(e)
                            else:
                                edges.remove(e)
                                record_remove(e[0], e[1])
                            fixed = True
                            break
                        break
                if not fixed:
                    stats['proxy_fail'] += 1
                    return None
            elif 'kekul' in emsg or 'aromatic' in emsg:
                # defensive: no aromatic bond should remain; force singles
                for e in edges:
                    if e[2] == 4:
                        e[2] = 1
                        stats['proxy_kekule_changes'] += 1
            else:
                # unknown failure: drop the weakest-posterior edge and retry
                if not edges:
                    stats['proxy_fail'] += 1
                    return None
                weakest = min(edges, key=lambda e: _p_any_bond(_p5_of(e, bond_type_probs)))
                edges.remove(weakest)
                record_remove(weakest[0], weakest[1])
    if not ok:
        stats['proxy_fail'] += 1
        return None

    # ---- success: proxy confidence from posterior support ----
    out_bonds = [(i, j, o) for i, j, o in edges]
    stats['proxy_direct'] = int(stats['proxy_removed_bonds'] == 0
                                and stats['proxy_added_bonds'] == 0
                                and stats['proxy_order_changes'] == 0
                                and stats['proxy_charge_adjustments'] == 0)
    stats['proxy_repaired'] = 1 - stats['proxy_direct']
    if atom_conf is not None and len(atom_conf) > 0:
        c_atom = _lower_tail_mean(list(np.asarray(atom_conf, dtype=np.float32)))
    else:
        c_atom = 1.0
    # Bond-level support, LAYERED — not one pooled lower-tail:
    #   C_existing: lower-tail of final-bond supports (incl. Kekulé edges).
    #                A bond the model was unsure about (P(single)=0.21 across
    #                a near-uniform posterior) but needed no repair must
    #                still depress confidence — "few edits" is not "model is
    #                sure". Aromatic -> Kekulé conversions are
    #                representation changes: min(1, P(arom)+P(chosen)).
    #   C_edit:     min over REAL edit supports. One critical bridge with
    #                P=0.06 must NOT be diluted by 20 confident bonds.
    #   C_absent:   lower-tail over no-bond halfedges within covalent reach:
    #                if the model cannot decide whether a close pair should
    #                be bonded (P(no)=0.21), the topology is not 96% sure.
    #   C_tautomer: margin between best and second-best heteroatom double
    #                assignment (from the Kekulé solver).
    kekule_keys = {(min(i, j), max(i, j)) for i, j, o in bonds if o == 4}
    bond_scores = []
    for i, j, o in out_bonds:
        key = (min(i, j), max(i, j))
        p5 = bond_type_probs.get(key)
        if p5 is None:
            continue
        if key in kekule_keys:
            bond_scores.append(min(1.0, float(p5[4] + p5[o])))
        else:
            bond_scores.append(float(p5[o]))
    c_existing = _lower_tail_mean(bond_scores) if bond_scores else 1.0
    c_edit = min(edit_supports) if edit_supports else 1.0
    absent_scores = []
    out_keys = {(min(i, j), max(i, j)) for i, j, o in out_bonds}
    for (i, j), p5 in bond_type_probs.items():
        if (i, j) in out_keys:
            continue
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if d <= _covalent_dmax(elements[i], elements[j]):
            absent_scores.append(float(p5[0]))
    c_absent = _lower_tail_mean(absent_scores) if absent_scores else 1.0
    c_topo = min(c_existing, c_edit, c_absent)
    # aromatic protonation ambiguity (unresolvable pi count) lowers trust:
    # a wrong-chemistry proxy must not push at full strength
    n_amb = stats.get('proxy_aromatic_ambiguous', 0)
    if n_amb > 0:
        c_topo *= max(0.4, 1.0 - 0.3 * n_amb)
    n_final = max(len(out_bonds), 1)
    r_topo = (stats['proxy_added_bonds'] + stats['proxy_removed_bonds']) / n_final
    c_proxy = min(c_atom, c_topo, c_taut) * math.exp(-2.0 * r_topo)
    c_proxy = min(max(c_proxy, 0.0), 1.0)
    stats['proxy_confidence'] = c_proxy
    stats['proxy_conf_atom'] = c_atom
    stats['proxy_conf_topo'] = c_topo
    stats['proxy_r_topo'] = r_topo
    return out_bonds, charges, stats


def _add_hydrogens_rdkit(coords, elements, bonds, formal_charges=None):
    """Use RDKit to add hydrogens in 3D to heavy atom coords.

    Builds the molecule DIRECTLY with RWMol — no SDF round-trip. The
    temporary-file I/O of the old path was a measurable chunk of the
    per-guidance-step CPU overhead (a guidance event runs this once per
    proxy). formal_charges (proxy path) are applied when given; the
    exception message still carries valence/kekulize/aromatic keywords so
    the caller's failure classification keeps working.

    Returns:
        coords_h: np.ndarray (N+H, 3)
        elements_h: list of int
        bonds_h: list of (i,j,type)
        n_heavy: int — number of heavy atoms (to strip H charges later)
    """
    from rdkit import Chem
    n_heavy = len(coords)
    try:
        m = Chem.RWMol()
        for i, el in enumerate(elements):
            a = Chem.Atom(int(el))
            if formal_charges is not None and formal_charges[i] != 0:
                a.SetFormalCharge(int(formal_charges[i]))
            m.AddAtom(a)
        bmap = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE}
        for i, j, order in bonds:
            m.AddBond(int(i), int(j), bmap.get(int(order), Chem.BondType.SINGLE))
        mol = m.GetMol()
        Chem.SanitizeMol(mol)
        conf = Chem.Conformer(n_heavy)
        for i in range(n_heavy):
            conf.SetAtomPosition(i, (float(coords[i][0]), float(coords[i][1]),
                                     float(coords[i][2])))
        mol.AddConformer(conf)
        mol_h = Chem.AddHs(mol, addCoords=True)
        cnf = mol_h.GetConformer()
        all_atoms = list(mol_h.GetAtoms())
        n_total = len(all_atoms)
        new_coords = np.zeros((n_total, 3), dtype=np.float32)
        new_elements = []
        for i, atom in enumerate(all_atoms):
            p = cnf.GetAtomPosition(i)
            new_coords[i] = [p.x, p.y, p.z]
            new_elements.append(atom.GetAtomicNum())
        bmap2 = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                 Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
        new_bonds = []
        for b in mol_h.GetBonds():
            new_bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                              bmap2.get(b.GetBondType(), 1)))
        return new_coords, new_elements, new_bonds, n_heavy
    except Exception as e:
        raise RuntimeError(f"rdkit_failed: {e}") from e


# ============================================================
#  MACE force computation
# ============================================================

# Module-level cache for the MACE calculator (loaded once per process)
_mace_calc = None
_mace_device = "cpu"


def init_mace_calculator(model_path=None, device="cuda", default_dtype="float32"):
    """Initialize and cache the MACE calculator. Call once before guidance loop.

    default_dtype: official MACECalculator option ("default dtype of model").
    The MACE-OFF24 weights are float64; float32 inference is officially
    supported, runs ~1.3x faster on consumer GPUs, and preserves force
    directions exactly (measured: direction cosine = 1.000000).
    """
    global _mace_calc, _mace_device
    from mace.calculators import MACECalculator
    if model_path is None:
        model_path = os.path.expanduser("~/.cache/mace/MACE-OFF24_medium.model")
    try:
        _mace_calc = MACECalculator(model_paths=model_path, device=device,
                                    default_dtype=default_dtype)
    except TypeError:
        # older mace without the official option: load float64, then cast
        _mace_calc = MACECalculator(model_paths=model_path, device=device)
        _mace_calc.models[0].float()
    _mace_device = device
    print(f"  MACE calculator loaded: {model_path} on {device} "
          f"(dtype={next(_mace_calc.models[0].parameters()).dtype})")


def mace_force(coords, elements, batch_index=None):
    """Compute MACE atomic forces for a single molecule.

    Args:
        coords:  np.ndarray (N, 3) or torch.Tensor — atom positions in Å
        elements: list of int — atomic numbers
        batch_index: optional, for multi-molecule batch (not used in guidance)

    Returns:
        forces: torch.Tensor (N, 3) — atomic forces in eV/Å
        energy: float — total potential energy in eV
    """
    global _mace_calc
    if _mace_calc is None:
        raise RuntimeError("MACE calculator not initialized. Call init_mace_calculator() first.")

    from ase import Atoms

    # Convert to numpy if needed
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(elements, torch.Tensor):
        elements = elements.detach().cpu().numpy()
    elif isinstance(elements, (list, tuple)):
        elements = np.array(elements, dtype=int)

    atoms = Atoms(numbers=elements.tolist(), positions=coords)
    atoms.calc = _mace_calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()  # (N, 3) in eV/Å

    return torch.from_numpy(forces).float(), float(energy)


def mace_pocket_force_with_h(lig_coords, lig_elements, lig_bonds,
                              pocket_coords, pocket_elements,
                              formal_charges=None):
    """Compute pocket-ligand interaction force with hydrogen addition.

    H added to ligand (not protein), pocket force = F(ligand+H + pocket) - F(ligand+H),
    extracting heavy-atom-only forces.

    formal_charges: optional proxy formal charges passed to hydrogen
    addition (proxy path with [N+] etc.). Return values are unchanged.

    Exceptions carry a prefix so the caller can distinguish RDKit vs MACE failures:
      "rdkit_failed: ..."   hydrogen addition / sanitize failure
      "mace_lig_failed: ..." / "mace_complex_failed: ..."   MACE numerical failure
    """
    # Add H to ligand
    try:
        lig_coords_h, lig_elements_h, lig_bonds_h, n_heavy = _add_hydrogens_rdkit(
            lig_coords, lig_elements, lig_bonds, formal_charges=formal_charges)
    except Exception as e:
        raise RuntimeError(f"rdkit_failed: {e}") from e

    # MACE 1: ligand+H alone
    try:
        F_lig_all, E_lig = mace_force(lig_coords_h, lig_elements_h)
    except Exception as e:
        raise RuntimeError(f"mace_lig_failed: {e}") from e
    F_lig = F_lig_all[:n_heavy]

    # MACE 2: ligand+H + pocket
    # pocket atoms come after ligand heavy atoms but before ligand H atoms
    # Reorder: lig_heavy[0..n_heavy-1] + pocket + lig_H[n_heavy..]
    combined_coords = np.concatenate([lig_coords_h[:n_heavy], pocket_coords,
                                       lig_coords_h[n_heavy:]], axis=0)
    combined_elements = np.concatenate([lig_elements_h[:n_heavy], pocket_elements,
                                         lig_elements_h[n_heavy:]], axis=0)
    try:
        F_total_all, E_total = mace_force(combined_coords, combined_elements)
    except Exception as e:
        raise RuntimeError(f"mace_complex_failed: {e}") from e

    # Extract: first n_heavy heavy ligand atoms, then N_pocket pocket atoms, then H atoms
    F_total_lig = F_total_all[:n_heavy]

    F_inter = F_total_lig - F_lig
    E_inter = E_total - E_lig

    return F_inter, float(E_inter), F_total_lig, F_lig, float(E_total)


# The evaluation scripts (released separately from this repository) call this
# directly to score a single conformation, so it has no caller here.
def mace_energy_only(coords, elements):
    """Compute only MACE total energy (no forces), returns scalar in eV."""
    global _mace_calc
    if _mace_calc is None:
        raise RuntimeError("MACE calculator not initialized.")

    from ase import Atoms
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(elements, torch.Tensor):
        elements = elements.detach().cpu().numpy()
    elif isinstance(elements, (list, tuple)):
        elements = np.array(elements, dtype=int)

    atoms = Atoms(numbers=elements.tolist(), positions=coords)
    atoms.calc = _mace_calc
    return float(atoms.get_potential_energy())


