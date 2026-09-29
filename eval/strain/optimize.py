#!/usr/bin/env python3
"""MACE 两阶段几何优化 —— 算构象应变能。

Stage 1: 固定重原子（有口袋时连口袋一起固定），只松 H → E_gen
Stage 2: 放开配体全部原子（口袋仍固定）→ E_opt
ΔE = E_gen - E_opt，即构象离 MACE 最低点的距离；RMSD 是 Kabsch 对齐后的重原子 RMSD。

用法:
    python optimize.py --sdf mol.sdf --model MACE-OFF24_medium.model
    python optimize.py --base base_sdf/ --guided guided_sdf/ --model MACE-OFF24_medium.model
    python optimize.py --base base_sdf/ --guided guided_sdf/ \
        --pocket 3ctj_pocket_h.pdb --model MACE-OFF24_medium.model

--pocket 传加氢口袋（与 MACE 能量评估同一套）。
"""
import argparse, os, sys, csv, time
import numpy as np
from rdkit import Chem
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
import warnings
warnings.filterwarnings("ignore")


def kabsch_rmsd(P, Q):
    """Kabsch-aligned heavy-atom RMSD."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.eye(3); D[-1, -1] = np.sign(d)
    R = Vt.T @ D @ U.T
    P_aligned = P @ R.T
    return np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))


def pdb_to_atoms(pdb_path):
    """PDB → (symbols, positions) for pocket atoms (heavy atoms only)."""
    ELEM = {'H': 'H', 'C': 'C', 'N': 'N', 'O': 'O', 'S': 'S', 'P': 'P', 'F': 'F',
            'CL': 'Cl', 'BR': 'Br', 'I': 'I'}
    symbols, positions = [], []
    with open(pdb_path) as f:
        for l in f:
            if l.startswith('ATOM') or l.startswith('HETATM'):
                # element: prefer the element column (76:78), fall back to the
                # atom name with a numeric prefix stripped (1HB -> HB).
                e = l[76:78].strip().upper()
                if not e:
                    name = l[12:16].strip().upper()
                    if name and name[0].isdigit():
                        name = name[1:]
                    e = name
                if e.startswith('H') and len(e) <= 2:
                    sym = 'H'          # keep pocket hydrogens (protonated pocket)
                else:
                    sym = ELEM.get(e)
                if sym is None:
                    continue
                symbols.append(sym)
                positions.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return symbols, np.array(positions, dtype=np.float32)


def pdb_to_pocket_full(pdb_path):
    """PDB -> (symbols, positions, resids) for the FULL pocket (keeps H)."""
    ELEM = {'H': 'H', 'C': 'C', 'N': 'N', 'O': 'O', 'S': 'S', 'P': 'P', 'F': 'F',
            'CL': 'Cl', 'BR': 'Br', 'I': 'I'}
    symbols, positions, resids = [], [], []
    with open(pdb_path) as f:
        for l in f:
            if l.startswith('ATOM') or l.startswith('HETATM'):
                e = l[76:78].strip().upper()
                if not e:
                    name = l[12:16].strip().upper()
                    if name and name[0].isdigit():
                        name = name[1:]
                    e = name
                if e.startswith('H') and len(e) <= 2:
                    sym = 'H'
                else:
                    sym = ELEM.get(e)
                if sym is None:
                    continue
                symbols.append(sym)
                positions.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                resids.append(l[22:27].strip())
    return symbols, np.array(positions, dtype=np.float32), resids


def sdf_to_atoms(sf):
    """SDF → ASE Atoms with explicit H. Adds H via RDKit if SDF has none."""
    mol = Chem.SDMolSupplier(sf, removeHs=False)[0]
    if mol is None:
        return None, None, None
    # Add H if molecule doesn't have explicit H
    mol_h = Chem.AddHs(mol, addCoords=True)
    conf = mol_h.GetConformer()
    elements = [atom.GetSymbol() for atom in mol_h.GetAtoms()]
    positions = np.array([conf.GetAtomPosition(i) for i in range(mol_h.GetNumAtoms())])
    # Heavy atom indices (Z > 1)
    heavy_idx = [i for i, atom in enumerate(mol_h.GetAtoms()) if atom.GetAtomicNum() > 1]
    return Atoms(symbols=elements, positions=positions), heavy_idx, mol_h


def optimize_two_stage(atoms, heavy_idx, calc, pocket_indices=None, steps2=300):
    """Two-stage relaxation.

    Without pocket:
      Stage 1: fix heavy, relax H → E_gen. Stage 2: relax all → E_opt.
    With pocket (pocket_indices = indices of pocket atoms):
      Pocket atoms fixed throughout.
      Stage 1: fix ligand heavy + pocket → relax ligand H → E_gen.
      Stage 2: fix pocket → relax all ligand atoms → E_opt.
    """
    atoms.calc = calc

    if pocket_indices is not None:
        fixed1 = sorted(set(heavy_idx) | set(pocket_indices))
    else:
        fixed1 = heavy_idx

    # Stage 1: relax ligand H only
    atoms.set_constraint(FixAtoms(indices=fixed1))
    opt_h = LBFGS(atoms, logfile=None)
    opt_h.run(fmax=0.05, steps=50)
    E_gen = atoms.get_potential_energy()
    heavy_gen = atoms.positions[heavy_idx].copy()
    n_steps_h = opt_h.nsteps

    # Stage 2: full ligand relaxation (pocket still fixed if present)
    if pocket_indices is not None:
        atoms.set_constraint(FixAtoms(indices=pocket_indices))
    else:
        atoms.set_constraint()
    opt_all = LBFGS(atoms, logfile=None)
    opt_all.run(fmax=0.02, steps=steps2)
    E_opt = atoms.get_potential_energy()
    heavy_opt = atoms.positions[heavy_idx].copy()
    n_steps_all = opt_all.nsteps
    converged_all = opt_all.converged()
    final_fmax = np.max(np.abs(atoms.get_forces()))

    rmsd = kabsch_rmsd(heavy_gen, heavy_opt)
    return E_gen, E_opt, rmsd, n_steps_h, n_steps_all, converged_all, final_fmax


def process_single(sf, calc, pocket_data=None, pocket_full=None, pocket_cutoff=5.0, steps2=300):
    """处理单个 SDF，返回 dict（success=True/False）

    pocket_data: (symbols, positions) 或 None。若提供，配体+口袋合并松弛，口袋固定。
    pocket_full: (symbols, positions, resids) 全加氢口袋 + 残基。
                 若提供，按当前配体坐标动态截取 pocket_cutoff A 内残基
                 （per-molecule pocket，与引导口径一致）。
    """
    result = {"sdf": os.path.basename(sf), "success": False}
    try:
        atoms, heavy_idx, mol = sdf_to_atoms(sf)
        if atoms is None:
            result["error"] = "SDF parse failed"
            return result
        if len(heavy_idx) < 3:
            result["error"] = f"too few heavy atoms ({len(heavy_idx)})"
            return result

        pocket_indices = None
        n_pocket = 0
        if pocket_full is not None:
            p_sym, p_pos, p_resids = pocket_full
            lig_pos = atoms.positions[heavy_idx]
            d = np.linalg.norm(p_pos[:, None, :] - lig_pos[None, :, :], axis=2)
            min_d = d.min(axis=1)
            nearby = set(np.unique([p_resids[i] for i in range(len(p_resids)) if min_d[i] < pocket_cutoff]))
            if not nearby:
                # fallback: nearest residue only
                nearest = p_resids[int(np.argmin(min_d))]
                nearby = {nearest}
            mask = np.array([r in nearby for r in p_resids], dtype=bool)
            p_sym_sub = [s for s, m in zip(p_sym, mask) if m]
            p_pos_sub = p_pos[mask]
            n_lig = len(atoms)
            atoms = atoms + Atoms(symbols=p_sym_sub, positions=p_pos_sub)
            pocket_indices = list(range(n_lig, n_lig + len(p_sym_sub)))
            n_pocket = len(p_sym_sub)
        elif pocket_data is not None:
            p_sym, p_pos = pocket_data
            n_lig = len(atoms)
            atoms = atoms + Atoms(symbols=p_sym, positions=p_pos)
            pocket_indices = list(range(n_lig, n_lig + len(p_sym)))
            n_pocket = len(p_sym)

        n_heavy = len(heavy_idx)
        t0 = time.time()
        E_gen, E_opt, rmsd, n_h, n_all, converged, final_fmax = optimize_two_stage(
            atoms, heavy_idx, calc, pocket_indices=pocket_indices, steps2=steps2)
        elapsed = time.time() - t0

        E_diff_eV = E_gen - E_opt
        E_diff_kcal = E_diff_eV * 23.0605

        result.update({
            "success": True,
            "n_heavy": n_heavy,
            "n_pocket": n_pocket,
            "E_gen_eV": round(E_gen, 4),
            "E_opt_eV": round(E_opt, 4),
            "E_diff_eV": round(E_diff_eV, 4),
            "E_diff_kcal": round(E_diff_kcal, 2),
            "E_diff_per_heavy_kcal": round(E_diff_kcal / n_heavy, 2),
            "opt_rmsd_A": round(rmsd, 4),
            "steps_h": n_h,
            "steps_all": n_all,
            "converged": converged,
            "final_fmax": round(final_fmax, 4),
            "time_s": round(elapsed, 2),
        })
    except Exception as e:
        result["error"] = repr(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="MACE two-stage geometry optimization")
    parser.add_argument("--sdf", type=str, help="单个 SDF 文件")
    parser.add_argument("--dir", type=str, help="SDF 目录")
    parser.add_argument("--base", type=str, help="baseline SDF 目录")
    parser.add_argument("--guided", type=str, help="guided SDF 目录")
    parser.add_argument("--pocket", type=str, default=None, help="口袋 PDB（口袋内应变模式）")
    parser.add_argument("--model", type=str, required=True, help="MACE 模型路径")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"],
                        help="MACE 推理精度：float32 更快(~1.5-1.9x)，float64 更准（官方推荐优化用）")
    parser.add_argument("--steps2", type=int, default=300,
                        help="Stage 2 全松弛最大步数（默认300；不收敛会撞上限）")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading MACE model: {args.model}")
    print(f"dtype={args.dtype}, Stage2 max_steps={args.steps2}")
    calc = MACECalculator(model_paths=[args.model], device=args.device,
                          default_dtype=args.dtype)
    pocket_data = None
    pocket_full = None
    if args.pocket:
        p_sym, p_pos, p_resids = pdb_to_pocket_full(args.pocket)
        pocket_full = (p_sym, p_pos, p_resids)
        print(f"Pocket: {args.pocket} (full {len(p_sym)} atoms incl. H; "
              f"per-molecule 5.0 A residue cutoff, fixed during relaxation)")
    print(f"Device: {args.device}, two-stage LBFGS (H relax + full relax)")

    all_results = []

    # 单文件
    if args.sdf:
        r = process_single(args.sdf, calc, pocket_data, pocket_full=pocket_full,
                           steps2=args.steps2)
        all_results.append(r)
        if r.get("success"):
            print(f"  {r['sdf']}: ΔE={r['E_diff_kcal']} kcal/mol, rmsd={r['opt_rmsd_A']:.3f} Å, "
                  f"{r['steps_h']}+{r['steps_all']} steps, {r['time_s']}s")
        else:
            print(f"  {r['sdf']}: FAILED - {r.get('error', 'unknown')}")

    # 批量
    if args.dir:
        sdfs = sorted([f for f in os.listdir(args.dir) if f.endswith(".sdf")])
        if args.max: sdfs = sdfs[:args.max]
        print(f"Processing {len(sdfs)} SDFs from {args.dir}...")
        t0 = time.time()
        for i, sf in enumerate(sdfs):
            r = process_single(os.path.join(args.dir, sf), calc, pocket_data, pocket_full=pocket_full,
                               steps2=args.steps2)
            all_results.append(r)
            if (i + 1) % 50 == 0:
                n_ok = sum(1 for rr in all_results if rr.get("success"))
                print(f"  {i+1}/{len(sdfs)} ({(time.time()-t0)/(i+1):.1f}s/mol, {n_ok} ok)")
        n_ok = sum(1 for r in all_results if r.get("success"))
        print(f"Done: {n_ok}/{len(sdfs)} success in {time.time()-t0:.0f}s")

    # 配对比较 (SMILES-matched)
    if args.base and args.guided:
        from collections import defaultdict

        def canon_smi(sf):
            mol = Chem.SDMolSupplier(sf, removeHs=False)[0]
            if mol is None: return None
            return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)

        base_by_smi = defaultdict(list)
        for sf in sorted([f for f in os.listdir(args.base) if f.endswith(".sdf")]):
            smi = canon_smi(os.path.join(args.base, sf))
            if smi: base_by_smi[smi].append(sf)

        guided_by_smi = defaultdict(list)
        for sf in sorted([f for f in os.listdir(args.guided) if f.endswith(".sdf")]):
            smi = canon_smi(os.path.join(args.guided, sf))
            if smi: guided_by_smi[smi].append(sf)

        common_smi = sorted(set(base_by_smi.keys()) & set(guided_by_smi.keys()))
        n_pairs = sum(min(len(base_by_smi[s]), len(guided_by_smi[s])) for s in common_smi)
        if args.max: n_pairs = min(n_pairs, args.max)
        print(f"SMILES-matched: {len(common_smi)} shared, {n_pairs} pairs...")

        t0 = time.time()
        base_results, guided_results, paired_results = [], [], []
        paired = 0
        for smi in common_smi:
            for b_idx in range(len(base_by_smi[smi])):
                if args.max and paired >= args.max: break
                if b_idx >= len(guided_by_smi[smi]): break
                bsf = base_by_smi[smi][b_idx]
                gsf = guided_by_smi[smi][b_idx]
                br = process_single(os.path.join(args.base, bsf), calc, pocket_data, pocket_full=pocket_full,
                                    steps2=args.steps2)
                gr = process_single(os.path.join(args.guided, gsf), calc, pocket_data, pocket_full=pocket_full,
                                    steps2=args.steps2)
                if br.get("success") and gr.get("success"):
                    br["smiles"] = smi; gr["smiles"] = smi
                    base_results.append(br); guided_results.append(gr)
                    paired_results.append({
                        "smiles": smi,
                        "n_heavy": br["n_heavy"],
                        "base_E_diff_kcal": br["E_diff_kcal"], "guided_E_diff_kcal": gr["E_diff_kcal"],
                        "delta_E_improvement_kcal": round(br["E_diff_kcal"] - gr["E_diff_kcal"], 2),
                        "base_E_diff_per_heavy": br["E_diff_per_heavy_kcal"],
                        "guided_E_diff_per_heavy": gr["E_diff_per_heavy_kcal"],
                        "base_rmsd_A": br["opt_rmsd_A"], "guided_rmsd_A": gr["opt_rmsd_A"],
                        "base_converged": br["converged"], "guided_converged": gr["converged"],
                        "base_final_fmax": br["final_fmax"], "guided_final_fmax": gr["final_fmax"],
                        "base_steps_all": br["steps_all"], "guided_steps_all": gr["steps_all"],
                    })
                    paired += 1
                if paired % 50 == 0:
                    print(f"  {paired}/{n_pairs} ({(time.time()-t0)/paired:.1f}s/pair)")

        # Failure stats
        n_total = paired
        n_success = len(base_results)
        n_failed = n_total - n_success
        print(f"\n  Success: {n_success}/{n_total}, Failed/NaN: {n_failed} ({100*n_failed/max(1,n_total):.1f}%)")

        if base_results:
            diffs = [b["E_diff_kcal"] - g["E_diff_kcal"] for b, g in zip(base_results, guided_results)]
            better = sum(1 for d in diffs if d > 0)
            print(f"\nPaired ({n_success} pairs, {time.time()-t0:.0f}s):")
            print(f"  Base mean ΔE:     {np.mean([r['E_diff_kcal'] for r in base_results]):.2f} kcal/mol")
            print(f"  Guided mean ΔE:   {np.mean([r['E_diff_kcal'] for r in guided_results]):.2f} kcal/mol")
            print(f"  Mean ΔΔE (b-g):   {np.mean(diffs):.2f} kcal/mol (+ = guided better)")
            print(f"  Better: {better}/{len(diffs)} ({100*better/len(diffs):.0f}%)")
            print(f"  Base mean rmsd:  {np.mean([r['opt_rmsd_A'] for r in base_results]):.3f} Å")
            print(f"  Guided mean rmsd:{np.mean([r['opt_rmsd_A'] for r in guided_results]):.3f} Å")
            cb = sum(1 for r in base_results if r["converged"])
            cg = sum(1 for r in guided_results if r["converged"])
            print(f"  Base converged:  {cb}/{len(base_results)}")
            print(f"  Guided converged:{cg}/{len(guided_results)}")
            fb = np.mean([r["final_fmax"] for r in base_results])
            fg = np.mean([r["final_fmax"] for r in guided_results])
            print(f"  Base mean fmax:  {fb:.4f} eV/Å")
            print(f"  Guided mean fmax:{fg:.4f} eV/Å")

            all_results = paired_results

    # CSV
    if all_results:
        if args.output:
            out_path = args.output
        elif args.dir:
            out_path = os.path.join(args.dir, "mace_opt_results.csv")
        elif args.sdf:
            out_path = os.path.splitext(args.sdf)[0] + "_mace_opt.csv"
        else:
            out_path = "mace_opt_results.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader(); w.writerows(all_results)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
