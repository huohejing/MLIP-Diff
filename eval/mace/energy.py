"""MACE 能量评估：配体自能 E_lig、复合物总能 E_total、相互作用能 E_int = E_total - E_lig - E_pocket。

配体用 RDKit 加氢（Chem.AddHs(addCoords=True)），口袋从 full_pocket.npz 读
（由 scripts/prepare_target_mace.py 生成）。基线组与引导组按 SMILES 配对，
同一 SMILES 组内按顺序一一对应，逐对输出两侧能量和差值。

输入：{root}/validation/{target}/full_pocket.npz
      {root}/validation/{target}/baseline/*_SDF/*.sdf
      {root}/validation/{target}/scale_*/*_SDF/*.sdf
输出：{root}/validation/{target}_full_eval_off24.csv

用法：
    python energy.py --target 3ctj_split
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from rdkit import Chem

# 仓库根，用于 import models.physical_guidance
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.physical_guidance import init_mace_calculator, mace_energy_only  # noqa: E402

DEFAULT_ROOT = os.environ.get('MLIPDIFF_ROOT', os.getcwd())


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else smi


def coords_elements(sf):
    """SDF → (坐标 (N,3) float32, 元素 (N,) int32)，加氢后的。"""
    mol = Chem.SDMolSupplier(sf, removeHs=False)[0]
    if mol is None:
        return None, None
    h = Chem.AddHs(mol, addCoords=True)
    cf = h.GetConformer()
    n = h.GetNumAtoms()
    c = np.array([[cf.GetAtomPosition(i).x, cf.GetAtomPosition(i).y, cf.GetAtomPosition(i).z]
                  for i in range(n)], dtype=np.float32)
    return c, np.array([a.GetAtomicNum() for a in h.GetAtoms()], dtype=np.int32)


def smiles_of(sf):
    mol = Chem.SDMolSupplier(sf, removeHs=False)[0]
    if mol is None:
        return None
    return canon(Chem.MolToSmiles(Chem.RemoveHs(Chem.AddHs(mol))))


def collect_by_smiles(sdf_dir):
    """{smiles: [sdf_path, ...]}，保留同一 SMILES 的全部实例，按文件名排序。"""
    by_smi = defaultdict(list)
    for sf in sorted(glob.glob(os.path.join(sdf_dir, '*.sdf'))):
        s = smiles_of(sf)
        if s:
            by_smi[s].append(sf)
    return by_smi


def find_sdf_dir(root):
    dirs = sorted(glob.glob(f'{root}/*_SDF') + glob.glob(f'{root}/**/*_SDF', recursive=True))
    return dirs[-1] if dirs else None


def main():
    ap = argparse.ArgumentParser(description='MACE 能量评估')
    ap.add_argument('--target', required=True, help='{root}/validation/ 下的目录名')
    ap.add_argument('--root', default=DEFAULT_ROOT,
                    help='项目根目录（默认 $MLIPDIFF_ROOT，再退回当前目录）')
    ap.add_argument('--mace-model', default='~/.cache/mace/MACE-OFF24_medium.model')
    ap.add_argument('--device', default=None, help='cuda 或 cpu，默认自动')
    ap.add_argument('--mace-dtype', default='float32', help='MACE 推理精度')
    args = ap.parse_args()

    target_dir = os.path.join(args.root, 'validation', args.target)
    npz = os.path.join(target_dir, 'full_pocket.npz')
    if not os.path.isfile(npz):
        raise SystemExit(f'缺少口袋文件 {npz}，先用 scripts/prepare_target_mace.py 生成')

    device = args.device
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_mace_calculator(device=device, model_path=os.path.expanduser(args.mace_model),
                         default_dtype=args.mace_dtype)

    pp = np.load(npz)
    pc, pe = pp['coords'], pp['elements']
    Ep = mace_energy_only(pc, pe)
    print(f'{args.target}: E_pocket = {Ep:.1f} eV ({len(pc)} 原子)')

    def e_total(sf):
        c, el = coords_elements(sf)
        if c is None:
            return None
        return mace_energy_only(np.concatenate([c, pc], axis=0),
                                np.concatenate([el, pe], axis=0))

    def e_lig(sf):
        c, el = coords_elements(sf)
        if c is None:
            return None
        return mace_energy_only(c, el)

    b_dir = find_sdf_dir(os.path.join(target_dir, 'baseline'))
    if not b_dir:
        raise SystemExit(f'{args.target}: 找不到 baseline 的 SDF 目录')
    b_by_smi = collect_by_smiles(b_dir)
    print(f'{args.target}: baseline {len(b_by_smi)} 个 SMILES，'
          f'{sum(len(v) for v in b_by_smi.values())} 个分子')

    guided_dirs = {}
    for d in sorted(glob.glob(f'{target_dir}/scale_*/')):
        g = find_sdf_dir(d)
        if g:
            guided_dirs[os.path.basename(d.rstrip('/')).replace('scale_', '')] = g
    print(f'{args.target}: 引导组 {sorted(guided_dirs)}')

    rows = []
    for scale, g_dir in sorted(guided_dirs.items()):
        group = f'{args.target} dir{scale}'
        g_by_smi = collect_by_smiles(g_dir)
        matched = 0
        for smi in sorted(set(b_by_smi) & set(g_by_smi)):
            for i in range(min(len(b_by_smi[smi]), len(g_by_smi[smi]))):
                bsf, gsf = b_by_smi[smi][i], g_by_smi[smi][i]
                mat = Chem.SDMolSupplier(bsf, removeHs=False)[0]
                bet, get_ = e_total(bsf), e_total(gsf)
                bel, gel = e_lig(bsf), e_lig(gsf)
                bei = bet - bel - Ep if bet is not None and bel is not None else None
                gei = get_ - gel - Ep if get_ is not None and gel is not None else None
                matched += 1
                rows.append({
                    'group': group,
                    'smiles': smi,
                    'n_heavy': mat.GetNumAtoms() if mat else '',
                    'base_E_total_eV': round(bet, 1) if bet else '',
                    'guided_E_total_eV': round(get_, 1) if get_ else '',
                    'diff_E_total_eV': round(bet - get_, 3) if bet and get_ else '',
                    'E_total_better': (bet - get_ > 0) if bet and get_ else '',
                    'base_E_lig_eV': round(bel, 3) if bel else '',
                    'guided_E_lig_eV': round(gel, 3) if gel else '',
                    'diff_E_lig_eV': round(bel - gel, 3) if bel and gel else '',
                    'E_lig_better': (bel - gel > 0) if bel and gel else '',
                    'base_E_int_eV': round(bei, 3) if bei else '',
                    'guided_E_int_eV': round(gei, 3) if gei else '',
                    'diff_E_int_eV': round(bei - gei, 3) if bei and gei else '',
                    'E_int_better': (bei - gei > 0) if bei and gei else '',
                })
        n_total = len(glob.glob(f'{g_dir}/*.sdf'))
        print(f'  {group}: {matched} 对 / {n_total} 个生成分子')

    fields = ['group', 'smiles', 'n_heavy',
              'base_E_total_eV', 'guided_E_total_eV', 'diff_E_total_eV', 'E_total_better',
              'base_E_lig_eV', 'guided_E_lig_eV', 'diff_E_lig_eV', 'E_lig_better',
              'base_E_int_eV', 'guided_E_int_eV', 'diff_E_int_eV', 'E_int_better']
    out = os.path.join(os.path.dirname(target_dir), f'{args.target}_full_eval_off24.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'已保存: {out} ({len(rows)} 行)')

    for g in sorted(set(r['group'] for r in rows)):
        gr = [r for r in rows if r['group'] == g]
        n = len(gr)
        n_tot = sum(1 for r in gr if r['E_total_better'] is True)
        n_lig = sum(1 for r in gr if r['E_lig_better'] is True)
        n_int = sum(1 for r in gr if r['E_int_better'] is True)
        print(f'  {g}: n={n}  E_total 更低 {n_tot}/{n}  E_lig 更低 {n_lig}/{n}  E_int 更低 {n_int}/{n}')


if __name__ == '__main__':
    main()
