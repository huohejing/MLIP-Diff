"""
Vina score_only 打分（+ base 配对）。任意引导目录都能跑，不硬编码变体。

方法：
- 打分：VinaDockingTask score_only，不动构象；过滤 abs(score)<100
- base 配对：先测 index 一致率（同 index 且同 SMILES），>0.9 用 index 对齐，否则 SMILES 配对
- 落盘：{out_dir}/vina_{target}_{variant}.json + .csv

用法：
  # 单变体（显式指定）
  python vina.py --target 3ctj --guided-dir <guided_sdf_dir> --out-dir <out>

  # 覆盖 base / 口袋
  python vina.py --target 1w51 --guided-dir <dir> --base-dir <b> --pocket <p>

base 目录与口袋的默认值按 $MLIPDIFF_ROOT（默认当前目录）解析，可用参数覆盖。
"""
import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docking_vina import VinaDockingTask

# 项目根目录：默认取 $MLIPDIFF_ROOT，再退回当前目录。
# base/口袋的默认路径、以及相对的 --guided-dir 都按它解析（都可用命令行参数覆盖）。
ROOT = os.environ.get('MLIPDIFF_ROOT', os.getcwd())
PROJ = ROOT
DATA = os.path.join(ROOT, 'data24')

# 默认映射（均可被 --base-dir / --pocket 覆盖）
DEFAULT_BASE = {
    '3ctj': 'DATA1000-777/3ctj/3ctj_base/sample_SDF',
    '1w51': 'DATA1000-777/1w51/1w51_base/sample_SDF',
    '5ywy': 'DATA1000-777/5ywy/5ywy_base/sample_SDF',
    '6mey': 'DATA1000-777/6mey/6mey_base/sample_SDF',
}
DEFAULT_POCKET = {
    '3ctj': 'sample/3ctj_pocket.pdb',
    '1w51': 'sample/1w51_pocket_h.pdb',
    '5ywy': 'sample/5ywy_pocket_h.pdb',
    '6mey': 'sample/6mey_pocket_h.pdb',
}


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else smi


def score_file(args):
    f, pocket = args
    try:
        mol = Chem.SDMolSupplier(f, removeHs=False)[0]
        if mol is None:
            return os.path.basename(f), None, None
        task = VinaDockingTask.from_generated_mol(mol, pocket, size_factor=None)
        r = task.run(mode='score_only')
        smi = canon(Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False))
        return os.path.basename(f), smi, r[0]['affinity']
    except Exception:
        return os.path.basename(f), None, None


def score_dir(sdf_dir, pocket, workers):
    files = sorted(glob.glob(os.path.join(sdf_dir, '*.sdf')))
    if not files:
        return []
    with Pool(workers) as pool:
        res = pool.map(score_file, [(f, pocket) for f in files])
    return [r for r in res if r[1] is not None and r[2] is not None and abs(r[2]) < 100]


def pair_index(base_rec, guided_rec, force_smiles=False):
    """返回 (method, pairs)，pairs = [(base_score, guided_score, smiles)]。"""
    b = {os.path.basename(r[0]): r for r in base_rec}
    g = {os.path.basename(r[0]): r for r in guided_rec}
    common = sorted(set(b) & set(g))
    same = sum(1 for i in common if b[i][1] == g[i][1])
    agree = same / len(common) if common else 0
    if agree > 0.9 and not force_smiles:
        return 'index', [(b[i][2], g[i][2], b[i][1]) for i in common]
    # SMILES 配对
    bm = {}
    for fn, s, sc in base_rec:
        bm.setdefault(s, []).append((sc, fn))
    gm = {}
    for fn, s, sc in guided_rec:
        gm.setdefault(s, []).append((sc, fn))
    pairs = []
    for s in sorted(set(bm) & set(gm)):
        bv, gv = sorted(bm[s]), sorted(gm[s])
        for i in range(min(len(bv), len(gv))):
            pairs.append((bv[i][0], gv[i][0], s))
    return 'smiles', pairs


def run_one(target, gdir, base_dir, pocket, out_dir, method, workers):
    if not os.path.isdir(gdir):
        print(f'[SKIP] {gdir}: 目录不存在')
        return
    g_rec = score_dir(gdir, pocket, workers)
    b_rec = score_dir(base_dir, pocket, workers)
    if not g_rec or not b_rec:
        print(f'[WARN] {gdir}: base或guided打分全失败')
        return

    if method == 'index':
        m, pairs = 'index', pair_index(b_rec, g_rec)[1]
    elif method == 'smiles':
        m, pairs = 'smiles', pair_index(b_rec, g_rec, force_smiles=True)[1]
    else:
        m, pairs = pair_index(b_rec, g_rec)

    if not pairs:
        print(f'[WARN] {gdir}: 无配对')
        return
    bv = [p[0] for p in pairs]
    gv = [p[1] for p in pairs]
    diffs = [p[1] - p[0] for p in pairs]
    better = sum(1 for d in diffs if d < 0)

    variant = os.path.basename(gdir.rstrip('/')).replace(target + '_', '')
    os.makedirs(out_dir, exist_ok=True)
    # 逐分子原始 CSV（全精度，不 round）
    out_csv = os.path.join(out_dir, f'vina_{target}_{variant}.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['smiles', 'base_vina', 'guided_vina', 'diff_kcal', 'guided_better'])
        for p in pairs:
            w.writerow([p[2], p[0], p[1], p[1] - p[0], str(p[1] < p[0])])
    out = {
        'target': target, 'variant': variant, 'guided_dir': gdir,
        'base': bv, 'guided': gv, 'method': m,
        'n_paired': len(pairs), 'pct_better': better / len(pairs),
        'mean_delta': float(np.mean(diffs)), 'median_delta': float(np.median(diffs)),
    }
    out_path = os.path.join(out_dir, f'vina_{target}_{variant}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f'[{target}/{variant}] n={len(pairs)} method={m} '
          f'better={better/len(pairs):.1%} Δmean={np.mean(diffs):+.3f} '
          f'Δmed={np.median(diffs):+.3f}')
    print(f'  原始CSV: {os.path.basename(out_csv)} | json: {os.path.basename(out_path)}')


def discover(target):
    """扫 data24 下所有 {target}_*_guided 目录（含 DATADATA 子目录）。"""
    pats = [os.path.join(DATA, f'{target}_*_guided'),
            os.path.join(DATA, 'DATADATA', target, f'{target}_*_guided')]
    dirs = set()
    for p in pats:
        dirs.update(glob.glob(p))
    return sorted(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--guided-dir', help='引导组目录（单变体模式）')
    ap.add_argument('--discover', action='store_true', help='自动发现该靶点所有 *_guided 目录')
    ap.add_argument('--base-dir', default=None, help='覆盖 base 目录（默认按靶点映射）')
    ap.add_argument('--pocket', default=None, help='覆盖口袋 PDB（默认按靶点映射）')
    ap.add_argument('--out-dir', default=DATA)
    ap.add_argument('--method', choices=['auto', 'index', 'smiles'], default='auto')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    base_dir = args.base_dir or os.path.join(DATA, DEFAULT_BASE.get(args.target, ''))
    pocket = args.pocket or os.path.join(ROOT, DEFAULT_POCKET.get(args.target, ''))
    if not os.path.isdir(base_dir):
        print(f'[FATAL] base 目录不存在: {base_dir}（用 --base-dir 指定）')
        return
    if not os.path.exists(pocket):
        print(f'[FATAL] 口袋不存在: {pocket}（用 --pocket 指定）')
        return

    gdirs = []
    if args.discover:
        gdirs = discover(args.target)
        print(f'发现 {args.target} 引导目录 {len(gdirs)} 个:')
        for d in gdirs:
            print(f'  {d}')
    elif args.guided_dir:
        gdirs = [args.guided_dir if os.path.isabs(args.guided_dir)
                 else os.path.join(PROJ, args.guided_dir)]
    else:
        print('[FATAL] 需 --guided-dir 或 --discover')
        return

    for g in gdirs:
        run_one(args.target, g, base_dir, pocket, args.out_dir, args.method, args.workers)


if __name__ == '__main__':
    main()
