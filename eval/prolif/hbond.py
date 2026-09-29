"""ProLIF 氢键计数：SMARTS 供体/受体 + 几何判定。口袋用加氢版 {t}_pocket_h.pdb。

输入：{results_root}/{t}/{variant}/*.sdf + {results_root}/{t}/native/{t}_pocket_h.pdb
输出：{results_root}/{t}/eval/{t}_{variant}_prolif_hbond.csv（file, n）

用法：
    python hbond.py --target 2wi6 --variants base total split --results-root /path/to/results8
"""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from rdkit import Chem

DEFAULT_RESULTS8 = os.environ.get(
    'MLIPDIFF_RESULTS8', os.path.join(os.getcwd(), 'results8'))

import prolif as plf

PROT = None  # 全局蛋白（initializer 预构建，fork 继承）


def init_prot(pocket):
    global PROT
    # sanitize=False：部分靶点蛋白有 N 超价（如 1w51），RDKit sanitize 会失败
    PROT = plf.Molecule.from_rdkit(Chem.MolFromPDBFile(pocket, removeHs=False, sanitize=False))


def worker(sf):
    try:
        mol = Chem.AddHs(Chem.MolFromMolFile(sf, removeHs=False), addCoords=True)
        if mol is None:
            return os.path.basename(sf), None
        lig = plf.Molecule.from_rdkit(mol)
        fp = plf.Fingerprint(['HBDonor', 'HBAcceptor'], count=True)
        ifp = fp.generate(lig, PROT)
        n = int(sum(v.sum() for v in ifp.values()))
        return os.path.basename(sf), n
    except Exception:
        return os.path.basename(sf), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--results-root', default=DEFAULT_RESULTS8,
                    help='results8 根目录（默认 $MLIPDIFF_RESULTS8，再退回 ./results8）')
    ap.add_argument('--variants', nargs='+', default=['base', 'total', 'split'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tag', default='', help='输出文件名后缀（新建 csv，不覆盖无 tag 的旧文件）')
    args = ap.parse_args()

    t = args.target
    r = os.path.join(args.results_root, t)
    pocket = os.path.join(r, 'native', f'{t}_pocket_h.pdb')
    if not os.path.isfile(pocket):
        print(f'[SKIP] {t}: 无加氢口袋')
        return
    os.makedirs(os.path.join(r, 'eval'), exist_ok=True)

    for v in args.variants:
        sdf_dir = os.path.join(r, v)
        files = sorted(glob.glob(os.path.join(sdf_dir, '*.sdf')))
        if not files:
            print(f'[SKIP] {t}/{v}: 无 SDF')
            continue
        with Pool(args.workers, initializer=init_prot, initargs=(pocket,)) as pool:
            res = dict(pool.map(worker, files))
        suffix = f'_{args.tag}' if args.tag else ''
        out_csv = os.path.join(r, 'eval', f'{t}_{v}_prolif_hbond{suffix}.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['file', 'n'])
            for fn in sorted(res):
                val = res[fn]
                w.writerow([fn, val if val is not None else ''])
        vals = [x for x in res.values() if x is not None]
        if vals:
            print(f'[{t}/{v}] ProLIF n={len(vals)} 氢键均值={np.mean(vals):.2f} '
                  f'→ {os.path.basename(out_csv)}')


if __name__ == '__main__':
    main()
