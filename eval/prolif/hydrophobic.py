"""ProLIF 疏水接触计数（Hydrophobic，count=True）。

疏水是重原子接触，口袋用重原子版 {t}_pocket.pdb 即可。

输入：{results_root}/{t}/{variant}/*.sdf + {results_root}/{t}/native/{t}_pocket.pdb
输出：{results_root}/{t}/eval/{t}_{variant}_prolif_hydrophobic.csv（file, n）

用法：
    python hydrophobic.py --target 5ni7 --results-root /path/to/results8
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

PROT = None


def init_prot(pocket):
    global PROT
    PROT = plf.Molecule.from_rdkit(
        Chem.MolFromPDBFile(pocket, removeHs=False, sanitize=False))


def worker(sf):
    try:
        mol = Chem.MolFromMolFile(sf, removeHs=True)  # 疏水重原子即可
        if mol is None:
            return os.path.basename(sf), None
        lig = plf.Molecule.from_rdkit(mol)
        fp = plf.Fingerprint(['Hydrophobic'], count=False)  # 疏水用每残基对布尔，count=True 会膨胀
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
    ap.add_argument('--tag', default='', help='输出文件名后缀（新建 csv）')
    args = ap.parse_args()

    t = args.target
    r = os.path.join(args.results_root, t)
    pocket = os.path.join(r, 'native', f'{t}_pocket.pdb')
    if not os.path.isfile(pocket):
        print(f'[SKIP] {t}: 无口袋')
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
        out_csv = os.path.join(r, 'eval', f'{t}_{v}_prolif_hydrophobic{suffix}.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['file', 'n'])
            for fn in sorted(res):
                val = res[fn]
                w.writerow([fn, val if val is not None else ''])
        vals = [x for x in res.values() if x is not None]
        if vals:
            print(f'[{t}/{v}] ProLIF疏水 n={len(vals)} 均值={np.mean(vals):.2f} '
                  f'→ {os.path.basename(out_csv)}')


if __name__ == '__main__':
    main()
