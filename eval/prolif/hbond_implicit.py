"""ProLIF 隐式 H 氢键计数（ImplicitHBDonor / ImplicitHBAcceptor），只需重原子。

需要 ProLIF >= 2.2，2.0.x 没有这两个 interaction。
这里用重原子口袋 {t}_pocket.pdb（隐式氢不需要加氢）。

输入：{results_root}/{t}/{variant}/*.sdf + {results_root}/{t}/native/{t}_pocket.pdb
输出：{results_root}/{t}/eval/{t}_{variant}_implicit_hbond.csv（file, n）

用法：
    python hbond_implicit.py --target 2wi6 --variants base total split --results-root /path/to/results8
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
        mol = Chem.MolFromMolFile(sf, removeHs=True)  # 隐式 H，只需重原子
        if mol is None:
            return os.path.basename(sf), None
        lig = plf.Molecule.from_rdkit(mol)
        fp = plf.Fingerprint(['ImplicitHBDonor', 'ImplicitHBAcceptor'], count=True,
                             parameters={'ImplicitHBDonor': {'ignore_geometry_checks': True},
                                         'ImplicitHBAcceptor': {'ignore_geometry_checks': True}})
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

    # 这两个 interaction 是 ProLIF 2.2 才加的；装在 2.0.x 上会在 worker 里被
    # try/except 吞掉，最后什么都不输出。先在这里挡一下。
    try:
        plf.Fingerprint(['ImplicitHBDonor', 'ImplicitHBAcceptor'])
    except Exception:
        raise SystemExit(f'本机 ProLIF {plf.__version__} 没有 ImplicitHBDonor / '
                         'ImplicitHBAcceptor，需要 ProLIF >= 2.2')

    t = args.target
    r = os.path.join(args.results_root, t)
    pocket = os.path.join(r, 'native', f'{t}_pocket.pdb')  # 重原子蛋白
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
        out_csv = os.path.join(r, 'eval', f'{t}_{v}_implicit_hbond{suffix}.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['file', 'n'])
            for fn in sorted(res):
                val = res[fn]
                w.writerow([fn, val if val is not None else ''])
        vals = [x for x in res.values() if x is not None]
        if vals:
            print(f'[{t}/{v}] 隐式H n={len(vals)} 氢键均值={np.mean(vals):.2f} '
                  f'→ {os.path.basename(out_csv)}')


if __name__ == '__main__':
    main()
