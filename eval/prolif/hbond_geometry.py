"""ProLIF 氢键几何：逐根氢键的 distance / DHA_angle。口袋用加氢版 {t}_pocket_h.pdb。

fp.generate(liga, prot, metadata=True) 才返回逐根氢键的距离和角度，否则只有计数。

输入：{results_root}/{t}/{variant}/*.sdf + {results_root}/{t}/native/{t}_pocket_h.pdb
输出：{results_root}/{t}/eval/{t}_{variant}_prolif_hbond_geom_{tag}.csv
      列 file, dist, DHA_angle；一行一根氢键

用法：
    python hbond_geometry.py --target 5ni7 --variants base total split --results-root /path/to/results8
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
    """返回 [(文件名, dist, angle), ...]，该分子检出的每根氢键一行。"""
    try:
        mol = Chem.AddHs(Chem.MolFromMolFile(sf, removeHs=False), addCoords=True)
        if mol is None:
            return []
        lig = plf.Molecule.from_rdkit(mol)
        fp = plf.Fingerprint(['HBDonor', 'HBAcceptor'], count=True)
        ifp = fp.generate(lig, PROT, metadata=True)
        base = os.path.basename(sf)
        rows = []
        for _key, by_inter in ifp.items():
            for _inter, metas in by_inter.items():
                for md in metas:                      # 每根氢键一个 dict
                    d = md.get('distance')
                    a = md.get('DHA_angle')
                    if d is not None:
                        rows.append((base, float(d), float(a) if a is not None else ''))
        return rows
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--results-root', default=DEFAULT_RESULTS8,
                    help='results8 根目录（默认 $MLIPDIFF_RESULTS8，再退回 ./results8）')
    ap.add_argument('--variants', nargs='+', default=['base', 'total', 'split'])
    ap.add_argument('--workers', type=int, default=2, help='默认 2，省内存')
    ap.add_argument('--tag', default='v2')
    args = ap.parse_args()

    t = args.target
    r = os.path.join(args.results_root, t)
    pocket = os.path.join(r, 'native', f'{t}_pocket_h.pdb')
    if not os.path.isfile(pocket):
        print(f'[SKIP] {t}: 无加氢口袋')
        return
    os.makedirs(os.path.join(r, 'eval'), exist_ok=True)

    for v in args.variants:
        files = sorted(glob.glob(os.path.join(r, v, '*.sdf')))
        if not files:
            print(f'[SKIP] {t}/{v}: 无 SDF')
            continue
        with Pool(args.workers, initializer=init_prot, initargs=(pocket,),
                  maxtasksperchild=40) as pool:
            res = pool.map(worker, files, chunksize=4)
        rows = [x for sub in res for x in sub]
        out = os.path.join(r, 'eval',
                           f'{t}_{v}_prolif_hbond_geom_{args.tag}.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['file', 'dist', 'DHA_angle'])
            for row in rows:
                w.writerow(row)
        ds = [x[1] for x in rows]
        n_mol = sum(1 for sub in res if sub)
        if ds:
            print(f'[{t}/{v}] {n_mol} 个分子有成键，共 {len(ds)} 根氢键  '
                  f'dist 均值={np.mean(ds):.3f} Å  '
                  f'范围 {min(ds):.2f}~{max(ds):.2f}  → {os.path.basename(out)}')
        else:
            print(f'[{t}/{v}] 无氢键 → {os.path.basename(out)}')


if __name__ == '__main__':
    main()
