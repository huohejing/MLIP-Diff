"""ProLIF 其他相互作用计数：pistack / saltbridge / halogen / pication / vdw。

接触型用 count=False（每残基对布尔）。口袋用加氢版 {t}_pocket_h.pdb —— 芳香性
感知需要蛋白被正常 sanitize，否则 PiStacking 不会命中。

输入：{results_root}/{t}/{variant}/*.sdf + {results_root}/{t}/native/{t}_pocket_h.pdb
输出：{results_root}/{t}/eval/{t}_{variant}_prolif_{contact}.csv（file, n）

用法：
    python contacts.py --target 5ni7 --contact pistack --results-root /path/to/results8
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

# 接触类型 → ProLIF 交互名
CONTACTS = {
    'pistack': ['PiStacking'],
    'saltbridge': ['Anionic', 'Cationic'],
    'halogen': ['XBAcceptor', 'XBDonor'],
    'pication': ['PiCation', 'CationPi'],
    'vdw': ['VdWContact'],
}

PROT = None


def load_protein(pocket):
    """读口袋，**保留芳香性 / 环信息**。

    ⚠️ 不能直接 sanitize=True：加氢口袋 {t}_pocket_h.pdb 含高价原子，
       RDKit 完整 sanitize 会失败并返回 None（1w51 实测）。
    ⚠️ 也不能只 sanitize=False：那样芳香原子数 = 0、环信息不初始化，
       PiStacking / PiCation 恒为 0（这就是原来那个 bug）。
    → 折中：跳过 SANITIZE_PROPERTIES（价键检查），其余照做。
      芳香性与环信息正常感知（实测每靶点芳香原子 64~119 个）。
    """
    m = Chem.MolFromPDBFile(pocket, removeHs=False, sanitize=False)
    if m is None:
        return None
    Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                    Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    return plf.Molecule.from_rdkit(m)


def init_prot(pocket):
    global PROT
    PROT = load_protein(pocket)


def worker(args):
    sf, contact = args
    try:
        mol = Chem.MolFromMolFile(sf, removeHs=True)  # 重原子
        if mol is None:
            return os.path.basename(sf), None
        lig = plf.Molecule.from_rdkit(mol)
        fp = plf.Fingerprint(CONTACTS[contact], count=False)  # 残基对级
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
    ap.add_argument('--contact', required=True, choices=list(CONTACTS))
    ap.add_argument('--variants', nargs='+', default=['base', 'total', 'split'])
    ap.add_argument("--workers", type=int, default=2, help="默认 2，省内存")
    ap.add_argument('--tag', default='', help='输出文件名后缀')
    args = ap.parse_args()

    t = args.target
    r = os.path.join(args.results_root, t)
    # ★ 用加氢口袋（与氢键脚本一致）；无氢版作为兜底
    pocket = os.path.join(r, 'native', f'{t}_pocket_h.pdb')
    if not os.path.isfile(pocket):
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
        # maxtasksperchild：每个 worker 处理 N 个分子后重启，避免 ProLIF 累积内存（防 OOM）
        with Pool(args.workers, initializer=init_prot, initargs=(pocket,),
                  maxtasksperchild=40) as pool:
            res = dict(pool.map(worker, [(sf, args.contact) for sf in files],
                                chunksize=4))
        suffix = f'_{args.tag}' if args.tag else ''
        out_csv = os.path.join(r, 'eval', f'{t}_{v}_prolif_{args.contact}{suffix}.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['file', 'n'])
            for fn in sorted(res):
                val = res[fn]
                w.writerow([fn, val if val is not None else ''])
        vals = [x for x in res.values() if x is not None]
        if vals:
            print(f'[{t}/{v}] {args.contact} n={len(vals)} 均值={np.mean(vals):.2f} '
                  f'→ {os.path.basename(out_csv)}')


if __name__ == '__main__':
    main()
