"""
Native 配体 Vina score_only 基线分：晶体 pose 原样打分，不做重对接。

协议与生成分子的打分完全一致（同一 VinaDockingTask、同一 pocket PDB、
20x20x20 盒），保证分数可比较。

输入：{root}/validation/{target}/{target}_ligand.sdf   晶体配体
      {root}/sample/{target}_pocket.pdb                口袋
输出：{root}/validation/native_vina_baseline.csv       增量更新，不覆盖其他靶点

用法:
    python native_baseline.py --targets 3ctj 1w51
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem
from docking_vina import VinaDockingTask

ROOT = os.environ.get('MLIPDIFF_ROOT', os.getcwd())
TARGETS = ['1a0q', '1w51', '3ctj', '5ni7', '5ywy', '6mey']


def score_native(target):
    """native 晶体 pose score_only，返回 (kcal/mol, 原子数)。"""
    ligand_sdf = os.path.join(ROOT, 'validation', target, f'{target}_ligand.sdf')
    pocket_pdb = os.path.join(ROOT, 'sample', f'{target}_pocket.pdb')
    mol = Chem.SDMolSupplier(ligand_sdf, removeHs=False)[0]
    if mol is None:
        return None, None
    n_atoms = mol.GetNumAtoms()
    task = VinaDockingTask.from_generated_mol(mol, pocket_pdb, size_factor=None)
    result = task.run(mode='score_only')
    return result[0]['affinity'], n_atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=TARGETS)
    args = ap.parse_args()

    rows = []
    for t in args.targets:
        try:
            score, n_atoms = score_native(t)
            print(f'{t}: native score_only = {score:.2f} kcal/mol ({n_atoms} 原子)')
            rows.append([t, round(score, 2), n_atoms])
        except Exception as e:
            print(f'{t}: 失败 {str(e)[:100]}')
            rows.append([t, '', ''])

    out = os.path.join(ROOT, 'validation', 'native_vina_baseline.csv')
    # 增量更新：读已有 CSV，只更新/新增本次 targets，避免覆盖其他靶点
    existing = {}
    if os.path.exists(out):
        with open(out) as f:
            for r in csv.DictReader(f):
                existing[r['target']] = (r['native_vina_kcal'], r['n_atoms'])
    for t, sc, na in rows:
        existing[t] = (str(sc), str(na))
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['target', 'native_vina_kcal', 'n_atoms'])
        for t in TARGETS:
            if t in existing:
                w.writerow([t, existing[t][0], existing[t][1]])
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
