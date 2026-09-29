"""
results8 单靶点 PBR —— 第 1 步（PoseBusters，pb 环境）。

对 total/split 变体跑 PoseBusters(config='dock')，vs 全蛋白，输出逐分子 CSV + PBR 统计。
这一步记录完整 22 项、**不过滤水的检查项**；去水在第 2 步 build_pbr_paired.py 里做
（那一步要同时出 full 和 nowater 两个口径，这里提前滤掉就再也算不出 full 了）。

输入：{results_root}/{target}/native/{target}_protein.pdb   全蛋白
      {results_root}/{target}/{variant}/*.sdf              待评分分子
输出：{results_root}/{target}/eval/{target}_{variant}_pbr.csv
      file, pbr_pass, failed_checks

用法（pb 环境）：
    python results8_pbr.py --target 5ni7 --results-root /path/to/results8
"""
import argparse
import csv
import glob
import os

from posebusters import PoseBusters

DEFAULT_RESULTS8 = os.environ.get(
    'MLIPDIFF_RESULTS8', os.path.join(os.getcwd(), 'results8'))

NON_TEST = ('mol_pred_loaded', 'mol_cond_loaded', 'molecule position', 'file', 'molecule', 'idx')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--variants', nargs='+', default=['total', 'split'])
    ap.add_argument('--results-root', default=DEFAULT_RESULTS8,
                    help='results8 根目录（默认取 $MLIPDIFF_RESULTS8，再退回 ./results8）')
    ap.add_argument('--max-mols', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    t = args.target
    r = os.path.join(args.results_root, t)
    protein = os.path.join(r, 'native', f'{t}_protein.pdb')
    if not os.path.isfile(protein):
        print(f'[SKIP] {t}: 无全蛋白 {protein}')
        return
    os.makedirs(os.path.join(r, 'eval'), exist_ok=True)
    pb = PoseBusters(config='dock', max_workers=args.workers)

    for v in args.variants:
        sdf_dir = os.path.join(r, v)
        files = sorted(glob.glob(os.path.join(sdf_dir, '*.sdf')))
        if args.max_mols:
            files = files[:args.max_mols]
        if not files:
            print(f'[SKIP] {t}/{v}: 无 SDF')
            continue
        # 直接把 SDF 文件路径交给 PoseBusters
        res = pb.bust(files, mol_cond=protein, full_report=False)
        res = res.reset_index()
        test_cols = [c for c in res.columns
                     if res[c].dtype == bool and c not in NON_TEST]
        if not test_cols:
            print(f'[WARN] {t}/{v}: 无测试列')
            continue
        passed = res[test_cols].all(axis=1)
        pbr = float(passed.mean())

        out_csv = os.path.join(r, 'eval', f'{t}_{v}_pbr.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['file', 'pbr_pass', 'failed_checks'])
            for row in res.to_dict('records'):
                failed = [c for c in test_cols if not bool(row[c])]
                fname = os.path.basename(str(row['file']))
                w.writerow([fname, str(len(failed) == 0), ';'.join(failed)])
        n = len(res)
        n_fail = int((~passed).sum())
        print(f'[{t}/{v}] n={n} PBR={pbr:.1%} ({n-n_fail}/{n}) '
              f'→ {os.path.basename(out_csv)}')


if __name__ == '__main__':
    main()
