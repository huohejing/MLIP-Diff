#!/usr/bin/env python
"""PBR 配对版：同一个分子（SMILES 配对）引导前 vs 引导后的通过情况翻转。

配对方式（2026-09-16 改，与 MACE / 氢键 / MM-GBSA / strain 统一）：
  去立体 canonical SMILES 分组 → **同一 SMILES 组内按 mol_id 数值顺序一一配对**
  （base[i] ↔ guided[i]，取 min 即停），**不去重、不展开、不取最小 mol_id**。
  实现走 `eval_utils.zip_by_smiles`，五个指标共用同一份配对代码。
  → 与 huizonghuizong 里 MACE full_eval 口径一致（数据质量问题清单 §1.4）。
  旧口径为「去立体 + base 取最小 mol_id」，结果已备份到 backup_pairing_before_zip/。
  ⚠️ base 实例数少于 guided 时，多出的 guided 实例配不上对，计入 n_unmatched_instances。

四个格子（配对布尔）：
  gain  = base 不过 → guided 过   （引导修好了）
  loss  = base 过   → guided 不过 （引导弄坏了）
  tie   = 都过 / 都不过

统计：
  better_rate           = gain / n_paired          （与 vina/mmgbsa 表同分母口径）
  better_rate_discordant= gain / (gain + loss)     （只在发生过翻转的分子里看赢面）
  mcnemar_p             = McNemar 精确检验（双尾），配对布尔的正确检验

★ 第 2 步：去水在这里做（不在 results8_pbr.py 里）。
  口径 nowater：通过 = 22 项全过，**或** 失败项恰好是 {minimum_distance_to_waters}。
  只排除 minimum_distance_to_waters 这一项，与已有结果表一致。

输出：{out_dir}/pbr_paired_nowater.csv      主口径
      {out_dir}/pbr_paired_full.csv         补充材料口径
      {results_root}/{t}/eval/{t}_{total,split}_pbr_paired.csv   逐分子配对明细

用法（需 rdkit）：
    python build_pbr_paired.py --results-root /path/to/results8 --out-dir /path/to/out
"""
import argparse
import csv
import os
import sys
from math import comb

from rdkit import Chem

# eval_utils.py 在上一级目录（eval/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils import zip_by_smiles

DEFAULT_RESULTS8 = os.environ.get(
    'MLIPDIFF_RESULTS8', os.path.join(os.getcwd(), 'results8'))
TARGETS = ['1w51', '3ctj', '2wi6', '3u5l', '4hbm', '5ni7', '5ywy', '6mey']
VARIANTS = ['total', 'split']
WATER_CHECK = 'minimum_distance_to_waters'
MODES = ['nowater', 'full']


def canon_smi(path):
    """单分子 SDF → 去立体 canonical SMILES。"""
    try:
        m = Chem.MolFromMolFile(path, removeHs=False)
        if m is None:
            return None
        mn = Chem.RemoveHs(m)
        Chem.SanitizeMol(mn)
        return Chem.MolToSmiles(mn, isomericSmiles=False)
    except Exception:
        return None


def load_eval(res8, target, variant):
    """{mol_id: (pass_full, pass_nowater)}，来自 PBR eval CSV。"""
    p = f'{res8}/{target}/eval/{target}_{variant}_pbr.csv'
    d = {}
    if not os.path.exists(p):
        return d
    with open(p, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            stem = r['file']
            mid = os.path.splitext(os.path.basename(stem))[0]
            if not mid.isdigit():
                continue
            ok = r['pbr_pass'].strip() == 'True'
            failed = set(filter(None, r['failed_checks'].split(';')))
            d[int(mid)] = (ok, ok or failed == {WATER_CHECK})
    return d


def load_group(res8, target, variant):
    """{mol_id: (smiles, pass_full, pass_nowater)}，SDF 与 eval 取交集。"""
    gdir = f'{res8}/{target}/{variant}'
    ev = load_eval(res8, target, variant)
    out = {}
    if not os.path.isdir(gdir):
        return out
    for fn in os.listdir(gdir):
        if not fn.endswith('.sdf') or not fn[:-4].isdigit():
            continue
        mid = int(fn[:-4])
        if mid not in ev:
            continue
        s = canon_smi(os.path.join(gdir, fn))
        if s:
            out[mid] = (s, ev[mid][0], ev[mid][1])
    return out


def mcnemar_exact(gain, loss):
    """McNemar 精确检验（双尾），基于不一致对的二项分布。"""
    n = gain + loss
    if n == 0:
        return ''
    k = min(gain, loss)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return f'{min(p, 1.0):.3g}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default=DEFAULT_RESULTS8,
                    help='results8 根目录（默认取 $MLIPDIFF_RESULTS8，再退回 ./results8）')
    ap.add_argument('--out-dir', default=os.getcwd(),
                    help='汇总表输出目录（默认当前目录）')
    ap.add_argument('--targets', nargs='+', default=TARGETS)
    ap.add_argument('--variants', nargs='+', default=VARIANTS)
    args = ap.parse_args()

    res8, outdir = args.results_root, args.out_dir
    os.makedirs(outdir, exist_ok=True)
    tables = {m: [] for m in MODES}

    for t in args.targets:
        base = load_group(res8, t, 'base')
        if not base:
            print(f'{t}: 缺 base，跳过')
            continue
        base_smiles = {base[mid][0] for mid in base}
        base_items = [(mid, base[mid][0], base[mid][1], base[mid][2])
                      for mid in sorted(base)]

        for v in args.variants:
            grp = load_group(res8, t, v)
            if not grp:
                continue
            # guided 侧保持 ({mol_id 数值序}) —— zip 的顺序即由它决定
            guided_items = [(mid, grp[mid][0], grp[mid][1], grp[mid][2])
                            for mid in sorted(grp)]
            pairs, unmatched = zip_by_smiles(base_items, guided_items)
            n_not_in_base = sum(1 for it in guided_items if it[1] not in base_smiles)
            n_unmatched = sum(unmatched.values())

            detail = []
            gains = {m: 0 for m in MODES}
            losses = {m: 0 for m in MODES}
            ties = {m: 0 for m in MODES}
            bp = {m: 0 for m in MODES}
            gp = {m: 0 for m in MODES}

            for (bmid, bsmi, bpf, bpn), (mid, gs, gpf, gpn) in pairs:
                row = [gs, mid, bmid]
                for m, gv, bv in (('full', gpf, bpf), ('nowater', gpn, bpn)):
                    bp[m] += bv
                    gp[m] += gv
                    if gv and not bv:
                        gains[m] += 1
                    elif bv and not gv:
                        losses[m] += 1
                    else:
                        ties[m] += 1
                    row += [int(bv), int(gv)]
                detail.append(row)

            n = len(detail)
            if not n:
                continue
            with open(f'{res8}/{t}/eval/{t}_{v}_pbr_paired.csv',
                      'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['smiles', 'guided_mol_id', 'base_mol_id',
                            'base_pass_full', 'guided_pass_full',
                            'base_pass_nowater', 'guided_pass_nowater'])
                w.writerows(detail)

            for m in MODES:
                g_, l_ = gains[m], losses[m]
                tables[m].append([t, v, n, bp[m] / n, gp[m] / n,
                                  (gp[m] - bp[m]) / n * 100, g_, l_, ties[m],
                                  g_ / n, g_ / (g_ + l_) if g_ + l_ else '',
                                  mcnemar_exact(g_, l_),
                                  n_not_in_base, n_unmatched])
                print(f'{t:<7}{v:<8}{m:<9}n={n:<5}'
                      f'gain={g_:<4}loss={l_:<4}tie={ties[m]:<5}'
                      f'better={g_ / n * 100:>6.1f}%  p={mcnemar_exact(g_, l_)}')
            print()

    for m in MODES:
        out = f'{outdir}/pbr_paired_{m}.csv'
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['target', 'variant', 'n_paired', 'base_pass_rate',
                        'guided_pass_rate', 'delta_pp', 'n_gain', 'n_loss',
                        'n_tie', 'better_rate', 'better_rate_discordant',
                        'mcnemar_p', 'n_smiles_not_in_base',
                        'n_unmatched_instances'])
            w.writerows(tables[m])
        print(f'→ {out}  ({len(tables[m])} 行)')


if __name__ == '__main__':
    main()
