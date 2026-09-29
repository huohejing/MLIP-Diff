"""评估共享工具：SMILES 对齐 + 追加式 CSV 保存。

被 score_generated_vina / posebusters_batch / plif_analysis 共用。
"""
import csv
import glob
import os

from rdkit import Chem


def canonical_smiles(sdf_path):
    """读 SDF → canonical SMILES（去 H，不含立体）。解析失败返回 None。"""
    m = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if m is None:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=False)


def build_smi_map(sdf_dir):
    """{canonical_smiles: sdf_path}，同一 SMILES 多个文件取第一个。"""
    smi = {}
    for p in sorted(glob.glob(os.path.join(sdf_dir, '*.sdf'))):
        s = canonical_smiles(p)
        if s:
            smi.setdefault(s, p)
    return smi


def align_by_smiles(base_dir, guided_dir):
    """按 canonical SMILES 对齐 base/guided 两目录，返回 [(base_sf, guided_sf, smiles)]。

    ⚠️ 同一 SMILES 有多个实例时**只留排序第一个文件**（去重）。要保留全部实例请用
       align_by_smiles_zip（组内按顺序配对，与 MACE full_eval 口径一致）。
    """
    b = build_smi_map(base_dir)
    g = build_smi_map(guided_dir)
    return [(b[s], g[s], s) for s in sorted(set(b) & set(g))]


def build_smi_list(sdf_dir):
    """{canonical_smiles: [sdf_path, ...]}，保留同一 SMILES 的**全部实例**，按文件排序。"""
    smi = {}
    for p in sorted(glob.glob(os.path.join(sdf_dir, '*.sdf'))):
        s = canonical_smiles(p)
        if s:
            smi.setdefault(s, []).append(p)
    return smi


def align_by_smiles_zip(base_dir, guided_dir):
    """按 canonical SMILES 对齐，同一 SMILES 组内**按 mol_id 数值顺序一一配对**。

    对每个 SMILES，取 base 实例列表与 guided 实例列表，按 mol_id **从小到大** zip：
        base[0]↔guided[0], base[1]↔guided[1], ... 取 min(len) 即停。
    **不去重、不展开、不取最小 mol_id** —— 与 huizonghuizong 里 MACE full_eval 口径一致
    （数据质量问题清单 §1.4）。返回 [(base_sf, guided_sf, smiles)]。

    ⚠️ 排序键是 mol_id 的**数值**（2.sdf 在 10.sdf 之前），不是文件名字符串。
       所有指标必须用同一顺序，否则同一 SMILES 组内配对结果会不同。
    """
    def _items(d):
        out = []
        for s, paths in d.items():
            for p in paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                out.append((int(stem) if stem.isdigit() else float('inf'), s, p))
        return sorted(out)

    pairs, _ = zip_by_smiles(_items(build_smi_list(base_dir)),
                             _items(build_smi_list(guided_dir)))
    return [(bi[2], gi[2], bi[1]) for bi, gi in pairs]


def zip_by_smiles(base_items, guided_items):
    """通用"按 SMILES 组内顺序 zip"配对，供 PBR / MM-GBSA / strain / 氢键 共用。

    base_items / guided_items: 形如 [(mol_id, smiles, *payload), ...] 的**已按 mol_id 排序**列表。
    对每个 SMILES，取 base 实例列表与 guided 实例列表按顺序一一配对：
        base[0]↔guided[0], base[1]↔guided[1], ... 取 min(len) 即停。
    **不去重、不展开、不取最小 mol_id** —— 与 huizonghuizong 里 MACE full_eval 口径一致
    （数据质量问题清单 §1.4）。

    返回 (pairs, unmatched)：
      pairs     = [(base_item, guided_item), ...]，元素是原 tuple，payload 原样带出
      unmatched = {smiles: n}，base 侧实例数不足、被截断掉的 guided 实例数
                  （这些 guided 分子没配上对，不是"base 里没这个分子"）
    """
    b, g = {}, {}
    for it in base_items:
        b.setdefault(it[1], []).append(it)
    for it in guided_items:
        g.setdefault(it[1], []).append(it)

    pairs, unmatched = [], {}
    for s in sorted(set(b) & set(g)):
        pairs.extend(zip(b[s], g[s]))
        extra = len(g[s]) - len(b[s])
        if extra > 0:
            unmatched[s] = extra
    return pairs, unmatched


def append_summary(csv_path, fieldnames, rows, key_cols):
    """追加写汇总 CSV：读已有行，按 key_cols 去重更新，再写回（历史不丢）。"""
    data = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                key = tuple(r.get(k) for k in key_cols)
                data[key] = {k: r.get(k, '') for k in fieldnames}
    for r in rows:
        key = tuple(r[k] for k in key_cols)
        data[key] = {k: r[k] for k in fieldnames}
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(data.values())
