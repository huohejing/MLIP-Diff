"""评估共享工具：SMILES 对齐 + 追加式 CSV 保存。"""
import csv
import glob
import os

from rdkit import Chem


def canonical_smiles(sdf_path):
    """SDF → canonical SMILES（去 H、去立体）。解析失败返回 None。"""
    m = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if m is None:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=False)


def build_smi_map(sdf_dir):
    """{smiles: sdf_path}，同一 SMILES 取第一个文件。"""
    smi = {}
    for p in sorted(glob.glob(os.path.join(sdf_dir, '*.sdf'))):
        s = canonical_smiles(p)
        if s:
            smi.setdefault(s, p)
    return smi


def align_by_smiles(base_dir, guided_dir):
    """按 SMILES 对齐，同一 SMILES 只留第一个实例。"""
    b = build_smi_map(base_dir)
    g = build_smi_map(guided_dir)
    return [(b[s], g[s], s) for s in sorted(set(b) & set(g))]


def build_smi_list(sdf_dir):
    """{smiles: [sdf_path, ...]}，保留同一 SMILES 的全部实例。"""
    smi = {}
    for p in sorted(glob.glob(os.path.join(sdf_dir, '*.sdf'))):
        s = canonical_smiles(p)
        if s:
            smi.setdefault(s, []).append(p)
    return smi


def align_by_smiles_zip(base_dir, guided_dir):
    """按 SMILES 对齐，组内按 mol_id 数值顺序一一配对。"""
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
    """组内顺序 zip 配对（各指标共用）。

    base_items / guided_items: 形如 [(mol_id, smiles, *payload), ...]，已按 mol_id 排序。
    返回 (pairs, unmatched)：pairs 是 [(base_item, guided_item), ...]；
    unmatched 是 {smiles: n}，base 侧实例不足而被截断掉的 guided 实例数。
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
    """追加写汇总 CSV，按 key_cols 去重更新。"""
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
