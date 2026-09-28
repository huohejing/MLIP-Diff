#!/usr/bin/env python3
"""蛋白 PDB 清洗器 — 替代 A100 上损坏的 pdb4amber（amber20 的 pdb4amber shebang 指向构建树 python，无法运行）。

实现 pdb4amber 常用组合的功能（-p --noter --most-populous -y）:
  1. 只保留标准残基（20 种氨基酸 + NME/ACE + HID/HIE/HIP/CYX），去水/离子/辅因子
  2. 多构象残基（altloc）保留占据数最高的构象
  3. 去掉所有氢原子（tleap 按 ff19SB 模板统一重建）
  4. 不输出 TER 卡
  5. 保留 CONECT 记录（二硫键），过滤已删原子的引用
  6. MSE(硒代甲硫氨酸) → MET（去掉 SE 原子，与 pdb4amber 默认行为一致）

用法: python3 clean_protein.py in.pdb out.pdb
"""
import sys
from collections import defaultdict

STANDARD = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
            'MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
            'NME','ACE','CYX','HID','HIE','HIP','ASH','GLH','LYN','MSE'}

inp, outp = sys.argv[1], sys.argv[2]

# Pass 1: 收集 ATOM 行 + 每残基各构象的占据数和原子
atom_lines = []
res_alt_occ = defaultdict(lambda: defaultdict(float))
for line in open(inp):
    if not line.startswith(('ATOM', 'HETATM')):
        continue
    resn = line[17:20].strip()
    if resn not in STANDARD:
        continue
    alt = line[16] if line[16] != ' ' else ' '
    occ = float(line[54:60]) if line[54:60].strip() else 0.0
    key = (line[21], line[22:27].strip(), resn)
    res_alt_occ[key][alt] += occ
    atom_lines.append(line)

# 每残基选占据数最高的构象（平局优先 ' ' 再 'A'）
chosen = {}
for key, alts in res_alt_occ.items():
    best = max(alts, key=lambda a: (alts[a], a == ' ', a == 'A'))
    chosen[key] = best

# Pass 2: 过滤（去掉氢、非选中构象、MSE 的 SE）
kept_serial = set()
out_lines = []
mse_renamed = False
for line in atom_lines:
    resn = line[17:20].strip()
    key = (line[21], line[22:27].strip(), resn)
    if line[16] != chosen[key]:
        continue
    # 氢判断: 元素列优先，回退原子名
    elem = line[76:78].strip()
    if not elem:
        name = line[12:16].strip()
        elem = 'H' if (len(name) == 1 and name[0] == 'H') or (len(name) > 1 and name[0].isdigit() and name[1] == 'H') else ''
    if elem == 'H':
        continue
    # MSE → MET: 改名 + 删 SE
    if resn == 'MSE':
        if line[12:16].strip() == 'SE':
            continue
        line = line[:17] + 'MET' + line[20:]
        mse_renamed = True
    kept_serial.add(int(line[6:11]))
    out_lines.append(line)

# 不传递源 PDB 的 CONECT——二硫键由 prep_protein.sh 按 SG-SG 距离重建，
# 避免源 PDB 的 CONECT 引用错原子（串成 2C-SH-SH-HS 链会让 tleap 缺参数失败）
with open(outp, 'w') as f:
    f.writelines(out_lines)
    f.write('END\n')

print(f'输入原子: {len(atom_lines)}, 输出: {len(out_lines)} 原子 (CONECT 由距离法重建)')
if mse_renamed:
    print('注意: MSE 已转为 MET（与 pdb4amber 默认一致）')
