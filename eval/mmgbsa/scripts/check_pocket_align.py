#!/usr/bin/env python3
"""验证口袋 PDB 与全蛋白 PDB 坐标一致性（pose 可直接放入全蛋白坐标系的前提）。

用法: python3 check_pocket_align.py <pocket.pdb> <protein.pdb> [阈值A=0.01]
退出码: 0=通过, 1=不通过（坐标不一致，禁止直接打分）
"""
import sys


def parse(f):
    atoms = {}
    for l in open(f):
        if l.startswith(('ATOM', 'HETATM')):
            key = (l[21], l[22:27].strip(), l[12:16].strip(), l[16])
            atoms[key] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
    return atoms


def main():
    pkt_pdb, prot_pdb = sys.argv[1], sys.argv[2]
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    p = parse(pkt_pdb)
    f = parse(prot_pdb)
    n, maxd, worst = 0, 0.0, None
    for k, v in p.items():
        if k in f:
            d = sum((a - b) ** 2 for a, b in zip(v, f[k])) ** 0.5
            n += 1
            if d > maxd:
                maxd, worst = d, k
    frac = n / len(p) if p else 0
    print(f'pocket atoms: {len(p)}, matched: {n} ({frac:.1%}), max coord diff: {maxd:.4f} A at {worst}')
    ok = (frac >= 0.99) and (maxd <= thresh)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
