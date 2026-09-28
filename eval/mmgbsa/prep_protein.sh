#!/bin/bash
# 单帧 MM-GBSA 管线 — Step A: 蛋白准备（每靶点一次）
# 用法: bash prep_protein.sh <target>
# 输出: prep/<target>/protein_noh.pdb, rec.prmtop, rec.inpcrd, gap_report.txt
#
# 流程: pdb4amber 预处理 → 二硫键检测补 CONECT → 缺口检查(报告不阻断) → tleap 建受体拓扑
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -u
export AMBERHOME=$HOME/soft/amber20
export PATH=$AMBERHOME/bin:$PATH

TARGET=$1
ROOT=${MLIPDIFF_MMGBSA:-$HOME/mmgbsa}
SRC=$ROOT/proteins/${TARGET}_protein.pdb
OUT=$ROOT/prep/$TARGET
mkdir -p $OUT
cd $OUT || exit 1

echo "== [$TARGET] 蛋白清洗 (clean_protein.py 替代损坏的 pdb4amber) =="
python3 $SCRIPT_DIR/clean_protein.py $SRC protein_noh.pdb > clean.log 2>&1
if [ $? -ne 0 ]; then echo "FAILED: clean_protein"; cat clean.log; exit 1; fi
cat clean.log

echo "== [$TARGET] 二硫键检测 =="
# 检测 CYS SG-SG < 2.5A 的配对，与已有 CONECT 对比，缺失则补 CONECT 行
python3 - <<PY > ss_pairs.txt
import sys, itertools
coords = {}
for line in open('protein_noh.pdb'):
    if line.startswith(('ATOM','HETATM')):
        serial = int(line[6:11]); resn = line[17:20].strip(); atom = line[12:16].strip()
        x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
        coords[serial] = (resn, atom, x, y, z)
sg = {s:v for s,v in coords.items() if v[1]=='SG' and v[0]=='CYS'}
for (s1,v1),(s2,v2) in itertools.combinations(sg.items(), 2):
    d = ((v1[2]-v2[2])**2 + (v1[3]-v2[3])**2 + (v1[4]-v2[4])**2) ** 0.5
    if d < 2.5:
        print(f'{s1} {s2} {d:.2f}')
PY
python3 - <<PY
# 用距离检测结果重建全部 SS 的 CONECT 行（不信任源 PDB 的 CONECT 记录）
pairs = []
for line in open('ss_pairs.txt'):
    parts = line.split()
    if len(parts) >= 2:
        pairs.append((int(parts[0]), int(parts[1])))
with open('protein_noh.pdb', 'a') as f:
    for a, b in pairs:
        f.write(f'CONECT{a:5d}{b:5d}\n')
print(f'重建 SS CONECT: {len(pairs)} 对')
PY

echo "== [$TARGET] 缺口检查（SEQRES vs ATOM）=="
python3 - <<PY > gap_report.txt
from collections import defaultdict
seqres = defaultdict(list)
atoms = defaultdict(set)
for line in open('$SRC'):
    if line.startswith('SEQRES'):
        chain = line[11]
        for k in range(0, len(line[19:70]), 4):
            r = line[19+k:23+k].strip()
            if r: seqres[chain].append(r)
    elif line.startswith(('ATOM','HETATM')):
        atoms[line[21]].add(line[22:27].strip())
for ch in sorted(seqres):
    missing = [(f'{ch}:{i+1}', seqres[ch][i]) for i in range(len(seqres[ch]))
               if str(i+1) not in atoms[ch]]
    if missing:
        print(f'chain {ch}: SEQRES 列了 {len(seqres[ch])} 个残基, 其中 {len(missing)} 个的 1-based 序号未出现在 ATOM 记录中')
        print(f'          前几个: {missing[:8]}{" ..." if len(missing)>8 else ""}')
        print('          注: 本检查按「SEQRES 的第 i 个残基 vs resSeq == i+1」比对, 即假设残基编号从 1 开始.')
        print('              编号有偏移的 PDB (如从 1001 起) 会整体误报, 不代表残基真的缺失.')
    else:
        print(f'chain {ch}: 完整 ({len(seqres[ch])} 残基)')
PY
cat gap_report.txt

echo "== [$TARGET] tleap 建受体拓扑 =="
cat > tleap.in <<EOF
source leaprc.protein.ff19SB
set default PBRadii mbondi2
REC = loadpdb protein_noh.pdb
saveamberparm REC rec.prmtop rec.inpcrd
saveoff REC rec.lib
quit
EOF
tleap -f tleap.in > tleap.log 2>&1
if [ ! -s rec.prmtop ] || [ ! -s rec.inpcrd ] || [ ! -s rec.lib ]; then
    echo "FAILED: tleap (rec.prmtop/rec.lib 未生成)"
    grep -E "Error" tleap.log | head -10
    tail -5 tleap.log
    exit 1
fi
grep -E "CYX|charge|Warning" tleap.log | head -10
ls -lh rec.prmtop rec.inpcrd
echo "DONE: $TARGET 受体拓扑就绪"
