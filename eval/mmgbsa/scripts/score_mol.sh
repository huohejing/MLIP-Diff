#!/bin/bash
# 单帧 MM-GBSA 管线 — Step B: 单分子打分（无 minimization、无 MD）
# 用法: bash score_mol.sh <target> <group> <sdf_path>
# 输出: results/<target>/<group>/<mol_id>.txt 一行结果 (status,vdw,eel,egb,esurf,total)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -u
export AMBERHOME=$HOME/soft/amber20
export PATH=$AMBERHOME/bin:$PATH

TARGET=$1; GROUP=$2; SDF=$(readlink -f $3)
MOLID=$(basename $SDF .sdf)
ROOT=${MLIPDIFF_MMGBSA:-$REPO_ROOT}
W=$ROOT/work/$TARGET/$GROUP/$MOLID
RES=$ROOT/results/$TARGET/$GROUP
mkdir -p $W $RES
cd $W || exit 1

write_result() {  # $1=status, $2..=可选 stage（放在 total 列）
    local st=$1; shift
    echo "$st,$TARGET,$GROUP,$MOLID,,NA,NA,NA,NA,$*" > $RES/$MOLID.txt
    exit 0
}

# 0. RDKit 预处理: 芳香键 Kekulé 化 + 加氢 + 提取形式电荷
# 原因1: antechamber 的 SDF 解析不识别芳香键(类型4)，芳香碳被读成 3 价
#        → acdoctor 报 "Weird atomic valence"，必须转 Kekulé 式再补氢
# 原因2: 每个配体用 RDKit 形式电荷传给 antechamber -nc（不再强制中性）
RDKIT_PY=${MLIPDIFF_PYTHON:-python}
$RDKIT_PY - <<PY || write_result "FAILED" "rdkit"
from rdkit import Chem
mol = Chem.SDMolSupplier('$SDF', removeHs=False)[0]
if mol is None:
    raise SystemExit(1)
Chem.SanitizeMol(mol)
nc = Chem.GetFormalCharge(mol)
molh = Chem.AddHs(mol, addCoords=True)
Chem.Kekulize(molh)
w = Chem.SDWriter('lig_h.sdf')
w.write(molh)
w.close()
with open('nc.txt', 'w') as f:
    f.write(str(nc))
PY
NC=$(cat nc.txt)

# 1. 配体参数化 (AM1-BCC, GAFF2, 按形式电荷)
antechamber -i lig_h.sdf -fi sdf -o lig.mol2 -fo mol2 -c bcc -at gaff2 -rn LIG -nc $NC \
    > antechamber.log 2>&1
[ $? -ne 0 ] && write_result "FAILED" "antechamber"

parmchk2 -i lig.mol2 -f mol2 -o lig.frcmod > parmchk2.log 2>&1

# 2. tleap: 受体 loadpdb(带坐标) + 配体 combine
# 注: OFF 库(saveoff/loadoff)不保存坐标，受体坐标会丢——所以必须 loadpdb
cat > tleap.in <<EOF
source leaprc.protein.ff19SB
source leaprc.gaff2
set default PBRadii mbondi2
REC = loadpdb $ROOT/prep/$TARGET/protein_noh.pdb
loadamberparams lig.frcmod
LIG = loadmol2 lig.mol2
COM = combine {REC LIG}
saveamberparm COM com.prmtop com.inpcrd
saveamberparm LIG lig.prmtop lig.inpcrd
quit
EOF
tleap -f tleap.in > tleap.log 2>&1
[ $? -ne 0 ] && write_result "FAILED" "tleap"

# 3. 单帧 MM-GBSA (igb=2 OBC, 生理离子强度, 帧 1 就 1 帧)
cat > mmpbsa.in <<EOF
&general
  startframe=1, endframe=1, interval=1,
  verbose=2, keep_files=0,
/
&gb
  igb=2, saltcon=0.100,
/
EOF
MPI_OFF=1 MMPBSA.py -O -i mmpbsa.in -o FINAL.dat \
    -sp com.prmtop -cp com.prmtop \
    -rp $ROOT/prep/$TARGET/rec.prmtop -lp lig.prmtop \
    -y com.inpcrd > mmpbsa.log 2>&1
[ $? -ne 0 ] && write_result "FAILED" "mmpbsa"

# 4. 解析 DELTA 分项 → 结果行（FINAL.dat 每行带 "|" 前缀）
vals_line=$(python3 - <<PY
import sys
nc = open('nc.txt').read().strip()
vals = {}; in_diff = False
for line in open('FINAL.dat'):
    s = line[1:].strip() if line.startswith('|') else line.strip()
    if s.startswith('Differences'):
        in_diff = True; continue
    if in_diff:
        p = s.split()
        if len(p) >= 2 and p[0] in ('VDWAALS','EEL','EGB','ESURF'):
            vals[p[0]] = p[1]
        elif len(p) >= 3 and p[0] == 'DELTA' and p[1] == 'TOTAL':
            vals['TOTAL'] = p[2]
if 'TOTAL' not in vals:
    sys.exit(1)
# 完整 10 字段: OK,target,group,mol_id,nc,vdw,eel,egb,esurf,total
print(f"OK,$TARGET,$GROUP,$MOLID,{nc},{vals.get('VDWAALS','NA')},{vals.get('EEL','NA')},{vals.get('EGB','NA')},{vals.get('ESURF','NA')},{vals['TOTAL']}")
PY
) || write_result "FAILED" "parse"
echo "$vals_line" > $RES/$MOLID.txt

# 5. 清理大文件（保留 dat 和日志供复查）
rm -f com.prmtop com.inpcrd lig.prmtop lig.inpcrd rec.lib leap.log mdinfo
echo "OK: $TARGET/$GROUP/$MOLID nc=$NC total=$(echo $vals_line | cut -d, -f10)"
