#!/bin/bash
# 单帧 MM-GBSA 管线 — Step C: 批量编排（nice 降优先级，-P 8 并行）
# 用法: bash run_group.sh <target> <group> <sdf_dir>
# 汇总: results/<target>/<group>_mmgbsa.csv
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -u

TARGET=$1; GROUP=$2; SDF_DIR=$3
ROOT=${MLIPDIFF_MMGBSA:-$REPO_ROOT}
RES=$ROOT/results/$TARGET/$GROUP
mkdir -p $RES

[ -f $ROOT/prep/$TARGET/rec.prmtop ] || { echo "先跑 bash prep_protein.sh $TARGET"; exit 1; }

find $SDF_DIR -maxdepth 1 -name '*.sdf' | sort -V > /tmp/mmgbsa_${TARGET}_${GROUP}.txt
N=$(wc -l < /tmp/mmgbsa_${TARGET}_${GROUP}.txt)
echo "[$TARGET/$GROUP] 共 $N 个分子, -P 8 + nice 并行"

cat /tmp/mmgbsa_${TARGET}_${GROUP}.txt | \
    xargs -P 8 -I{} nice -n 10 bash $ROOT/scripts/score_mol.sh $TARGET $GROUP {}

# 汇总（只从 *.txt 行文件聚合，可重复执行）
{
    echo "status,target,group,mol_id,nc,vdwaals,eel,egb,esurf,total"
    cat $RES/*.txt 2>/dev/null | sort -t, -k4,4 -V
} > $RES/${GROUP}_mmgbsa.csv

OK_N=$(grep -c '^OK' $RES/${GROUP}_mmgbsa.csv)
FAIL_N=$(grep -c '^FAILED' $RES/${GROUP}_mmgbsa.csv)
echo "完成: OK=$OK_N FAILED=$FAIL_N → results/$TARGET/${GROUP}_mmgbsa.csv"
