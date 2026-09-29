# PBR (PoseBusters)

Conformational plausibility of the generated molecules. **PBR** is the fraction
of molecules that pass every PoseBusters check, run with `config='dock'`.

## Requirements

```
pip install posebusters
```

## Usage

```
python eval/pbr/results8_pbr.py --target 3ctj --variants base total split \
    --results-root /path/to/results8
```

## Layout

```
{results_root}/
├── {target}/native/{target}_protein.pdb    INPUT — whole receptor
├── {target}/base/*.sdf                     INPUT — unguided molecules
└── {target}/{total,split}/*.sdf            INPUT — guided molecules
```

`--max-mols N` scores only the first N molecules per variant, for a quick check.

## Output

`{results_root}/{target}/eval/{target}_{variant}_pbr.csv`, one row per molecule:

| Column | |
|:---|:---|
| `file` | molecule file name |
| `pbr_pass` | all 22 checks passed |
| `failed_checks` | names of the failed checks, `;`-joined |

## Water

PBR is reported **without** `minimum_distance_to_waters`. A molecule passes when
all 22 checks pass, or when that one check is its only failure.

DiffGui is trained without crystallographic water as a conditioning input, so
the model never sees a solvent shell and cannot learn to avoid one. Failing a
generated molecule for colliding with water it cannot see measures a blind spot
of the model, not the quality of the conformation it produced.

The second water check, `volume_overlap_with_waters`, still counts.
