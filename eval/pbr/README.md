# PBR (PoseBusters)

Conformational plausibility of the generated molecules, scored with
[PoseBusters](https://github.com/maabuu/posebusters) `config='dock'`, which
excludes the RMSD-based tests — de novo molecules have no reference
conformation.

**PBR** is the fraction of molecules passing every check.

## Environment

Run in the **`pb`** environment (PoseBusters 0.6.5). It needs a recent RDKit —
the shared `diffgui_cpu` environment has RDKit 2022.09, which lacks the
`GetProp(autoConvert)` argument PoseBusters 0.6.5 calls, and crashes. That is
the only reason this step has its own environment.

```
conda activate pb
```

## Usage

```
python eval/pbr/results8_pbr.py --target 3ctj --variants base total split \
    --results-root /path/to/results8
```

Expected layout:

```
{results_root}/
├── {target}/native/{target}_protein.pdb    whole receptor — the clash checks need all of it
├── {target}/base/*.sdf                     unguided molecules
└── {target}/{total,split}/*.sdf            guided molecules
```

`--results-root` defaults to `$MLIPDIFF_RESULTS8`, then to `./results8`.
`--max-mols N` scores only the first N molecules per variant, for a quick check
of an installation.

## Output

`{results_root}/{target}/eval/{target}_{variant}_pbr.csv`, one row per molecule:

| Column | |
|:---|:---|
| `file` | molecule file name |
| `pbr_pass` | all 22 checks passed |
| `failed_checks` | `;`-joined names of every failed check |

All 22 checks are recorded here, **water included** — the water checks are
excluded downstream, not at scoring time, so the record stays complete.

## The water convention

PBR is reported **without** the water-related checks. Of the two PoseBusters
water checks —

| Check | |
|:---|:---|
| `minimum_distance_to_waters` | closest contact between ligand and any water |
| `volume_overlap_with_waters` | ligand volume overlapping water volume |

— only `minimum_distance_to_waters` is excluded. A molecule counts as passing
when all 22 checks pass, **or** when its only failure is that one check:

```python
WATER_CHECK = 'minimum_distance_to_waters'
```

DiffGui is trained without crystallographic water as a conditioning input, so
the model never sees a solvent shell and cannot learn to avoid one. Scoring with
water included would fail generated molecules for colliding with water they
cannot see — that measures a blind spot of the model, not the quality of the
conformation it produced.

`volume_overlap_with_waters` is kept in the tally, so this convention is not the
same as "all water checks dropped".
