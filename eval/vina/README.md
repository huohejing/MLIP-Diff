# Vina

AutoDock Vina `score_only` scoring of the generated conformations, plus the
crystal-ligand baseline that BNC is measured against.

## Requirements

- AutoDock Vina: `pip install vina`
- `meeko`, `openbabel`

## Usage

```
# generated molecules — guided vs base
python eval/vina/score.py --target 3ctj \
    --guided-dir /path/to/guided_sdf \
    --base-dir   /path/to/base_sdf \
    --pocket     /path/to/3ctj_pocket.pdb \
    --out-dir    /path/to/out

# crystal ligand baseline
python eval/vina/native_baseline.py --targets 3ctj 1w51
```

`--discover` scans `<root>/data24` for every `{target}_*_guided` directory and
scores all of them. `--base-dir` and `--pocket` otherwise fall back to
per-target defaults resolved against `$MLIPDIFF_ROOT` (default: current
directory).

## Protocol

`score_only` against the given pocket — no re-docking, no minimisation, the
generated pose is left as it is. Base and guided molecules are therefore scored
on their own conformations, which is what makes the two comparable.

Scores with `|affinity| >= 100` kcal/mol are dropped as non-physical.

## Output

`{out_dir}/vina_{target}_{variant}.csv`, one row per pair, full precision:

| Column | |
|:---|:---|
| `smiles` | canonical SMILES, no stereochemistry |
| `base_vina`, `guided_vina` | affinities, kcal/mol |
| `diff_kcal` | `guided - base`; negative means guidance improved the score |
| `guided_better` | `guided < base` |

plus `vina_{target}_{variant}.json` with `n_paired`, `pct_better`, `mean_delta`,
`median_delta`.

`native_baseline.py` writes `{root}/validation/native_vina_baseline.csv`. It
updates only the targets named on the command line, so scoring one target does
not wipe the others.

## Files

| | |
|:---|:---|
| `score.py` | scoring and base pairing |
| `native_baseline.py` | crystal-ligand baseline |
| `docking_vina.py` | `VinaDockingTask`, from DiffGui |

`score.py` is not called `vina.py` on purpose: Python puts the script's own
directory first on `sys.path`, so a local `vina.py` would shadow the `vina`
package that `docking_vina.py` imports.
