# ProLIF

Protein–ligand interaction counts for the generated conformations.

## Requirements

```
pip install prolif
```

`hbond_implicit.py` additionally needs **ProLIF ≥ 2.2** — 2.0.x has no
`ImplicitHBDonor` / `ImplicitHBAcceptor`.

## Usage

```
python eval/prolif/hbond.py           --target 3ctj --variants base total split
python eval/prolif/hbond_geometry.py  --target 3ctj --variants base total split
python eval/prolif/hbond_implicit.py  --target 3ctj --variants base total split
python eval/prolif/hydrophobic.py     --target 3ctj --variants base total split
python eval/prolif/contacts.py        --target 3ctj --contact pistack
```

`contacts.py --contact` is one of `pistack`, `saltbridge`, `halogen`,
`pication`, `vdw`. `--workers` defaults to 4.

`--results-root` defaults to `$MLIPDIFF_RESULTS8`, then to `./results8`.

## Input

```
{results_root}/{target}/
├── native/{target}_pocket_h.pdb    hydrogenated pocket (PDBFixer)
├── native/{target}_pocket.pdb      heavy-atom pocket
├── base/*.sdf                      unguided molecules
└── {total,split}/*.sdf             guided molecules
```

| Script | Pocket |
|:---|:---|
| `hbond.py`, `hbond_geometry.py`, `contacts.py` | `{target}_pocket_h.pdb` |
| `hbond_implicit.py`, `hydrophobic.py` | `{target}_pocket.pdb` |

The pocket is hydrogenated with PDBFixer. Ligands are hydrogenated inside the
scripts with RDKit, `Chem.AddHs(mol, addCoords=True)`.

## Output

`{results_root}/{target}/eval/{target}_{variant}_{name}.csv`.

All but `hbond_geometry.py` write `file,n` — one count per molecule.
`hbond_geometry.py` writes `file,dist,DHA_angle`, one row per hydrogen bond, in
Å and degrees.
