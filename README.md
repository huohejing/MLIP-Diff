# mlip-diff

**MLIP-guided diffusion for protein-conditioned 3D ligand generation.**

Inference-time, training-free physical guidance for a pretrained diffusion-based
molecular generator. Machine-learning interatomic potential (MLIP) energies and
forces are injected into the reverse-diffusion trajectory as dynamic feedback,
so that generated ligand conformations are refined by atomistic physical
information without retraining the generative model.

> **Status.** Manuscript in preparation.
>
> Archived version:
> [10.5281/zenodo.22975994](https://doi.org/10.5281/zenodo.22975994)
> (the concept DOI — it always resolves to the latest version).

---

## Method

The generative backbone is [DiffGui](https://github.com/QiaoyuHu89/DiffGui), an
E(3)-equivariant diffusion model that jointly generates ligand atom types, bond
types and 3D coordinates under a protein-pocket condition. The guidance module
is added purely at inference time; **no DiffGui weights are modified and no
additional training is performed.**

During late-stage denoising, the predicted ligand state is reconstructed into a
chemistry-aware proxy and evaluated with **MACE-OFF24(M)** under a fixed local
protein environment. Two evaluations are performed:

| Evaluation | System | Force |
|:---|:---|:---|
| Complex | ligand proxy + fixed local protein environment | `F_complex` |
| Isolated | ligand proxy alone | `F_ligand` |

The **interaction-related** force component is obtained by decomposition:

```
F_interaction = F_complex - F_ligand
```

This separates ligand-intrinsic conformational contributions from
protein-environment-induced interaction contributions, and allows their
relative influence to be balanced during guidance. The correction is scaled by
the intrinsic coordinate-update magnitude of the current diffusion step rather
than by the raw force magnitude, so the diffusion trajectory determines the
spatial scale of each update.

Atom and bond identities continue to evolve according to the original DiffGui
discrete transition process throughout; only the continuous coordinate
trajectory is modified.

## Repository layout

```
models/           DiffGui backbone + physical guidance module
  physical_guidance.py    guidance implementation (force evaluation, decomposition,
                          proxy reconstruction, coordinate correction)
scripts/          sampling, target preparation, evaluation and figure scripts
configs/          sampling and training configuration files
data/             (reserved — released via Zenodo, not yet in this repo)
utils/            DiffGui utilities (dataset, transforms, reconstruction, metrics)
```

## Installation

```bash
conda env create -f env.yml
conda activate diffgui
```

See the [DiffGui repository](https://github.com/QiaoyuHu89/DiffGui) for the
full dependency list, including the `torch_cluster` / `torch_scatter` wheels
matching your CUDA and PyTorch versions.

## Path configuration

No script contains an absolute path. All base directories are resolved
relative to the repository root by `utils/paths.py`, so a fresh clone runs
from wherever you put it.

If your layout differs (for example, results live on a scratch disk), override
with environment variables instead of editing any file:

| Variable | Default | Meaning |
|:---|:---|:---|
| `MLIPDIFF_ROOT` | auto-detected repo root | repository root |
| `MLIPDIFF_DATA` | `<root>/data` | results shipped with the repo |
| `MLIPDIFF_VALIDATION` | `<root>/validation` | raw sampling output |
| `MLIPDIFF_WORK` | `<root>/work` | scratch / intermediates |
| `MLIPDIFF_MACE_MODEL` | `~/.cache/mace/MACE-OFF24_medium.model` | MLIP checkpoint |
| `MLIPDIFF_MMGBSA` | `<work>/mmgbsa` | MM-GBSA working directory |
| `MLIPDIFF_PROTEIN_PARAMS` | *(none)* | precomputed AMBER protein params (legacy `mode: amber` only) |

Note: `mace_model_path` in a sampling config is used **as written** — it is not
tilde-expanded. Give it an absolute path, or export `MLIPDIFF_MACE_MODEL`.

## Model and potential files

These are **not** redistributed in this repository. Obtain them from their
original sources:

| File | Source |
|:---|:---|
| `trained.pt` (DiffGui checkpoint) | [Google Drive](https://drive.google.com/drive/folders/1pQk1FASCnCLjYRd7yc17WfctoHR50s2r) |
| `bond_trained.pt` (bond predictor) | same folder |
| `MACE-OFF24_medium.model` | [MACE-OFF](https://github.com/ACEsuit/mace-off) |

Place the DiffGui checkpoints in `ckpt/`.

## Usage

### 1. Prepare a target

```bash
python scripts/prepare_target_mace.py \
    --protein 3ctj_protein.pdb \
    --ligand  3ctj_ligand.sdf \
    --name    3ctj \
    --outdir  sample
```

This writes three files, because a guided run uses the pocket in two different
ways and the two uses need different structures:

| File | Role |
|:---|:---|
| `3ctj_pocket.pdb` | pocket residues within the radius, heavy atoms only — the DiffGui conditioning input (`model.target`) |
| `3ctj_pocket_h.pdb` | the same residues completed and hydrogenated with PDBFixer — the MACE protein environment (`physical_guidance.pocket_pdb`) |
| `full_pocket.npz` | flat coordinate/element arrays of the pocket, read by the evaluation scripts |

The hydrogenated pocket is not optional: polar hydrogens fix the hydrogen-bond
geometry the guidance force is computed from. Hydrogenation needs
`pdbfixer` — if it is missing the script stops rather than emitting a
non-hydrogenated file under the `_h` name.

### 2. Guided sampling

Edit `configs/sample/sample.yml`, then:

```bash
python scripts/sample.py --outdir ./outputs --config ./configs/sample/sample.yml --device cuda:0
```

The `physical_guidance` block of the config controls the guidance schedule:

```yaml
sample:
  physical_guidance:
    enabled: true
    start_step: 25       # last 25 denoising steps
    interval: 1          # correct on every one of them
    readiness: true      # chemistry-aware proxy reconstruction
    ...
```

Set `enabled: false` for the unguided baseline.

### 3. Evaluation

Evaluation scripts are in `scripts/`; see each script's docstring for its
required inputs and written outputs.

## Data

The generated ligand conformations and the energetic and structural evaluation
results for the eight benchmark systems are released separately, and are not
part of this repository yet. They will be added as a new version of the archived
record ([10.5281/zenodo.22975994](https://doi.org/10.5281/zenodo.22975994)).

Unguided and guided structures are paired by canonical SMILES, so every
comparison is made between identical molecular identities.

## Citation

```bibtex
@article{...,
  title   = {...},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

Archived release: [10.5281/zenodo.22975994](https://doi.org/10.5281/zenodo.22975994)

This is the concept DOI — it always resolves to the latest version, so it stays
valid when the data and analysis scripts are added.

## License and attribution

This work builds on DiffGui, which is licensed under the MIT License.
The original copyright notice is retained in `LICENSE`.

- **DiffGui** — https://github.com/QiaoyuHu89/DiffGui
  Copyright (c) 2024 Qiaoyu Hu
- See `NOTICE` for a description of the modifications made in this repository.
