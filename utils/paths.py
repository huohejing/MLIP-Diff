"""Path resolution for this repository.

Scripts must not hardcode absolute paths — a checkout has to run from
wherever the user cloned it. Import the base directories from here instead:

    from utils.paths import ROOT, DATA, VALIDATION, WORK

Every value can be overridden with an environment variable, so a user with a
different layout (results on a scratch disk, validation data elsewhere) does
not have to edit any script.

    MLIPDIFF_ROOT        repository root            (default: auto-detected)
    MLIPDIFF_DATA        shipped results            (default: <root>/data)
    MLIPDIFF_VALIDATION  raw sampling outputs       (default: <root>/validation)
    MLIPDIFF_WORK        scratch / intermediates    (default: <root>/work)
    MLIPDIFF_MACE_MODEL  MACE-OFF24 checkpoint      (default: ~/.cache/mace/...)
"""

import os

# Auto-detect: this file lives in <root>/utils/paths.py
_DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get('MLIPDIFF_ROOT', _DEFAULT_ROOT)

DATA = os.environ.get('MLIPDIFF_DATA', os.path.join(ROOT, 'data'))
VALIDATION = os.environ.get('MLIPDIFF_VALIDATION', os.path.join(ROOT, 'validation'))
WORK = os.environ.get('MLIPDIFF_WORK', os.path.join(ROOT, 'work'))

# Convenience aliases used by the analysis scripts
EVAL_RESULTS = os.path.join(WORK, 'eval_results')
OUTPUTS_VSCODE = os.path.join(WORK, 'outputs_vscode')

MACE_MODEL = os.environ.get(
    'MLIPDIFF_MACE_MODEL',
    os.path.expanduser('~/.cache/mace/MACE-OFF24_medium.model'),
)



def ensure(*dirs):
    """Create directories if missing, so a script can write without pre-setup."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
