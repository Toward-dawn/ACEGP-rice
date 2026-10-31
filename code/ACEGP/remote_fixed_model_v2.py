from pathlib import Path
import os
from revised_inputs import load_arrays
from public_paths import DATA_ROOT

ROOT = DATA_ROOT
COHORT = os.environ.get("ACEGP_COHORT", "development")


def arrays():
    return load_arrays(ROOT, COHORT)
