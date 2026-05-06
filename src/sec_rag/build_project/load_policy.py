from pathlib import Path

import yaml

from sec_rag.build_project.project_structure import (
    CONFIG_DIR,
)

def load_policy(policy, version='v1'):
    with open(str(CONFIG_DIR / policy / Path(policy+'_'+version+'.yml')), 'r') as f:
        return yaml.safe_load(f)