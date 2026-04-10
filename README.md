## Installation

### Quick install (most users)
HAVEN'T TESTED THIS METHOD YET
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Conda install (if the above fails on libsndfile)
```bash
conda env create -f environment.yml
conda activate seasound
pip install -e .
```