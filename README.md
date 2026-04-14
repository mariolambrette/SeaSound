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


### Dry run:
```
seasound --config config_file.yaml --dry-run
```
Validates config and shows input/output paths - useful for validation before
triggering full run

### Data loading:
```
seasound --config config_file.yaml --load-only
```
Runs only the data loading portion of the pipline (i.e. read .wav file and save
TOB and (if configured) STFT to cache)
Enables faster anlysis downstream as data can be read directly from cache for
analysis pipeline.


