# Data Directory

Session data from the consciousness app is stored here but **not committed to git** (private data).

## What's Stored Here
- `*.h5` / `*.hdf5` - Session data in HDF5 format
- `*.json` - Session metadata and drawing actions
- `*.csv` - Exported CSV data (optional)
- `session_*` - Individual session folders

## Why Data is Excluded
- **Privacy**: Your brainwave (EEG) data is personal
- **Size**: Session files can be large
- **Local Only**: Training data is specific to your consciousness patterns

## Directory Structure
```
data/
├── session_20251101_123456_abc123.h5
├── session_20251101_123456_abc123.json
└── [more sessions...]
```

## Generating Data
Run the app in generate mode:
```bash
python run.py --mode generate
```

## Using Your Data for Training
```bash
python run.py --mode train --data-dir data/
```
