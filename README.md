# Pairs Trading Research Repository
Authored by Andrew Hall and Seth Peterson

## Getting Started

### 1. Create virtual environment
```bash
python -m venv .venv # Windows/Conda

python3 -m venv .venv #MacOS/Linux
```

### 2. Activate virtual environment
```bash
.venv/Scripts/activate # Windows

source .venv/bin/activate #MacOS/Linux/Conda
```

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

### 4. Install requirements
```bash
pip install -r requirements.txt
```

## Replication

### Experiment Files
To run an experiment, simply execute the corresponding experiment py file from the command line as a module. Note that there should be no back slashes or .py extensions in the command.

```bash
python -m research.experiments.experiment1
```