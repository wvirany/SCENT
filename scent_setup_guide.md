# SCENT Setup Guide for Killarney Cluster

### Step 1: Load System Modules
```bash
# Start clean
module purge

# Load required modules
module load python/3.11 scipy-stack rdkit/2023.09.5 cuda/12.6

# Verify RDKit is accessible
pip list | grep rdkit
python -c 'import rdkit'
```

### Step 2: Create Virtual Environment
```bash
# Create environment using Alliance Canada best practices
virtualenv --no-download scent_env
source scent_env/bin/activate

# Upgrade pip using their wheelhouse
pip install --no-index --upgrade pip
```

### Step 3: Install Core Dependencies
```bash
# Install packages from Alliance Canada wheelhouse (use --no-index)
pip install --no-index torch torch-geometric dgl gin-config omegaconf numpy pandas tqdm wandb more-itertools typing-extensions pydantic wurlitzer torchmetrics
```

### Step 4: Install SCENT
```bash
# Navigate to your SCENT repository
cd /path/to/your/SCENT

# Install SCENT without dependency checking to avoid version conflicts
pip install --no-deps -e .
```

### Step 5: Verify Installation
```bash
# Test all imports work
python -c "import torch; import rdkit; import gin; import wandb; import rgfn"
```

## Important Notes

- Load RDKit module BEFORE creating virtual environment
- Always use `--no-index` flag for packages available in Alliance Canada wheelhouse
- Use `--no-deps` for SCENT installation to avoid version conflicts
- Alliance Canada provides newer compatible versions than SCENT specifies - this is normal
- This setup avoids heavy dependencies like PyTDC, jupyter, openbabel that cause installation issues

## For Future Sessions

Each time you want to use SCENT:
```bash
# Load the same modules
module load StdEnv/2023 rdkit/2023.09.5 python/3.11 scipy-stack cuda/12.6

# Activate environment
source scent_env/bin/activate

# Ready to run experiments
python train.py --cfg configs/experiments/beta-experiments/test_minimal.gin
```

## Troubleshooting

If you get import errors, likely missing dependencies include:
- `wurlitzer` (for GNEprop proxy imports)
- `torchmetrics` (for metrics)
- `more-itertools` (for replay buffer)

Add them with: `pip install --no-index <package_name>`
