# Visceral Fat Prediction from Facial Images (6hr Hackathon)

Rapid-prototyping project for training a model to predict visceral fat from facial images, with a focus on shipping a working demo quickly.

## Project goals
- Train and validate a baseline model quickly.
- Iterate rapidly on experiments.
- Produce a demo-ready inference path.

## Quick start
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment template:

```bash
cp .env.example .env
```

4. Start with notebooks for fast exploration and migrate stable code into `src/`.

## Recommended workflow
- Use PyTorch Lightning for clean training loops.
- Track experiments with Weights & Biases.
- Keep configs in `configs/` (Hydra-ready layout).
- Save demo assets/checkpoints under `artifacts/`.

## Directory overview
- `configs/` - experiment and model configs
- `data/` - raw/interim/processed/external datasets
- `notebooks/` - rapid experiments and EDA
- `src/` - reusable project code
- `scripts/` - CLI scripts for training/eval/inference
- `demos/` - demo application assets and code
- `tests/` - unit/integration tests
- `artifacts/` - model checkpoints and exported artifacts
- `docs/` - project notes and context summaries

## Suggested next steps
- Add a baseline datamodule and regressor in `src/`.
- Create first experiment config in `configs/`.
- Build a minimal demo endpoint/UI in `demos/`.
