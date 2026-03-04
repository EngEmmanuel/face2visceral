# (Face2Visceral) Visceral Fat Prediction from Facial Images (6hr Hackathon)
>>>>>>> 51774e5 (hackathon end)

Predict visceral fat ratio from facial imagery using a PyTorch Lightning model and a lightweight FastAPI inference service.

## Quick Demo (for visitors)

This is the fastest way to interact with the project.

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start the inference API (with a trained checkpoint)

```bash
python scripts/serve/inference_api.py \
	--checkpoint artifacts/face_to_visceral/checkpoints/<best>.ckpt \
	--port 8000
```

### 3) Start the frontend

```bash
cd elegant
python -m http.server 5500
```

Open http://localhost:5500/index.html and upload a photo.

If you need to test API reachability directly, use:

```bash
curl -s http://127.0.0.1:8000/health
```

### Security

- In the default setup above (`localhost:5500` frontend and `127.0.0.1:8000` API), selfie uploads are sent only to your own machine and processed locally.
- They are not sent to a third-party cloud service by this project’s demo flow.
- This changes if you expose these ports to a network (for example via public hosting, port forwarding, or tunneling) or modify the frontend/API URL targets.

---

## Inference / Testing

Use these commands when you already have a trained checkpoint.

### API endpoints
- `GET /health`
- `POST /predict` (`multipart/form-data`, file field: `image`)

### Batch prediction (paired face + CT)

```bash
python scripts/predict/predict_face_visceral_ratio.py \
	--checkpoint artifacts/face_to_visceral/checkpoints/<best>.ckpt \
	--face-input path/to/face_or_folder \
	--ct-input path/to/ct_or_folder
```

Predictions are written to `artifacts/face_to_visceral/predictions.csv`.

---

## Training (model development only)

The UTKFace and AATTCT/CT dataset steps below are only required for training.

### Data preparation for training

### 1) Build UTKFace balanced subset

```bash
python scripts/setup/create_utkface_balanced_subsample.py \
	--source-dir /path/to/UTKFace \
	--output-dir data/utkdataset
```

### 2) Build AATTCT visceral subset

```bash
python scripts/setup/create_aattct_visceral_subset.py
```

### 3) Build paired face/CT manifest

```bash
python scripts/setup/create_face_ct_pairs_by_age_bins.py
```

### Train

```bash
python scripts/train/train_face_to_visceral_lightning.py
```

Checkpoints are saved under `artifacts/face_to_visceral/checkpoints/`.

## Repository structure
- `scripts/setup/` dataset preparation scripts
- `scripts/train/` training entrypoints
- `scripts/predict/` offline prediction scripts
- `scripts/serve/` API serving code
- `scripts/models/` model definitions
- `data/` local datasets (ignored)
- `artifacts/` checkpoints and run outputs

## Contributors

- Emmanuel Oladokun — [🔗 LinkedIn](https://www.linkedin.com/in/emmanuel-oladokun1/)
- Nate Carey — [🔗 LinkedIn](https://www.linkedin.com/in/natecarey5/)
- Mark Higgins
