# Demo Run Guide

## 1) Start backend API

Use your active Python environment:

```bash
python scripts/serve/inference_api.py \
	--checkpoint artifacts/face_to_visceral/checkpoints/<best>.ckpt \
	--port 8000
```

## 2) Start frontend server

In a separate terminal:

```bash
cd elegant
python -m http.server 5500
```

## 3) Open the app

```text
http://localhost:5500/index.html
```

## 4) Quick backend health check

```bash
curl -s http://127.0.0.1:8000/health
```

Expected: JSON containing `"status":"ok"`.

## Notes

- Keep backend and frontend running at the same time.

## Security

- In the default setup above (`localhost:5500` frontend and `127.0.0.1:8000` API), selfie uploads are sent only to your own machine and processed locally.
- They are not sent to a third-party cloud service by this project’s demo flow.
- This changes if you expose these ports to a network (for example via public hosting, port forwarding, or tunneling) or modify the frontend/API URL targets.

