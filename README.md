Clash Royale Bot (Educational, 2025 UI)

This is a modular Clash Royale automation bot scaffold designed for educational use. It plays via ADB on an Android emulator (e.g., BlueStacks), uses OCR + YOLO11 or heuristics (fixed ROIs) for vision, supports a rule-based planner with a bandit, and a PPO RL agent (Stable-Baselines3). It’s Windows‑friendly, dry‑run safe, and optimized for repeatable loops.

Important: This project is for educational purposes only. Do not use it to violate any game’s Terms of Service.

Features
- Vision: OCR (PaddleOCR) and YOLO11 (`ultralytics`) integration. If YOLO is unavailable or low‑confidence, fall back to fixed ROI heuristics. Auto‑saves unknown ROI crops to `data/crops/` for future fine‑tuning.
- State: Tracks elixir, gold, daily wins (OCR), hand cards (OCR placeholder), and tower HPs (placeholder via YOLO+OCR later).
- Planning:
  - Rules + bandit: If elixir ≥ 4, choose card via epsilon‑greedy bandit; deploy defensively if YOLO finds enemy troops near your towers, otherwise push offence into safe zone. Post‑battle, tap rewards and update bandit using gold delta + win bonus.
  - PPO: Discrete action space (card index 0–3, grid 9x5), reward +1 tower damage, −1 loss. Includes offline training from logged episodes.
- Actions: ADB taps/swipes with normalized or pixel centers. Windows/BlueStacks compatible.
- Automation: Battle loop, rewards collection, upgrades. Chests are removed (2025), so progression focuses on wins and upgrades.
- Determinism: `--seed` sets numpy/random seeds.
- Logging: `logs/run.log`, episode JSONL in `logs/episodes/`.
- CI: Py3.11, fast tests with minimal deps.

Quick Start (Windows/BlueStacks)

## Option 1: Automated Local Setup (Recommended for Powerful Hardware)

For local high-performance setup with GPU acceleration (NVIDIA RTX 3090, etc.):

1) **Run the setup script**
```bash
python setup_local.py
```
This will:
- Detect your hardware (CPU, GPU, CUDA)
- Install optimized dependencies (GPU-accelerated PyTorch, PaddleOCR)
- Configure settings for your hardware
- Set up directories and test the installation

2) **Start your emulator**
- Launch BlueStacks or Android emulator
- Connect ADB: `adb connect 127.0.0.1:5555`

3) **Run the bot**
```bash
python main.py --mode play --seed 123
```

## Option 2: Manual Setup

1) Python env
- Install Python 3.11+ (3.8+ minimum)
- `py -3.11 -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`

2) Requirements
- **For GPU acceleration (RTX 3090, etc.)**: `pip install -r requirements-gpu.txt`
  - Includes CUDA-enabled PyTorch and GPU-accelerated PaddleOCR
- **For local full feature set**: `pip install -r requirements.txt`
- **Torch on Windows**: If `pip install torch` fails, install from https://pytorch.org/get-started/locally/ matching your CUDA. CPU works but is slower.
- **For faster test-only setup**: `pip install -r requirements-ci.txt`

3) Configure (ROIs/YOLO)
- **For high-performance setup**: Use `config-gpu.yaml` as template (automatically copied by setup script)
- **Manual config**: Edit `config.yaml` (ADB host/port, OCR, YOLO, RL params, 2025 UI ROIs):
  - `ocr.elixir_roi: [0.4,0.92,0.6,0.98]`
  - `ocr.gold_roi: [0.85,0.02,0.95,0.08]`
  - `vision.battle_roi: [0.4,0.85,0.6,0.92]`
  - `vision.cards_centers: [[0.2,0.92],[0.4,0.92],[0.6,0.92],[0.8,0.92]]`
  - `vision.rewards_roi_center: [0.3,0.4,0.7,0.6]`, `vision.rewards_roi_bottom: [0.4,0.85,0.6,0.92]`
  - `vision.upgrade_roi: [0.45,0.55,0.55,0.65]`
  You may need to adjust ROIs for your BlueStacks layout/screenshots. Tip: Take a screenshot, measure pixel x,y ranges, then divide by width/height to get normalized values.
  - **YOLO**: `vision.enable_yolo: true`, `vision.yolo_model: yolo11n.pt` (imgsz auto-optimized for your hardware).
  - **GPU Settings**: Configure `hardware` section for optimal performance:
    ```yaml
    hardware:
      use_gpu: true              # Enable GPU acceleration
      performance_mode: "high"   # "low", "medium", "high" for RTX 3090
      enable_gpu_ocr: true       # GPU-accelerated OCR
      enable_gpu_yolo: true      # GPU-accelerated YOLO
    ```

4) Run (play loop)
- `python main.py --mode play --seed 123`

5) Train PPO
- Online mock env: `python train.py --timesteps 2000000 --checkpoint checkpoints/run1`
- Offline from logs: `python train.py --timesteps 2000000 --checkpoint checkpoints/run1 --offline`

6) Evaluate
- `python eval.py --episodes 100`

Repo Structure
- `main.py` — Main loop: capture, detect (YOLO+OCR+ROIs), update state, plan/act, log to run.log and JSONL episodes.
- `train.py` — PPO training (online mock env or offline from episodes).
- `eval.py` — Mock evaluation stats.
- `planner/rules.py` — Rules planner with bandit and 2025 progression (rewards, upgrades).
- `planner/ppo.py` — PPO + `MockClashEnv` + `OfflineLogEnv` for training from logs.
- `actions/adb.py` — ADB controller, normalized taps and pixel clicks, dry-run safe.
- `vision/` — `capture.py` (ADB), `ocr.py` (PaddleOCR), `detect.py` (YOLO+heuristics), `templates.py` (ROI helpers).
- `state.py` — Dict-style game state.
- `utils/` — Config loader and logging setup.
- `tests/` — Offline-friendly unit tests with YOLO/OCR mocks.
- `.github/workflows/ci.yml` — CI running tests on push.

Notes
- **BlueStacks**: Ensure ADB is reachable (e.g., `adb connect 127.0.0.1:5555`). Windows display scale 100% recommended.
- **Vision**: If YOLO model is not present or fails to load, detection falls back to ROIs + OCR. Unknown/low‑conf crops save to `data/crops/` as e.g. `unknown_button_*.png` for future fine‑tuning.
- **OCR**: PaddleOCR can be heavy; tests mock OCR and YOLO predictions.
- **GPU Acceleration**: For NVIDIA GPUs (RTX 3090, etc.), the bot automatically detects and optimizes settings:
  - **YOLO**: Higher resolution (640px vs 320px), FP16 precision, GPU inference
  - **OCR**: GPU-accelerated PaddleOCR for faster text recognition
  - **Performance**: ~2-5x faster than CPU-only setup
- **Safety**: All device actions are no-ops in CI/dry-run; set `adb.dry_run: false` to enable real taps.

## Performance Optimization

The bot automatically detects your hardware and optimizes settings:

| Hardware | YOLO Resolution | OCR GPU | Performance |
|----------|----------------|---------|-------------|
| RTX 3090 (24GB) | 640px | Yes | ~5x faster |
| RTX 3070 (8GB) | 512px | Yes | ~3x faster |
| GTX 1060 (6GB) | 416px | Yes | ~2x faster |
| CPU Only | 320px | No | Baseline |

Monitor performance with:
```bash
# Enable performance monitoring in config.yaml
hardware:
  monitor_performance: true
```

License
- MIT. Educational use only. No affiliation with Supercell.
