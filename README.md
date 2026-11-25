# Real-Time Indian Number Plate Recognition (ANPR)

Detects number plates in images, video files, a webcam or an RTSP stream and shows the **plate text with a
confidence probability** on every box.

```
frame → YOLO plate detector → ByteTrack IDs → crop → EasyOCR → Indian-format repair
      → per-plate voting over frames → green box + "RJ 14 AB 1234  93%"
```

| Stage | What it does | File |
|---|---|---|
| Detect | YOLO (trained in Colab on the Kaggle dataset) finds the plate box | `anpr/detector.py` |
| Track | ByteTrack gives each plate a stable ID across frames | `anpr/detector.py` |
| Read | EasyOCR on the cropped, upscaled, contrast-enhanced plate; two-row plates supported | `anpr/ocr.py` |
| Repair | Fixes O↔0, I↔1, S↔5, B↔8 … using the Indian plate layout and valid state codes | `anpr/plate_rules.py` |
| Vote | Character-level voting over the last 15 reads; locks once stable (saves FPS) | `anpr/tracks.py` |
| Draw/log | Boxes, labels, FPS HUD, CSV log, best crop per plate | `anpr/visualize.py`, `anpr/logger.py` |

## Project structure

```
anpr-realtime/
├── main.py                      # entry point (webcam / video / image / folder / stream)
├── config.yaml                  # all thresholds and options
├── requirements.txt
├── README.md
├── anpr/
│   ├── __init__.py
│   ├── config.py                # YAML loader with defaults
│   ├── detector.py              # YOLO detect + track
│   ├── ocr.py                   # EasyOCR + preprocessing + 2-row handling
│   ├── plate_rules.py           # Indian plate regex / state codes / character repair
│   ├── tracks.py                # per-plate voting and lock
│   ├── pipeline.py              # glues everything together
│   ├── sources.py               # threaded webcam capture, video file capture
│   ├── visualize.py             # drawing
│   └── logger.py                # CSV + crops
├── tests/                       # pytest (no GPU / weights needed)
├── notebooks/
│   └── 01_train_plate_detector_colab.ipynb
├── weights/                     # put best.pt here
├── samples/                     # your test images / videos
└── outputs/                     # annotated results, plate_log.csv, plate_crops/
```

## 1. Train the detector (Colab)

1. Open `notebooks/01_train_plate_detector_colab.ipynb` in Google Colab, set **Runtime → T4 GPU**.
2. Run all cells (about 20–40 min). It downloads the Kaggle dataset, validates it, trains YOLO, evaluates,
   runs a "road-scene" stress test and prints the recommended confidence threshold.
3. Download `best.pt` and put it in `weights/best.pt`.

## 2. Set up VS Code

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate

# PyTorch first - choose the command for your machine at https://pytorch.org/get-started/locally/
# NVIDIA GPU (e.g. your ThinkPad P1 with 4 GB VRAM), CUDA 12.1 build:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:  pip install torch torchvision

pip install -r requirements.txt
```
Select the `.venv` interpreter in VS Code (Ctrl+Shift+P → *Python: Select Interpreter*).

## 3. Run

```bash
python main.py --source 0                        # webcam, press q to quit
python main.py --source samples/traffic.mp4 --save
python main.py --source samples/car.jpg --save
python main.py --source samples/ --save --no-show
python main.py --source 0 --conf 0.45 --imgsz 480   # faster / stricter
python main.py --source samples/traffic.mp4 --no-ocr # detector only
```

Outputs go to `outputs/`: annotated media, `plate_log.csv` (one row per plate) and `plate_crops/`.

```bash
python -m pytest tests -q
```

## Reading the confidence numbers

* **Big label `RJ 14 AB 1234  93%`** – combined score = √(detector confidence × OCR confidence).
* **Small line `det 95% | ocr 91%`** – the two parts. A `*` means the plate is *locked* (stable).
* Colours: **green** = valid Indian plate layout, **orange** = text read but layout is invalid
  (probably a wrong read), **blue-grey** = plate found, text not read yet.
* These are model scores, **not calibrated probabilities** (same caveat as in Lab 11). Use them to rank
  and filter results, and tune thresholds on your own footage.

## Tuning for speed and accuracy

| Problem | Try |
|---|---|
| Low FPS on 4 GB GPU | `imgsz: 480`, `ocr.every_n_frames: 8`, `ocr.max_per_frame: 1`, train/use `yolo11n` |
| Plates far away are missed | `imgsz: 960` (slower), lower `detector.conf` to 0.25 |
| Many false boxes | raise `detector.conf` to the best-F1 value from Colab, e.g. 0.45 |
| Wrong characters | raise `ocr.lock_votes`, make sure `min_sharpness` is not too low, fine-tune OCR (see below) |
| Plate text stays unread | `min_plate_width` may be too high, or crop too blurry (`min_sharpness: 0` to test) |

## Known limits and next steps

* The dataset is detection-only. EasyOCR is **pretrained, not fine-tuned**; accuracy is best on clear,
  front-facing plates. To go further, collect plate crops (the logger already saves them), label the text
  and fine-tune an OCR model (e.g. `fast-plate-ocr`, PaddleOCR, or a CRNN).
* The training images are mostly close-ups; real traffic footage has smaller, blurred plates. The Colab
  notebook trains with strong scale augmentation and includes a stress test, but you should add some real
  road frames of your own for the best result.
* ANPR processes personal data. Only use it on footage you are permitted to record, store logs securely and
  follow local rules.
