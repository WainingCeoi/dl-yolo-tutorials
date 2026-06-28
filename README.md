# DL & YOLO Tutorials

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)
![Managed with uv](https://img.shields.io/badge/managed%20with-uv-purple.svg)

> A hands-on, self-contained playground for learning **deep learning with PyTorch** and **object detection with YOLO** — two tutorial repos consolidated into one tidy, uv-managed monorepo.

It is organized as **two learning tracks** that share a single Python environment:

| Track | Prefix | You learn to… |
|-------|--------|---------------|
| **Deep Learning (PyTorch)** | `dl_` | build neural networks from scratch — tensors → a classifier → a CNN |
| **Object Detection (YOLO)** | `yolo_` | run, apply, and **train** YOLO models with [Ultralytics](https://docs.ultralytics.com/) |

Within `lectures/` and `projects/`, every file is prefixed by its track. `lectures/` are guided walkthroughs (concepts + code); `projects/` are small end-to-end applications.

---

## Tutorial structure

### Track 1 — Deep Learning with PyTorch (`dl_*`)

Follow the lectures in order, then try the projects. Notebooks live in `lectures/`, scripts in `projects/`.

| Step | File | What you do | Key concepts |
|------|------|-------------|--------------|
| 1 | `lectures/dl_0_check_env.ipynb` | Verify the environment is working | `torch`, Apple-Silicon (MPS) check |
| 2 | `lectures/dl_1_data_preprocessing.ipynb` | Turn a raw CSV into model-ready data | pandas, encoding categorical labels |
| 3 | `lectures/dl_2-4_tensor_basic_operation.ipynb` | Tensor fundamentals | creation, indexing, math ops |
| 4 | `lectures/dl_5-9_fc_nn_w_iris.ipynb` | Build a fully-connected classifier on the Iris dataset | `nn.Module`, training loop, evaluation, save/load |
| 5 | `lectures/dl_14-19_cnn_w_MNIST.ipynb` | Build a CNN that recognizes handwritten digits | convolution & pooling layers, `DataLoader`, train/test, inference |
| 6 | `projects/dl_nn_w_water_quality.py` | End-to-end script: predict water **potability** | train/test split, full training loop |
| 7 | `projects/dl_test_on_cnn.py` | Run the trained MNIST CNN on your own digit images | loading a saved model, image inference |

> Lecture numbers in the filenames map to the original [Codemy.com PyTorch course](https://youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1).

### Track 2 — Object Detection with YOLO (`yolo_*`)

| Step | File | What you do | Key concepts |
|------|------|-------------|--------------|
| 1 | `lectures/yolo_1_running_yolo.py` | Run a pretrained YOLO model on an image | `ultralytics.YOLO`, `predict` |
| 2 | `lectures/yolo_2_yolo_w_webcam.py` | Live detection from a webcam | streaming inference |
| 3 | `projects/yolo_1_car_counter.py` | Count vehicles crossing a region in a video | `ObjectCounter` solution, regions |
| 4 | `projects/yolo_2_people_counter.py` | Count people crossing a line | line counting, class filtering |
| 5 | `projects/yolo_3.1_ppe_detector_training.py` | **Train** a custom PPE detector on a labeled dataset | `model.train`, dataset YAML |
| 6 | `projects/yolo_3.2_ppe_detector_deploy.py` | Deploy your trained detector on webcam/video | loading `best.pt`, real-time predict |

> A 4th project (poker-hand detector) from the source course was intentionally skipped — it is an algorithm exercise rather than a YOLO one.
>
> Reference: [tutorial video](https://youtu.be/WgPbbWmnXJ8) · [Ultralytics docs](https://docs.ultralytics.com/) · [solutions](https://docs.ultralytics.com/solutions/) · labeling with [Label Studio](https://labelstud.io/) · datasets from [Roboflow Universe](https://universe.roboflow.com/).

---

## Repository layout

```
dl-yolo-tutorials/
├── lectures/     # guided walkthroughs — dl_*.ipynb (PyTorch) + yolo_*.py
├── projects/     # end-to-end apps — dl_*.py + yolo_*.py
├── datasets/     # grouped by data type
│   ├── tabular/                  # iris.csv, water_quality.csv
│   ├── mnist/                    # hand_writing/ samples (+ raw, auto-downloaded)
│   ├── images/                   # YOLO sample images
│   ├── videos/                   # YOLO sample videos
│   └── construction_site_safety/ # PPE detection dataset (manifest only — images re-exported)
├── models/       # weights — IRIS.pt, MNIST.pt, ppe_detector/, pretrained/ (auto-downloaded)
├── pyproject.toml · uv.lock · .python-version
└── LICENSE
```

---

## Getting started

### Prerequisites

- [**uv**](https://docs.astral.sh/uv/) — the package/-environment manager used here
- **Python 3.14** (uv will fetch it automatically if missing)
- The scripts target Apple Silicon via `device="mps"`. On other hardware, change `"mps"` to `"cuda"` (NVIDIA) or `"cpu"`.

### Install

```bash
uv sync          # creates .venv and installs everything from uv.lock
```

### Run

Everything uses paths relative to the **repository root**, so run from here:

```bash
# PyTorch projects
uv run python projects/dl_nn_w_water_quality.py
uv run python projects/dl_test_on_cnn.py

# YOLO projects
uv run python projects/yolo_1_car_counter.py

# Notebooks (the kernel runs from lectures/, which references ../datasets and ../models)
uv run jupyter notebook lectures
```

---

## Datasets & weights

Large, regenerable files are **not** committed — they download or re-export on demand:

| Asset | Status | How to get it |
|-------|--------|---------------|
| MNIST raw data | git-ignored | auto-downloaded on first run into `datasets/mnist/` (torchvision creates a `MNIST/` subfolder there) |
| Pretrained YOLO weights | git-ignored | Ultralytics fetches them into `models/pretrained/` on first use |
| `construction_site_safety` images (~163 MB) | git-ignored | re-export from [Roboflow Universe](https://universe.roboflow.com/) into `datasets/construction_site_safety/` (the `data.yaml` manifest is committed) |

Committed and ready to use: the small CSVs, MNIST handwriting samples, YOLO sample images/videos, and the trained PPE weights (`models/ppe_detector/weights/best.pt`).

---

## Suggested learning path

1. **PyTorch basics** → `dl_0` → `dl_1` → `dl_2-4`
2. **Your first network** → `dl_5-9` (Iris) → project `dl_nn_w_water_quality`
3. **Going convolutional** → `dl_14-19` (MNIST CNN) → project `dl_test_on_cnn`
4. **Using YOLO** → `yolo_1` → `yolo_2` → projects `yolo_1_car_counter`, `yolo_2_people_counter`
5. **Training your own detector** → `yolo_3.1` (train) → `yolo_3.2` (deploy)

---

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Consolidated from the `deep-learning-exercise` and `yolo-tutorial` repositories. Course material credit: [Codemy.com](https://youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1) (PyTorch) and the [Ultralytics YOLO tutorial](https://youtu.be/WgPbbWmnXJ8).
