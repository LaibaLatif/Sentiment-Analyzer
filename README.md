# Multimodal Mental-Health Signal Assistant

> Educational, **multi-agent** prototype that combines a **text** sentiment model, a **face-emotion** CNN, a **pretrained vision benchmark**, and a **deterministic crisis safety net** into a single, explainable Streamlit experience.
>
> ⚠ **Not a medical device.** Outputs are experimental and intended for analysis only. If you or someone you know is in crisis, contact local emergency services or a licensed mental-health professional.

---

## 1. Highlights

| Capability            | Implementation                                                                                                                                                                |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DIP preprocessing     | OpenCV cascade face detection (+ MediaPipe fallback), face crop with padding, CLAHE contrast enhancement, non-local means denoising, blur/brightness/contrast quality scoring |
| Text sentiment        | TF-IDF (word 1–2 + char 3–5) + Logistic Regression with optional GridSearchCV                                                                                                 |
| Face emotion          | PyTorch `SmallCNN` (48×48 RGB, 7 emotions) with **horizontal-flip TTA** at inference                                                                                          |
| Vision benchmark      | DeepFace (when TF available) → automatic fallback to a HuggingFace ViT                                                                                                        |
| Multimodal fusion     | Rule-based: text × face → `Stable / Mixed / Conflict / High Risk`                                                                                                             |
| Crisis safety net     | Regex-based lexicon (suicidal ideation, self-harm, planned attempt, hopelessness) — **deterministic override** when high-severity language is found                           |
| Threshold calibration | NLP cutoff swept on held-out split (no retraining) → `models/nlp_threshold.json`                                                                                              |
| Explainability        | LIME word-level chips + colour-coded text + structured face explanation                                                                                                       |
| Reports               | Plain text, markdown download, browser-printable HTML                                                                                                                         |
| Tests                 | `pytest` suite covering safety lexicon, supervisor decisions, fusion truth table, text cleaning, and an end-to-end smoke test                                                 |

## 2. Architecture (10 agents)

```mermaid
flowchart LR
    T[User text] --> NLP[NLPAgent\nTF-IDF + LR]
    I[User image] --> CNN[CVAgent / CNNAgent\nSmallCNN + TTA]
    I --> DF[DeepFaceAgent\nDeepFace or HF ViT]
    NLP --> SAFE[SafetyAgent\ncrisis lexicon]
    NLP --> FUSE
    CNN --> CMP[ComparisonAgent]
    DF --> CMP
    CMP --> FUSE[FusionAgent\nimage + multimodal]
    FUSE --> SUP[SupervisorAgent\nfinal decision]
    SAFE --> SUP
    SUP --> SUG[SuggestionsAgent\nmode-aware tips]
    SUP --> XAI[XAIAgent\nLIME + summary]
    SUG --> UI[Streamlit UI]
    XAI --> UI
    SUP --> UI
    HPT[HPTAgent\n(training-time)] -. tunes .-> NLP
```

| Agent                  | Role                                                                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `NLPAgent`             | Text sentiment + confidence; uses calibrated threshold when present                                                                 |
| `HPTAgent`             | Training-time GridSearchCV (C / max_iter / solver) for the LR head                                                                  |
| `DIImageProcessor`     | Face detect (cascade + optional MediaPipe fallback), quality assessment, CLAHE + denoise enhancement, confidence band + status note |
| `CVAgent` / `CNNAgent` | Trained PyTorch model; horizontal-flip TTA on by default                                                                            |
| `DeepFaceAgent`        | External benchmark (real DeepFace or HuggingFace ViT fallback)                                                                      |
| `ComparisonAgent`      | CNN vs DeepFace; on disagreement prefers DeepFace                                                                                   |
| `FusionAgent`          | Image fusion + multimodal text-vs-face decision                                                                                     |
| `SafetyAgent`          | Deterministic crisis-keyword override (suicide / self-harm / severe / hopelessness)                                                 |
| `SupervisorAgent`      | Combines all signals, attaches warnings, applies override                                                                           |
| `SuggestionsAgent`     | Mode-aware wellbeing tips (`Crisis / High Risk / Conflict / Mixed / Stable`) — deterministic, fully offline                         |
| `XAIAgent`             | LIME on text + structured face-side explanation                                                                                     |

## 3. Quickstart (Windows / PowerShell)

### 1) Create and activate virtual environment

```powershell
git clone <this repo>
cd dsproject
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Run the app

```powershell
python -m streamlit run app.py
```

Open: `http://localhost:8501`

### 4) Optional health checks

```powershell
python -c "import torch; print('Torch:', torch.__version__)"
python -c "from system import System; s = System(); print('System init OK')"
python -c "from agents.dip_processor import DIImageProcessor; print('DIP init OK:', DIImageProcessor()._mediapipe_available)"
python -m pytest
```

### DIP pipeline flow (image path)

1. Input image converted to RGB/BGR safely.
2. Quality checks run: blur (Laplacian variance), brightness, contrast.
3. Face detection tries OpenCV Haar cascade first; MediaPipe used as fallback when available.
4. If face exists, largest face is cropped with padding; otherwise full image is used.
5. Enhancement runs: CLAHE on luminance channel + non-local means denoising.
6. Output metadata includes: `quality_score`, `quality_issues`, `enhancement_applied`, `confidence_band`, and user-facing `note`.

### Startup issue fixed on this project (WinError 1114 / `c10.dll`)

If you hit this error while importing `System()` or `agents.cnn_agent`:

`OSError: [WinError 1114] ... torch\lib\c10.dll`

use this recovery sequence:

```powershell
python -m pip uninstall torch -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install opencv-python-headless
```

If you later see NumPy/SciPy compatibility warnings, stabilize with:

```powershell
python -m pip install "numpy<2.3" "scipy>=1.14,<1.15"
```

### Run the tests

```powershell
python -m pytest
```

### Recompute evaluation artefacts (no retraining)

```powershell
python -m scripts.eval_nlp                  # writes models/nlp_eval.json + nlp_confusion.png
python -m scripts.calibrate_nlp_threshold   # writes models/nlp_threshold.json
python -m scripts.eval_cnn --tta            # writes models/cnn_eval.json + cnn_confusion.png
```

The Streamlit sidebar and Report tab automatically pick up these JSON / PNG files when present.

## 4. Held-out evaluation

| Model                                    | Acc   | Macro-F1  | Weighted-F1 | ROC-AUC | n_test |
| ---------------------------------------- | ----- | --------- | ----------- | ------- | ------ |
| **NLP** (TF-IDF + LR, threshold 0.50)    | 0.951 | 0.943     | 0.951       | 0.990   | 10,536 |
| **NLP** (calibrated threshold 0.33)      | —     | **0.952** | —           | 0.990   | 10,536 |
| **CNN** (SmallCNN, FER held-out, TTA on) | 0.548 | 0.440     | 0.527       | —       | 7,178  |

CNN scores are typical of small from-scratch FER models; calibrated threshold + TTA give a small lift on the existing weights without retraining. Per-class metrics + confusion matrices are written to `models/`.

## 5. Project layout

```
dsproject/
├── app.py                    # Streamlit entry — thin wrapper around system + UI
├── streamlit_ui.py           # presentation (sidebar, tabs, charts, downloads)
├── streamlit_theme.py        # CSS + decorative HTML primitives
├── system.py                 # System.analyze orchestrator + report builders
├── config.py                 # paths, thresholds, label names
├── text_clean.py             # compiled-regex cleaning helpers
├── training_utils.py         # train_and_save_nlp / train_and_save_cnn
├── notebook_setup.py         # one-line path helper for Jupyter
├── agents/                   # 9 agents — one file per role
├── scripts/                  # post-training utilities (eval, calibration)
│   ├── eval_nlp.py
│   ├── eval_cnn.py
│   └── calibrate_nlp_threshold.py
├── tests/                    # pytest suite
├── notebooks/                # 01_data_preparation … 04_test_models
├── data/                     # text/Combined Data.csv + img/{train,test}/<emotion>/
├── models/                   # saved artefacts (joblib / .pt / JSON / PNG)
├── docs/                     # additional design notes (architecture, walkthroughs)
└── .streamlit/               # config + (gitignored) secrets
```

## 6. Edge cases handled

- **Empty text / no image** → friendly prompt, no model runs.
- **No face detected** → DIP falls back to full-frame enhancement (still analyzable).
- **Low-quality image** (blur/dark/bright/low contrast) → flagged in DIP metadata and shown with quality score + confidence band.
- **DeepFace offline** → CNN-only result with a sidebar warning.
- **CNN vs DeepFace disagreement** → ComparisonAgent picks DeepFace by policy.
- **Low confidence on any signal** → Supervisor downgrades `High Risk` → `Likely High Risk`.
- **Short / typo-laden text** → Supervisor warns and de-emphasises NLP confidence.
- **Crisis language (any obfuscation: `s.u.i.c.i.d.e`, `kms`, `saaaad`)** → SafetyAgent forces `Crisis — seek support` and shows hotlines (region selectable in sidebar: India, US, UK, EU, AU, CA, International).

## 7. Responsible-AI / ethics statement

- The system is **not a diagnostic tool**. The probabilistic models are not validated for clinical use.
- The crisis safety net is intentionally **deterministic** — when high-severity language is detected we override the probabilistic models and surface hotlines, because misclassification cost is asymmetric.
- The text the user types is **not stored** by the app and is **not sent to any external service** (no Gemini / OpenAI dependency in this repo).
- Trained weights are **gitignored from secrets**; `secrets.toml` only contains optional, redacted keys via `secrets.toml.example`.
- The CNN inherits the limitations of the FER-2013 dataset (skewed class balance, posed expressions, demographic bias).

## 8. Reproducibility

| Step                    | Command                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| Install deps            | `pip install -r requirements.txt`                                              |
| Train NLP               | open `notebooks/02_train_nlp.ipynb` (uses `training_utils.train_and_save_nlp`) |
| Train CNN               | open `notebooks/03_train_cnn.ipynb` (uses `training_utils.train_and_save_cnn`) |
| Evaluate NLP            | `python -m scripts.eval_nlp`                                                   |
| Calibrate NLP threshold | `python -m scripts.calibrate_nlp_threshold`                                    |
| Evaluate CNN (with TTA) | `python -m scripts.eval_cnn --tta`                                             |
| Run tests               | `python -m pytest`                                                             |
| Run app                 | `python -m streamlit run app.py`                                               |

## 9. Disclaimer (final)

This project is an **educational prototype** built for a coursework deliverable. The decisions, percentages, and downloadable reports it produces have **no clinical authority**. If you are in crisis, please reach a real human support service via [findahelpline.com](https://findahelpline.com).
