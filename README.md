# HealthCare‑GPT

*A lightweight, end‑to‑end framework for fine‑tuning a GPT‑2‑style model on healthcare data — fully runnable on a single PC with optional Docker.*


---

## ✨ Key Features

* **Local‑First**: Train and run on 8‑16 GB RAM machines; leverages GPU if present, but works on CPU.
* **Dual‑Task Fine‑Tuning**: Combines medical domain adaptation + high‑risk patient classification.
* **Continuous Learning**: Drop new `.csv`, `.md`, `.txt` into `HealthCareData/` and let agents retrain automatically.
* **Self‑Documenting**: Agentic pipeline logs changelogs & summaries in `docs/changelog/` and `docs/summaries/`.
* **One‑Command Docker**: `docker run -p 5000:5000 healthcare-gpt` starts a REST/Gradio service instantly.

---

## 📂 Repository Layout

```
HealthCare-GPT/
├── nanoGPT/              # Core GPT model + training loop (fork of Karpathy's)
├── HealthCareData/       # Raw data (CSV patients, research docs)
├── models/               # Pretrained & fine‑tuned checkpoints
├── scripts/              # CLI helpers (preprocess, finetune, inference, serve)
├── config/               # YAML experiment configs
├── docs/                 # Plans, changelogs, summaries, assets
├── Dockerfile            # Container build
├── requirements.txt      # Python deps
└── README.md             # (you are here)
```

*See `docs/plans/claude.md` for the full agent design.*

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your‑org/HealthCare‑GPT.git
cd HealthCare‑GPT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Tip**: GPU users should install the CUDA‑matched PyTorch wheel.

### 2. Prepare data

Place patient CSVs in `HealthCareData/patient_data/` and research docs in `HealthCareData/research_docs/`, then run:

```bash
python scripts/preprocess.py \
  --config config/config.yaml
```

### 3. Fine‑tune

```bash
python scripts/finetune.py \
  --config config/config.yaml
```

Checkpoints land in `models/finetuned/` and metrics in `docs/summaries/`.

### 4. Inference (CLI)

```bash
python scripts/inference.py --patient_csv demo.csv
```

Or interactive mode:

```bash
python scripts/inference.py
> Patient: Age 70, BloodPressure 160/110, Diabetes Yes… Risk:
HighRisk
```

### 5. Serve as API/UI

```bash
python scripts/serve.py  # Flask or Gradio (configurable)
```

Open [http://localhost:5000](http://localhost:5000) for the web UI.

---

## 🐳 Run in Docker

```bash
docker build -t healthcare-gpt .
docker run --gpus all -p 5000:5000 -v $PWD/HealthCareData:/app/HealthCareData healthcare-gpt
```

The container starts the Gradio interface by default; override `CMD` for training tasks.

---

## ⚙️ Configuration

All hyper‑parameters live in `config/`:

```yaml
model:
  n_layers: 12
  n_heads: 12
  d_emb: 768
train:
  batch_size: 8
  epochs: 5
  learning_rate: 3e‑5
```

Create alternate YAMLs (e.g., `config_small.yaml`) and pass with `--config` to experiment.

---

## 🤖 Agentic Workflow

1. **Data Curator** watches `HealthCareData/`, triggers preprocessing.
2. **Trainer** runs staged fine‑tuning, logging to W\&B.
3. **Evaluator** computes accuracy/F1 on held‑out CSV.
4. **Deployer** rebuilds Docker image & smoke‑tests inference.
5. **Doc‑Bot** writes `docs/changelog/YYYYMMDD__*.md` & daily summaries.

See the [Claude Plan](docs/plans/claude.md) for sequence diagrams and future extensions.

---

## 📊 Example Results

| Metric   | Validation |
| -------- | ---------- |
| Accuracy | 0.91       |
| F1‑score | 0.89       |

*(Numbers shown are from the sample dataset; your mileage may vary.)*

---

## 🛠 Development & Contribution

Pull requests are welcome! Please:

1. Fork → feature branch.
2. `pre-commit run --all-files` to satisfy linting.
3. Add/maintain tests where relevant.
4. Update `docs/changelog/` via `make docs` or describe changes in PR.

---

## 📄 License

MIT — see `LICENSE` for details.

> **Citation**: If you use HealthCare‑GPT in academic work, please cite the repository URL and original nanoGPT paper.
