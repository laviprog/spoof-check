<div align="center">

# Spoof Check

> Audio anti-spoofing service — detects whether an audio recording is **bonafide**
> (genuine human speech) or **spoofed** (synthetic, AI-generated, or replayed), powered by a deep
> learning model and served through a web interface.

[![Tests](https://github.com/laviprog/spoof-check/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/laviprog/spoof-check/actions/workflows/test.yml)
[![Linting](https://github.com/laviprog/spoof-check/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/laviprog/spoof-check/actions/workflows/lint.yml)
[![Type Checking](https://github.com/laviprog/spoof-check/actions/workflows/typecheck.yml/badge.svg?branch=main)](https://github.com/laviprog/spoof-check/actions/workflows/typecheck.yml)
[![Coverage](https://raw.githubusercontent.com/laviprog/spoof-check/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/laviprog/spoof-check/blob/python-coverage-comment-action-data/htmlcov/index.html)

![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6-F97316?logo=gradio&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

---

</div>

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
  - [Docker (recommended)](#-docker-recommended)
  - [Local Development](#local-development)
- [Configuration](#configuration)
- [Development](#development)
  - [Commands](#commands)
  - [Project Structure](#project-structure)
  - [Quality Tooling](#quality-tooling)
- [License](#-license)

## Features

- **Local inference** with [Spectra0](https://huggingface.co/MTUCI/spectra_0) — a Wav2Vec2 XLS-R 300M encoder combined with an ECAPA-TDNN classifier
- **Optional external model** — route predictions to a remote anti-spoofing API (with optional OAuth2 password-flow authentication)
- **Web UI** built with Gradio: upload a file, get per-chunk probabilities and a final verdict
- **Long audio support** — files are automatically split into overlapping chunks, predictions are aggregated
- **Common formats** — WAV, MP3, FLAC, OGG; any sample rate (resampled to 16 kHz), stereo downmixed to mono
- **CPU and CUDA** inference, with a dedicated CUDA Dockerfile
- **Model caching** — the model is downloaded from Hugging Face once and reused from the local `models/` directory

## How It Works

```
audio file ──► load & resample (16 kHz, mono)
           ──► split into 4 s chunks (50 % overlap)
           ──► Spectra0: Wav2Vec2 XLS-R ► MLP bridge ► ECAPA-TDNN
           ──► softmax per chunk ──► averaged probabilities
           ──► verdict: bonafide 🟢 / spoof 🔴
```

When the external model is configured, audio is instead uploaded to the remote anti-spoofing API, and windowed predictions are aggregated on the client side if the service does not return a global verdict.

## Quick Start

### 🐳 Docker (recommended)

```bash
git clone https://github.com/laviprog/spoof-check.git
cd spoof-check

make env      # create .env from .env.example, then edit if needed
make build    # docker compose build
make up       # docker compose up -d
```

The web interface is available at **http://localhost:7861** (mapped to `127.0.0.1` only).

> **GPU:** switch the Dockerfile in `docker-compose.yml` to `docker/Dockerfile.cuda`, uncomment the `deploy` section, and set `DEVICE=cuda` in `.env`.

### Local Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install   # uv sync
make env       # create .env
make run       # start the app at http://localhost:7860
```

On the first run the model (~1.2 GB) is downloaded from Hugging Face and cached in `models/`.

## Configuration

All settings are read from environment variables or a `.env` file (see [.env.example](.env.example)).

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `DEBUG` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `ENV` | `dev` | Environment: `dev` (console logs) or `prod` (JSON logs) |
| `FILES_DIR` | `data/files` | Directory for uploaded audio files |
| `MODELS_DIR` | `models` | Directory for cached model weights |
| `ROOT_PATH` | `/spoof-check` | Web root path (useful behind a reverse proxy) |
| `DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `cuda:0`, … |
| `COMPUTE_TYPE` | `float32` | Compute precision |
| `ANTISPOOFING_BASE_URL` | — | Base URL of the external anti-spoofing API. Unset → only the local model is used |
| `ANTISPOOFING_USERNAME` | — | Username for the external API (optional; auth is enabled only when both username and password are set) |
| `ANTISPOOFING_PASSWORD` | — | Password for the external API (optional) |

## Development

### Commands

```bash
make help
```

| Command | Description |
|---|---|
| `make install` | Install all dependencies (including dev) |
| `make run` | Run the app locally |
| `make test` | Run tests with coverage |
| `make lint` | Check code style (ruff) |
| `make format` | Auto-fix style and format code |
| `make typecheck` | Static type checking (ty) |
| `make check` | Lint + typecheck + tests |
| `make hooks` | Install pre-commit hooks |
| `make build` / `up` / `down` / `logs` | Docker Compose shortcuts |
| `make clean` | Remove caches and build artifacts |

### Project Structure

```
spoof-check/
├── src/
│   ├── api/
│   │   ├── anti_spoofing/   # External anti-spoofing API client and schemas
│   │   └── base_client.py   # Base async HTTP client (httpx)
│   ├── core/
│   │   ├── model.py         # Spectra0 architecture (Wav2Vec2 + ECAPA-TDNN)
│   │   └── inference.py     # SpoofDetector: loading, chunking, prediction
│   ├── services/
│   │   └── audio.py         # File handling + detection orchestration
│   ├── web/
│   │   └── gradio_app.py    # Gradio interface
│   ├── config.py            # Settings (pydantic-settings)
│   ├── log_config.py        # Structured logging (structlog)
│   └── main.py              # Entry point
├── tests/                   # Pytest test suite
├── docker/                  # Dockerfile (CPU) and Dockerfile.cuda (GPU)
├── docker-compose.yml
└── Makefile
```

### Quality Tooling

- **[ruff](https://github.com/astral-sh/ruff)** — linting and formatting
- **[ty](https://github.com/astral-sh/ty)** — static type checking
- **[pytest](https://pytest.org)** — tests with branch coverage
- **[pre-commit](https://pre-commit.com)** — hooks for whitespace, YAML/TOML checks, ruff, and ty

CI runs tests, linting, and type checking on every push to `main` and on pull requests (see [.github/workflows](.github/workflows)).

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
