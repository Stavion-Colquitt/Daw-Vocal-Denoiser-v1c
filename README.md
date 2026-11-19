> **Moved:** DAW Vocal Denoiser v1c now lives in **[Daw-Vocal-Denoiser/versions/v1c](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser/tree/main/versions/v1c)**, together with every other version,
> one shared codebase and a results table across all versions. This repository is archived (read-only); its full history
> is part of Daw-Vocal-Denoiser.

# DAW Vocal Denoiser v1c

> **AI disclosure:** This project was built with AI assistance (Claude, by Anthropic) for writing code and
> documentation. All creative and technical decisions (model design, training data, evaluation) are mine, and I
> reviewed everything before publishing.

🤗 [Model on Hugging Face](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c) ·

**Later versions:** [v1b](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser/tree/main/versions/v1b) (bigger GRU, best accuracy) · [v1d](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser/tree/main/versions/v1d) (5 ms lookahead, same quality at half the CPU) · 🎧 [Live A/B demo](https://huggingface.co/spaces/Stavion-Colquitt/Realtime_Denoisers) · 📋 [Project board](https://github.com/users/Stavion-Colquitt/projects/2)
v1: [Daw-Vocal-Denoiser](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser)

Version 1c of my [DAW Vocal Denoiser](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser): the same causal,
real-time model (48 kHz, **10 ms algorithmic latency**, 1.25M parameters) retrained with about **four times more of
my own singing**. The question v1c asks: does more in-domain vocal data (my voice, my rooms, my mics) help?

---

## Results (held-out test)

180 noisy clips (+ 20 clean pass-through clips) that no version trained on: my held-out song *Say Dat Name* (86 clips)
and held-out VocalSet singers (94 clips), mixed with dev-split noise at 0-40 dB SNR from a fixed seed. Every version
runs on exactly the same clips through the streaming engine (`compare.py`), so differences are paired.

| Version | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL | FAD-MERT (lower better) | Clean pass-through SI-SDR (dB) |
|---|---|---|---|---|---|---|
| Noisy input | 19.04 | 19.02 | 2.37 | 3.53 | 1.732 | |
| v1 | 22.95 | 23.70 | **3.05** | 3.97 | **0.419** | 45.6 |
| v1c after main run (47k steps) | 23.05 | 23.65 | 2.97 | 3.96 | 0.513 | 50.3 |
| **v1c final (108,779 steps)** | **23.27** | **23.85** | 3.03 | **3.98** | 0.485 | **51.1** |

**By voice**, v1c final vs v1 (paired, mean ± 95% CI):

| Voice | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL |
|---|---|---|---|---|
| My voice (86 clips) | **+0.95 ± 0.67** | **+0.62 ± 0.57** | +0.02 ± 0.02 | **+0.03 ± 0.02** |
| Other singers, VocalSet (94 clips) | -0.27 ± 0.27 | -0.29 ± 0.24 | -0.05 ± 0.02 | -0.02 ± 0.04 |

**What this shows:** four times more of my own singing made the model specialize to my voice, mics and rooms. It
is clearly better on my vocals and slightly worse on other singers, and it leaves clean takes more untouched
(51.1 vs 45.6 dB pass-through). The extra training length mattered: after its main run alone, v1c was not yet better
than v1. For other voices, [v1](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser) is still the better
general-purpose choice. Metric details: see *Evaluation metrics* below.

## Results (dev set)

| | v1 | v1c |
|---|---|---|
| Own vocal training data | 38.8 min, 8 songs | about 202 min, 38 songs |
| Training steps | 108,773 | 108,779 (47,079 main run + 61,700 extension) |
| Dev SI-SDR gain, published checkpoint | +2.39 dB | **+2.57 dB** |
| Peak dev SI-SDR gain during training | +2.39 dB | +2.59 dB (step 95,000) |

At the same number of training steps, v1c is **+0.18 dB** above v1 on the dev set. The published checkpoint is the
one with the lowest dev loss (step 108,779). The held-out test results are above.

| Real-time property | Value |
|--------------------|-------|
| Algorithmic latency | 10 ms (32 ms analysis / 10 ms synthesis windows, 5 ms hop) |
| CPU per 5 ms block (ONNX Runtime, desktop CPU) | 0.24 ms (real-time factor 0.048) |
| ONNX vs PyTorch parity | 113.6 dB |
| Parameters | 1,250,563 |

---

## What changed from v1

- **Data:** 1,246 more of my own vocal takes (163.7 min from 30 songs), pulled from the vocal tracks across my
  Ableton sessions, deduplicated, takes under 1 s skipped. The dev song (`Mad`) and test song (`Say Dat Name`) stay
  out of training. Everything else (VocalSet, DNS Challenge speech and noise, headphone bleed from my instrumentals,
  synthetic rooms, hiss and hum) is the same as v1.
- **Training:** a 47,079-step main run on an RTX 4060, then a 61,700-step extension on an RTX 2060 (learning rate
  re-warmed to 3e-4, cosine to 1e-5) so the total matches v1's step count.
- **Model:** unchanged. Asymmetric analysis/synthesis windows (Wang et al. 2021), compressed magnitude ->
  Linear -> LayerNorm -> 2-layer GRU (256) -> sigmoid magnitude mask, noisy phase kept.

---

## Files

| File | What it does |
|------|--------------|
| `model.py` | Model definition and asymmetric-window STFT |
| `stream.py` | Real-time block engine (240-sample blocks in and out) and `process_file` for offline use |
| `dataset.py` | On-the-fly noisy/clean mixing (SNR 0-40 dB, rooms, bleed, hiss, hum) |
| `prep_data.py` | Builds the v1 training set (resampling, splits) |
| `prep_own_extra.py` | Collects the extra v1c vocal takes from my Ableton sessions |
| `train.py` | Training loop (compressed spectral + multi-resolution STFT loss, AdamW, EMA 0.999, MLflow) |
| `run_v1c_ext.ps1` | The extension run (`--resume --extend_steps 61700`), then ONNX export and parity check |
| `export_onnx.py` | Exports the one-frame streaming ONNX model |
| `check_parity.py` | Checks ONNX vs PyTorch and measures CPU time per block |
| `eval.py` | Held-out test evaluation (SI-SDR, PESQ, pass-through) |
| `compare.py` | Runs several versions on identical clips; paired differences and a by-voice split |
| `results/` | Raw held-out test comparison (`compare_test.md`, `compare_test.json` with per-clip scores) |

The weights (`daw_denoiser.onnx`, `daw_denoiser.pth`) are on
[Hugging Face](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c).
My own vocal recordings are not included.

## Usage

```python
# pip install onnxruntime numpy
from stream import StreamEngine          # expects daw_denoiser.onnx next to it
eng = StreamEngine()
y_block = eng.process(x_block)           # float32[240] at 48 kHz -> 240 denoised samples, 10 ms later

from stream import process_file          # offline: whole mono 48 kHz array in, denoised array out
y = process_file(x)
```

## Evaluation metrics

`eval.py` runs the held-out test through the streaming engine and reports two groups of metrics, noisy input vs
enhanced output:

| Metric | What it measures | Why it's here |
|--------|------------------|---------------|
| SI-SDR (dB) | Waveform closeness to the clean take, ignoring level | Standard enhancement metric; the main number here |
| PESQ-WB | ITU-T P.862.2 predicted speech quality (at 16 kHz) | Kept for comparison with speech denoisers; it was designed for speech, so treat it as a rough guide on singing |
| Clean pass-through SI-SDR | How untouched clean vocals come out | The "do no harm" check for tracking |
| **SDR (BSS-Eval, 512-tap)** | Filter-based SDR, tolerant of small EQ/phase changes | Recommended over SI-SDR for music in recent singing-voice work |
| **ViSQOL v3, audio mode** | Full-band (48 kHz) perceptual quality, MOS-LQO 1 to 4.73 | The usual intrusive quality metric for music; Google's code, built from source in Docker (`tools/visqol`) |
| **MERT-MSE** | Distance between [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) music embeddings of output and clean take | Embedding metrics track listening tests better than SDR on singing |
| **FAD-MERT** | Fréchet distance between all outputs and all clean takes in MERT embedding space | Distribution-level check that outputs "sound like" clean vocals |

Before ViSQOL and MERT, every signal is loudness-normalized to -23 LUFS (EBU R128) so level differences don't bias
the scores. Results are also broken down (mean improvement with 95% confidence intervals) by input SNR band, noise
type (DNS noise, headphone bleed, hiss, hum) and vocal source, and every item is written to `eval_results_items.csv`.

```bash
pip install -r requirements-eval.txt          # torch is assumed to be installed already
docker build -t visqol:local tools/visqol      # one time; eval.py also builds it automatically if missing
python eval.py 200                             # held-out test
python eval.py 40 --split dev                  # check the pipeline on dev vocals without touching the test set
python eval.py 200 --no-visqol --no-mert       # speech-style metrics + SDR only (no Docker, no model download)
```

The test mixtures come from a fixed seed, so every run scores exactly the same clips. Objective metrics are still
only a proxy: a listening test (MUSHRA, ITU-R BS.1534) on real noisy takes is the final word.

References: Le Roux et al. 2019 (SI-SDR); Vincent et al. 2006 and Scheibler 2022 (BSS-Eval, fast_bss_eval);
Chinen et al. 2020 (ViSQOL v3); Li et al. 2023 (MERT); Kilgour et al. 2019 (FAD); Gupta et al. 2015,
"On the Perceptual Relevance of Objective Source Separation Measures for Singing"; "Teaching Speech Enhancement
Models To Sing" ([arXiv 2607.11630](https://arxiv.org/abs/2607.11630)) on SDR, MERT-MSE and loudness matching for singing.

## Docker

A CPU-only image (ONNX Runtime, no PyTorch) is on Docker Hub as
[`stavion/daw-denoiser`](https://hub.docker.com/r/stavion/daw-denoiser). Any sample rate, mono or stereo; output is
24-bit WAV at the input's rate.

```bash
# denoise a file in the current folder
docker run --rm -v "${PWD}:/data" stavion/daw-denoiser vocal.wav vocal_clean.wav

# or run it as a small HTTP service: POST audio to /denoise, get the cleaned WAV back
docker run --rm -p 8000:8000 stavion/daw-denoiser serve
curl -X POST --data-binary @vocal.wav http://localhost:8000/denoise -o vocal_clean.wav
```

To build it yourself, put `daw_denoiser.onnx` (from Hugging Face) next to the `Dockerfile` and run
`docker build -t daw-denoiser .`. The container's output matches `stream.py` run directly (122 dB, limited by the
24-bit WAV output).
## Data paths

Every data location is an environment variable with a default inside the repo, so a fresh clone works without
editing code:

| Variable | Default | What goes there |
|----------|---------|-----------------|
| `DAW_DATA` | `data/daw-denoiser` | Training pool written by `prep_data.py` (own_vocals/, vocalset/, bleed/) |
| `DNS_DATA` | `data/dns` | [DNS Challenge](https://github.com/microsoft/DNS-Challenge) fullband data: `clean_fullband/` and `noise_fullband/` |
| `DAW_CORPUS` | `data/corpus` | Your own vocal takes, one folder per song (`<song>/*.wav`), read by `prep_data.py` |
| `DAW_MUSIC` | `data/ableton-music` | Ableton project folders scanned by `prep_own_extra.py` for extra vocal takes |

On Windows, set one for the current shell with `$env:DAW_DATA = "E:\my\data"` (PowerShell) or permanently with
`setx DAW_DATA "E:\my\data"`. On Linux/macOS: `export DAW_DATA=/my/data`. `data/` is gitignored.

## Training data and licenses

| Source | License |
|--------|---------|
| My own vocal takes and instrumentals | Mine (not distributed) |
| [VocalSet](https://zenodo.org/records/1442513) (Wilkins et al., ISMIR 2018) | CC BY 4.0 |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) fullband speech and noise (subsets) | Per source, see the DNS repo |

No commercial sample packs. Code is MIT licensed (see `LICENSE`).

## Limits

- For monitoring and cleaning vocal takes (singing or speech) recorded in untreated rooms.
- Removes noise, not reverb. Mono, 48 kHz only (resample other rates first).
- Not tested on heavy distortion or SNR below 0 dB; very loud headphone bleed can survive.
