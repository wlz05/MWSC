# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

MWSC detects temporal inconsistencies between speech and visual motion. Dense audio and video encoders produce feature sequences, which are aligned by presentation timestamps and compared through multi-scale wavelet coherence. A separate pretrained SyncNet branch reports lip-sync confidence, distance, and audio-video offset.

---

## Architecture

```text
Video + Audio
     |
     +-- MWSC
     |    Grayscale frames + log-STFT
     |              |
     |    Dense video/audio encoders
     |              |
     |    Timestamp-constrained DTW + interpolation
     |              |
     |    PCA -> Morlet CWT -> scale coherence
     |              |
     |    Gaussian fusion -> MSSC -> decision
     |
     +-- SyncNet v2
          Color face tracks + MFCC
                    |
          Pretrained audiovisual embeddings
                    |
          Offset search -> LSE-C / LSE-D / audio lag
```

### Main Components

- Dense audiovisual encoders preserve temporal resolution. The compact architecture is the default; dense I3D and VGGish variants are also available.
- Timestamp-constrained DTW corrects sampling differences without warping away content mismatch.
- Morlet wavelets measure coherence at eight frequencies between 0.5 and 4 Hz.
- Gaussian-weighted coherence produces MSSC. Detection supports frozen thresholds and stream-specific adaptive thresholds.
- SyncNet v2 provides an independent lip-sync score using its official pretrained weights.

---

## Project Structure

```text
MWSC/
├── core/                   # Signal processing, alignment, media, and face crops
│   ├── align.py
│   ├── audio.py
│   ├── face.py
│   ├── media.py
│   ├── threshold.py
│   └── wavelet.py
├── model/                  # Dense encoders and pretrained SyncNet
│   ├── encoder.py
│   ├── expert.py
│   ├── extract.py
│   ├── sound.py
│   ├── vision.py
│   └── weights.py
├── dataset/                # Dataset import, identity splits, and real-pair sampling
│   ├── adapter.py
│   ├── manifest.py
│   ├── samples.py
│   ├── speech.py
│   └── split.py
├── engine/                 # Training, inference, evaluation, and DDP
│   ├── detect.py
│   ├── evaluate.py
│   ├── lipsync.py
│   ├── loss.py
│   ├── parallel.py
│   ├── stream.py
│   ├── sync.py
│   └── train.py
├── support/
│   ├── download.py
│   ├── metrics.py
│   └── storage.py
├── verify/                 # Unit and integration tests
├── catalog.py              # Dataset references
├── check.py
├── config.py
├── install.py
├── main.py                 # Command-line interface
├── notes.py
├── run.py                  # Four-seed distributed training
└── README.md
```

---

## Requirements

| Dependency | Version / use |
| --- | --- |
| Python | 3.10 or newer |
| PyTorch | 2.2 or newer; CUDA build with NCCL for multi-GPU training |
| NumPy | 1.26 or newer |
| SciPy | 1.11 or newer |
| OpenCV | 4.8 or newer; Haar detection requires OpenCV 4.x or OpenCV 5 contrib |
| PyAV | 12 or newer for container audio and presentation timestamps |
| FFmpeg + FFprobe | Alternative container decoder when PyAV is unavailable |

### Installation

```bash
git clone https://github.com/wlz05/MWSC.git
cd MWSC
python -B install.py --install --lipsync
python -B main.py doctor
```

Install the CUDA build of PyTorch for the training machine using the [official installation selector](https://pytorch.org/get-started/locally/). The eight-GPU training configuration uses Linux and NCCL.

---

## Pretrained Models

### SyncNet v2

```bash
python -B main.py fetch --example-output ../runtime/mwsc/example.avi
```

The official model is downloaded from Oxford and saved to `../runtime/mwsc/syncnet.model`. Downloads are validated before replacing the temporary file. An alternative local model can be supplied with `--syncnet-weights`.

### MWSC Encoders

Train the encoders on real audiovisual pairs before running MWSC detection. The SyncNet checkpoint and the MWSC checkpoint serve different networks. Optional I3D/VGGish backbone checkpoints must match the dense architecture's expected tensor names and shapes.

---

## Quick Start

### Lip-sync Scoring

Score the official face-cropped example:

```bash
python -B main.py lipsync --video ../runtime/mwsc/example.avi --face-cropped --device cpu --output ../runtime/mwsc/example.json
```

Score a talking-head video with automatic single-face detection:

```bash
python -B main.py lipsync --video ../videos/person.mp4 --output ../runtime/mwsc/person.json
```

Select a face manually when several people are visible:

```bash
python -B main.py lipsync --video ../videos/person.mp4 --face-box 200 100 450 400 --output ../runtime/mwsc/person.json
```

The box is expressed as `left top right bottom` in the original frame. Use `--face-cropped` only for videos that have already been cropped around a face. External WAV input is supported with `--audio`; `--audio-offset` specifies its physical start on the video timeline.

### MWSC Detection with SyncNet Scores

```bash
python -B main.py predict --video ../videos/person.mp4 --checkpoint ../runtime/mwsc/experiments/seed42/encoders.pt --no-update --lipsync --output ../runtime/mwsc/prediction.json
```

The report contains the MWSC decision and video-level scores under `scores`, with SyncNet measurements under `lip_sync`.

| Field | Meaning |
| --- | --- |
| `scores.mwsc_mean` | Mean MSSC over valid windows |
| `scores.mwsc_min` | Minimum MSSC over valid windows |
| `scores.mwsc_normalized_mean` | Mean MSSC divided by the configured Gaussian weight sum |
| `lip_sync.lse_c` | SyncNet confidence; higher is better |
| `lip_sync.lse_d` | Minimum embedding distance; lower is better |
| `lip_sync.zero_offset_distance` | Distance at the original audio-video alignment |
| `lip_sync.audio_lag_ms` | Positive: audio lags video; negative: audio leads video |
| `lip_sync.coverage` | Fraction of requested time covered by scored face tracks |

LSE-C measures the strength of the best offset match. A delayed clip can still have high confidence, so inspect the estimated lag as well. These scores are not fake probabilities. Missing evidence produces `null` scores and a reason.

---

## Dataset Preparation

| Dataset | Adapter | Task and annotations |
| --- | --- | --- |
| [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) | `fakeavceleb` | Real/real, real/fake, fake/real, and fake/fake video-audio combinations for audiovisual deepfake detection |
| [LAV-DF](https://github.com/ControlNet/LAV-DF) | `lavdf` | Local manipulation detection with fake intervals and audio/video modification flags |
| [AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M) | `avdeepfake1m` | Audiovisual manipulation detection and localization with modality-specific intervals |
| [DFDC](https://ai.meta.com/datasets/dfdc/) | `dfdc` | Cross-dataset visual manipulation testing; original-video metadata links derivatives |
| [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) | `lrs2` | Real speech pairs for encoder training and controlled audio-offset tests |
| [LRS3](https://arxiv.org/abs/1809.00496) | `lrs3` | TED/TEDx speech pairs for encoder training and cross-domain synchronization tests |
| [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) | `voxceleb2` | Real audiovisual speaker clips for representation training and identity-based splits |
| Custom CSV | `custom` | User datasets with authenticity labels, local intervals, and independent synchronization labels |

Encoder training selects real audiovisual pairs from the training split. FakeAVCeleb, LAV-DF, AV-Deepfake1M, and DFDC supply manipulation labels for detection evaluation. LRS2, LRS3, and VoxCeleb2 supply real speech clips for representation training and controlled synchronization tests. Controlled offsets use independently verified synchronized real clips.

The SyncNet scores in the example table use the [official SyncNet demonstration video](https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi).

Example preparation for FakeAVCeleb:

```bash
python -B main.py import --dataset fakeavceleb --root ../datasets/fakeavceleb --output ../runtime/mwsc/imported.csv
python -B main.py split --manifest ../runtime/mwsc/imported.csv --output ../runtime/mwsc/split.csv --seed 42
python -B main.py audit --manifest ../runtime/mwsc/split.csv --check-files --decode
```

### Other Dataset Imports

```bash
python -B main.py import --dataset lavdf --root ../datasets/lavdf --metadata ../datasets/lavdf/metadata.min.json --output ../runtime/mwsc/lavdf.csv
python -B main.py import --dataset avdeepfake1m --root ../datasets/avdeepfake1m --metadata ../datasets/avdeepfake1m/train_metadata.json ../datasets/avdeepfake1m/val_metadata.json --output ../runtime/mwsc/avdeepfake1m.csv
python -B main.py import --dataset dfdc --root ../datasets/dfdc/parta --metadata ../datasets/dfdc/parta/metadata.json --output ../runtime/mwsc/dfdc.csv
python -B main.py import --dataset lrs2 --root ../datasets/lrs2/main --filelist ../datasets/lrs2/train.txt --subset train --output ../runtime/mwsc/lrs2_train.csv
python -B main.py import --dataset lrs3 --root ../datasets/lrs3/trainval --subset train --output ../runtime/mwsc/lrs3_train.csv
python -B main.py import --dataset voxceleb2 --root ../datasets/voxceleb2 --output ../runtime/mwsc/voxceleb2.csv
python -B main.py import --dataset custom --metadata ../datasets/custom/manifest.csv --output ../runtime/mwsc/custom.csv
```

Metadata paths refer to local dataset files. Speech file-list entries are relative to `--root`; `--audio-root` pairs video entries with WAV files in a parallel directory. Import each speech subset separately with its `--subset`, then merge the CSV files through `--dataset custom --metadata`.

Use `split --keep-official` for a complete, labeled official partition and `split --seed 42` for a grouped research split. Both routes audit identity and source leakage. Use `--identity-map` with a `source,group` CSV when subject IDs are stored separately.

### Manifest Format

```csv
dataset,video,audio,label,group,source,split,duration,audio_offset,fake_segments,sync_label,offset_seconds
custom,../datasets/custom/real.mp4,,0,alice,clip001,train,4,0,[],0,0
custom,../datasets/custom/fake.mp4,,1,bob,clip002,test,4,0,"[[1.0,2.0]]",,
```

Paths are relative to the command's working directory. Labels use `0 = real` and `1 = fake`. Optional `audio_segments` and `video_segments` preserve modality-specific intervals; their union contributes to `fake_segments`.

`group` identifies people, while `source` links derivatives of the same original video. Splitting keeps connected identities and original sources together. Supply an identity map when the source metadata only identifies video families. `identity_namespace` and `source_namespace` link IDs across related datasets.

Synchronization truth is separate from authenticity: `sync_label` is `0 = synchronized`, `1 = asynchronous`, or empty when unknown. `offset_seconds` is a measured audio lag. A real label alone does not establish synchronization.

---

## Training

### Experiment Configuration

Our experiment configuration uses **8 NVIDIA RTX 4090 GPUs** and training seeds **42, 52, 3047, 3407**. The data split uses seed 42.

| Setting | Value |
| --- | --- |
| GPUs | 8 × NVIDIA RTX 4090 |
| Training seeds | 42, 52, 3047, 3407 |
| Split seed | 42 |
| Distributed training | One DDP process per GPU, NCCL |
| Per-GPU batch size | 2 |
| Global batch size | 16 |
| Learning rate | 0.0001 |
| Optimizer | AdamW |
| Epochs per seed | 20 |

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -B run.py --manifest ../runtime/mwsc/split.csv --devices 8 --seeds 42 52 3047 3407 --epochs 20 --batch-size 2 --workers 2
```

Each seed uses all eight GPUs through one DDP process per GPU. Seeds run sequentially, with separate checkpoints:

```text
../runtime/mwsc/experiments/
├── seed42/encoders.pt
├── seed52/encoders.pt
├── seed3047/encoders.pt
└── seed3407/encoders.pt
```

`--batch-size` and `--workers` are per process. Batch size 2 on eight devices gives a global batch size of 16; workers 2 creates 16 data-loading workers. Contrastive negatives are local to each rank. Validation samples are sharded without padding, and losses are reduced by sample count. Only rank zero writes the checkpoint.

Preview the launch commands:

```bash
python -B run.py --manifest ../runtime/mwsc/split.csv --devices 8 --dry-run
```

### One Seed with torchrun

```bash
torchrun --standalone --nproc-per-node=8 main.py train --manifest ../runtime/mwsc/split.csv --checkpoint ../runtime/mwsc/experiments/seed42/encoders.pt --seed 42 --epochs 20 --batch-size 2 --workers 2 --device cuda
```

### Single GPU

```bash
python -B main.py train --manifest ../runtime/mwsc/split.csv --checkpoint ../runtime/mwsc/experiments/seed42/encoders.pt --seed 42 --epochs 20 --batch-size 2 --device cuda:0
```

### Dense I3D / VGGish

```bash
python -B run.py --manifest ../runtime/mwsc/split.csv --devices 8 --architecture i3d_vggish --batch-size 1 --video-backbone ../weights/i3d.pt --audio-backbone ../weights/vggish.pt
```

The dense variants retain temporal resolution. Start with batch size 1 per device and measure memory use before increasing it. Resume a trained MWSC checkpoint through `main.py train --resume`.

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--devices` | `8` | Processes/devices per seed in `run.py` |
| `--seeds` | `42 52 3047 3407` | Training seeds in `run.py` |
| `--epochs` | `20` | Epochs per seed in `run.py` |
| `--batch-size` | `2` | Samples per process |
| `--workers` | `2` | Loading workers per process in `run.py` |
| `--learning-rate` | `0.0001` | AdamW learning rate |
| `--architecture` | `compact` | `compact` or `i3d_vggish` |
| `--image-size` | `224` | Video encoder input size |
| `--maximum-steps` | unset | Optional limit per epoch for a short run |
| `--config` | unset | Relative JSON configuration |

---

## Evaluation

Detection evaluation reports ROC-AUC, EER, accuracy, precision, recall, and F1. Temporal localization reports window proposal AP at IoU thresholds 0.3, 0.5, and 0.7. Synchronization evaluation reports offset error and synchronization classification separately.

Lip-sync evaluation uses [SyncNet's LSE-C and LSE-D](https://github.com/joonson/syncnet_python), also used in [AVGen-Bench](https://github.com/microsoft/AVGen-Bench). MSSC measures multi-scale audiovisual coherence within MWSC.

| Metric | Task | Direction |
| --- | --- | --- |
| ROC-AUC / F1 | Video and window authenticity classification | Higher |
| EER | Authenticity classification | Lower |
| Window proposal AP | Temporal manipulation localization | Higher |
| MSSC | Multi-scale audiovisual coherence | Higher agreement |
| LSE-C | Lip-sync confidence over candidate offsets | Higher |
| LSE-D | Minimum audiovisual embedding distance | Lower |
| Offset MAE | Audio-video delay estimation | Lower |

Calibrate the threshold on real validation videos, then evaluate the test split:

```bash
python -B main.py calibrate --manifest ../runtime/mwsc/split.csv --checkpoint ../runtime/mwsc/experiments/seed42/encoders.pt --output ../runtime/mwsc/experiments/seed42/calibration.json
python -B main.py evaluate --manifest ../runtime/mwsc/split.csv --checkpoint ../runtime/mwsc/experiments/seed42/encoders.pt --config ../runtime/mwsc/experiments/seed42/calibration.json --split test --output ../runtime/mwsc/experiments/seed42/test.json
```

Repeat calibration and evaluation for each seed. Compare mean and standard deviation across the four test runs. Calibration uses validation data only; it is bound to the checkpoint hash.

### Audio-Video Offset Tests

For manually verified synchronized real videos:

```bash
python -B main.py sync --manifest ../runtime/mwsc/verified.csv --checkpoint ../runtime/mwsc/experiments/seed42/encoders.pt --offsets -0.3 -0.2 -0.1 0 0.1 0.2 0.3 --output ../runtime/mwsc/offsets.json
```

The controlled-offset interface crops unsupported edges and preserves authenticity labels. It reports offset error and synchronization classification separately from deepfake classification.

### Default Scoring Parameters

| Parameter | MWSC | SyncNet |
| --- | --- | --- |
| Video sampling | 30 fps, grayscale | 25 fps, BGR face crops |
| Audio | 16 kHz mono, log-STFT | 16 kHz mono, 13-coefficient MFCC |
| Window | 2 seconds, 50% overlap | 5 seconds by default |
| Representation | Dense features, per-modality PCA | Pretrained 1024-dimensional embeddings |
| Comparison | Morlet coherence at 0.5–4 Hz | Euclidean distance over ±15 frames |

MSSC uses the raw Gaussian weights by default. Their sum is approximately 1.121, so MSSC can exceed 1. Normalized Gaussian weights are available through the configuration. Thresholds must be recalibrated when the scoring grid or weight mode changes.

---

## Tests

```bash
python -B main.py test
python -B main.py demo
```

The tests cover signal equations, dataset splits, training/inference, synchronization offsets, pretrained model compatibility, and distributed data sharding.

SyncNet results on the official 12.68-second example, using CPU inference:

| Preprocessing | LSE-C | LSE-D | Audio lag |
| --- | --- | --- | --- |
| Precropped, 5-second windows | 10.0991 | 5.4506 | -120 ms |
| Precropped, one full-video window | 10.0479 | 5.3519 | -120 ms |
| Automatic Haar crop, first 3 seconds | 7.4379 | 6.2309 | -120 ms |

Cropping and windowing define the scoring protocol. The face pipeline uses Haar detection, a supplied face box, or a precropped track. Each report records its preprocessing mode and model hash.

---

## References

- [Method description: CN122413017A](https://eureka.patsnap.com/patent/CN122413017A)
- [SyncNet: Out of time: automated lip sync in the wild](https://github.com/joonson/syncnet_python)
- [Wav2Lip evaluation](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation)
- [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)
- [LAV-DF](https://github.com/ControlNet/LAV-DF)
- [AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M)
- [AVGen-Bench](https://github.com/microsoft/AVGen-Bench)
- [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
- [LRS3: a large-scale dataset for visual speech recognition](https://arxiv.org/abs/1809.00496)
- [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
- [DFDC](https://ai.meta.com/datasets/dfdc/)

## Third-party License

The SyncNet-compatible network definitions retain Joon Son Chung's MIT license. Pretrained weights follow the terms on the [Oxford download page](https://www.robots.ox.ac.uk/~vgg/software/lipsync/).

```text
Copyright (c) 2016-present Joon Son Chung.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
