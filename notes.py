# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

PROJECT = {"name": "MWSC", "title": "MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection", "directory": "code/remain/MWSC", "entry": "main.py"}
PATENT = {
    "publication": "CN122413017A",
    "application": "CN202610565811.9",
    "applicant": "Sichuan University",
    "publication_date": "2026-07-17",
    "public_text_url": "https://eureka.patsnap.com/patent/CN122413017A",
    "verification_date": "2026-10-09",
    "local_document_reference": "User-supplied Word patent document",
    "local_document_sha256": "3177244b72678053e5b828fbf674bfa18fe94a35c9fe151ea51b351754ff8a06",
    "status": "Independent engineering reconstruction; not the inventors' original source code",
    "comparison": [
        "The public patent text matches the supplied Word document's core claims and preferred parameters.",
        "Public paragraph 0050 adds an explanation of DTW interpolation and equal sequence lengths.",
        "Public web extraction omits mathematical images; the Word equations determine the implementation.",
        "The public record was inspected through PatSnap; no official CNIPA PDF checksum comparison was possible.",
    ],
}
FORMULAS = {
    "coherence": "C(s) = abs(sum_t(Wv(t,s) * conjugate(Wa(t,s))))**2 / (sum_t(abs(Wv(t,s))**2) * sum_t(abs(Wa(t,s))**2))",
    "gaussian": "w(s) = exp(-(s-mu)**2/(2*sigma**2)) / (sigma*sqrt(2*pi))",
    "fusion": "MSSC = sum_s(w(s) * C(s))",
    "decision": "score < current_threshold means fake; equality means real",
    "update": "theta_next = alpha*theta_current + (1-alpha)*(mean(history)-k*std(history))",
}
PATENT_MAPPING = {
    "claim_2": ["core/media.py", "core/audio.py", "core/align.py"],
    "claim_3": ["model/encoder.py", "model/vision.py", "model/extract.py"],
    "claim_4": ["core/audio.py", "model/sound.py", "model/extract.py"],
    "claims_5_6_7": ["core/wavelet.py"],
    "claims_8_9": ["core/threshold.py"],
    "system_claim_10": ["engine/detect.py", "engine/stream.py"],
}
IMPLEMENTATION_CHOICES = (
    "The default grayscale image size is 224, sample rate is 16000, video cadence is 30, and segment length is 2 seconds.",
    "Audio uses pre-emphasis 0.97 and STFT frames of 25 ms with 10 ms hops.",
    "Segments overlap by 50 percent; timestamp DTW is restricted to 50 ms and runs within each segment.",
    "DTW changes timestamps only; it does not match lip content to audio content and erase forgery evidence.",
    "PCA is fit independently to the video and audio feature matrices in each segment.",
    "The temporal dimension survives spatial global average pooling.",
    "Morlet omega=6, eight linearly spaced frequencies, reflective boundary padding, and FFT size=512 are engineering choices.",
    "Physical frequencies between 0.5 and 4 Hz are mapped to Morlet scales in sample units.",
    "Gaussian parameters default to mean=4 Hz and standard deviation=1.5 Hz.",
    "The patent does not specify the discrete frequency grid or explicitly normalize Gaussian weights.",
    "patent_raw preserves the printed Gaussian formula; the maximum score is the sum of its discrete weights.",
    "normalized is an explicit engineering alternative; changing the grid or weight mode requires recalibration.",
    "The default threshold is 0.75, history holds 100 accepted scores, alpha=0.9, and k=2.",
    "The historical standard deviation uses population normalization, an unspecified implementation choice.",
    "Only accepted real segments update history, and the updated threshold is used for the next decision.",
    "The literal threshold formula is not clipped; inspect adaptive drift during deployment.",
    "Compact CNN encoders are an allowed equivalent network route; they require local pretraining.",
    "Dense I3D and VGGish convolutional encoders are supplied as an optional preferred-network route.",
    "Dense temporal strides and frequency-only VGGish pooling differ from ordinary clip classifiers.",
    "I3D input repeats grayscale into three channels; dense VGGish uses log-mel features derived from the STFT.",
    "Projection layers and altered timing require audio-visual fine-tuning even after importing backbone weights.",
    "Backbone import requires matching tensor keys and dimensions; partial silent loading is forbidden.",
    "Training on real pairs with temporal contrastive loss is an engineering extension; the patent omits training details.",
    "Validation-only quantile calibration and any-fake-window video aggregation are engineering extensions.",
    "Source grouping is automatic; true subject-disjoint grouping requires identity metadata or an identity map.",
    "Streaming requires contiguous mono PCM packets and presentation timestamps; missing packets yield errors or insufficient evidence.",
)
LIMITATIONS = (
    "No original trained weights or the patent's self-built dataset were released with the supplied document.",
    "The reported 94.7 percent patent AUC is not reproduced or claimed by this implementation.",
    "Random encoder weights are never accepted by the inference checkpoint loader.",
    "A trained checkpoint still needs held-out evaluation; successful execution is not evidence of detection accuracy.",
    "High synchronization does not prove authenticity; synchronized face swaps or coordinated fakes may evade this method.",
    "The global coherence equation can remain high under a constant phase offset; it is not a general lip-delay estimator.",
    "A 2-second window contains only one 0.5 Hz cycle and has substantial wavelet boundary effects.",
    "Silence, static features, large timestamp gaps, and insufficient samples return insufficient evidence.",
    "Ordinary patch-level VGGish embeddings have insufficient cadence for 4 Hz analysis; dense features are required.",
    "OpenCV decoding assumes constant frame rate; PyAV is preferred for timestamps and container audio.",
    "The window localization metrics are a research interface, not the official LAV-DF or AV-Deepfake1M leaderboard protocol.",
)
WORKFLOW = (
    "Run commands from code/remain/MWSC and use relative input/output paths.",
    "python -B install.py",
    "python -B install.py --install",
    "python -B main.py doctor",
    "python -B main.py notes",
    "python -B main.py datasets",
    "python -B main.py import --dataset fakeavceleb --root data/fakeavceleb --output manifests/fakeavceleb.csv",
    "python -B main.py import --dataset lavdf --root data/lavdf --metadata data/lavdf/metadata.min.json --output manifests/lavdf.csv",
    "python -B main.py import --dataset avdeepfake1m --root data/avdeepfake1m --metadata data/avdeepfake1m/train_metadata.json data/avdeepfake1m/val_metadata.json --output manifests/avdeepfake1m.csv",
    "python -B main.py import --dataset dfdc --root data/dfdc --metadata data/dfdc/metadata.json --output manifests/dfdc.csv",
    "python -B main.py import --dataset lrs2 --root data/lrs2/main --filelist data/lrs2/train.txt --subset train --output manifests/lrs2_train.csv",
    "python -B main.py import --dataset lrs3 --root data/lrs3/trainval --subset train --output manifests/lrs3_train.csv",
    "python -B main.py import --dataset voxceleb2 --root data/voxceleb2 --output manifests/voxceleb2.csv",
    "Use custom CSV imports to merge speech subsets and provide independent sync_label and offset_seconds annotations.",
    "python -B main.py split --manifest manifests/fakeavceleb.csv --output manifests/fakeavceleb_split.csv --seed 42",
    "python -B main.py split --manifest manifests/lavdf.csv --keep-official --output manifests/lavdf_official.csv",
    "python -B main.py audit --manifest manifests/fakeavceleb_split.csv --check-files",
    "python -B main.py audit --manifest manifests/fakeavceleb_split.csv --decode --limit 20",
    "python -B main.py train --manifest manifests/fakeavceleb_split.csv --checkpoint weights/encoders.pt --epochs 20 --batch-size 2 --device auto",
    "python -B main.py calibrate --manifest manifests/fakeavceleb_split.csv --checkpoint weights/encoders.pt --output runs/calibration.json",
    "python -B main.py evaluate --manifest manifests/fakeavceleb_split.csv --checkpoint weights/encoders.pt --config runs/calibration.json --split test --output runs/test.json",
    "python -B main.py sync --manifest manifests/speech_split.csv --checkpoint weights/encoders.pt --offsets -0.3 0 0.3 --split test --output runs/sync.json",
    "Controlled shifts require sync_label=0 on an explicitly verified baseline; unknown labels are never guessed.",
    "python -B main.py sync --manifest manifests/annotated_sync.csv --checkpoint weights/encoders.pt --split test --output runs/offset.json",
    "python -B main.py predict --video data/example.mp4 --checkpoint weights/encoders.pt --config runs/calibration.json --output runs/prediction.json",
    "python -B main.py predict --video data/example.mp4 --audio data/example.wav --checkpoint weights/encoders.pt --no-update",
    "python -B main.py extract --video data/example.mp4 --checkpoint weights/encoders.pt --duration 2 --output runs/features.npz",
    "python -B main.py score --features runs/features.npz",
    "python -B main.py export --checkpoint weights/encoders.pt --video-output weights/video.ts --audio-output weights/audio.ts",
    "python -B main.py demo",
    "python -B main.py test",
    "python -B check.py",
    "MWSC.md is the requested documentation exception; all delivered source files are Python.",
    "Training and explicit output commands create runtime data later; run format checks on clean source trees.",
)
HARDWARE = (
    "CPU inference is supported for experiments and tests; throughput depends on encoder architecture.",
    "Compact CNN training at 224 pixels can start on an NVIDIA GPU with 12 GB VRAM and batch size 1 or 2.",
    "Dense I3D training can need 24 GB or more VRAM; start at batch size 1 and measure actual peak memory.",
    "These memory figures are planning estimates, not measurements on this host.",
    "Use at least 16 GB host RAM for compact experiments and 32 GB or more for large media collections.",
    "Disk requirements are dominated by the selected dataset; measure the authorized download before allocation.",
    "The source code does not download datasets, weights, or GPU software automatically.",
)
INTERFACES = {"deepfake_datasets": "dataset/adapter.py", "speech_datasets": "dataset/speech.py", "manifest_schema": "dataset/manifest.py", "split_and_leakage_audit": "dataset/split.py", "synchronization_offsets": "engine/sync.py", "offline_detection": "engine/detect.py", "packet_streaming": "engine/stream.py", "evaluation": "engine/evaluate.py", "format_rules": "check.py"}
READINESS = {"implemented": ["Dense feature encoders and local pretraining", "Patent PCA/Morlet/coherence/EMA pipeline", "Bounded-window long-video inference", "Deepfake and speech dataset imports", "Media availability and source leakage audits", "Separate synchronization labels and controlled content shifts", "Offset-error and classification metrics"], "external_requirements": ["Authorized real dataset files", "Trained encoder weights", "PyAV or ffmpeg/ffprobe for embedded container audio", "Held-out calibration and real-data benchmark results"], "research_limits": ["Offset correlation is an additional diagnostic and not the patent coherence equation", "Official temporal-localization leaderboard evaluation requires its own prescribed protocol", "The patent's original weights, dataset, and reported AUC are unavailable for exact reproduction"]}

def report():
    return {"project": PROJECT, "patent": PATENT, "formulas": FORMULAS, "claim_mapping": PATENT_MAPPING, "implementation_choices": IMPLEMENTATION_CHOICES, "limitations": LIMITATIONS, "workflow": WORKFLOW, "hardware": HARDWARE, "interfaces": INTERFACES, "readiness": READINESS}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
