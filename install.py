# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import argparse
import importlib.util
import subprocess
import sys

REQUIRED = {"numpy": "numpy>=1.26", "scipy": "scipy>=1.11", "torch": "torch>=2.2", "cv2": "opencv-python>=4.8"}
OPTIONAL = {"av": "av>=12"}

def main():
    parser = argparse.ArgumentParser(description="Inspect or install dependencies; choose the PyTorch GPU build separately")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--without-av", action="store_true")
    parser.add_argument("--lipsync", action="store_true", help="Ensure automatic face detection is available for SyncNet")
    args = parser.parse_args()
    packages = {**REQUIRED, **({} if args.without_av else OPTIONAL)}
    if args.lipsync:
        packages["cv2"] = "opencv-python>=4.8,<5"
    missing = [requirement for name, requirement in packages.items() if importlib.util.find_spec(name) is None]
    if args.lipsync and importlib.util.find_spec("cv2") is not None:
        import cv2
        available = getattr(cv2, "CascadeClassifier", None) or getattr(getattr(cv2, "xobjdetect", None), "CascadeClassifier", None) or getattr(cv2, "xobjdetect_CascadeClassifier", None)
        if available is None:
            missing.append("opencv-python>=4.8,<5")
            print("Automatic faces: install a compatible OpenCV 4.x build")
    for name, requirement in packages.items():
        print(name + ": " + ("available" if importlib.util.find_spec(name) else "missing") + " (" + requirement + ")")
    if args.install and missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    elif missing:
        print("Run this file with --install to install missing packages")
    print("GPU installation reference: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
