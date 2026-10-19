# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from collections import deque
import math
import numpy as np
from support.storage import atomic_json, read_json

class DynamicThreshold:
    def __init__(self, config, stream_id="default", model_id="feature_interface"):
        self.config = config.validate()
        self.stream_id = str(stream_id)
        self.model_id = str(model_id)
        self.threshold = float(config.threshold)
        self.history = deque(maxlen=config.history_size)
        self.accepted_count = 0
        self.decision_count = 0

    def decide(self, score, update=True):
        score = float(score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("Synchronization score must be finite and nonnegative")
        before = self.threshold
        fake = score < before
        next_history = None
        if update and not fake:
            # Only accepted real scores enter the adaptive history.
            next_history = [*self.history, score][-self.config.history_size:]
            values = np.asarray(next_history, dtype=np.float64)
            scale = max(float(values.max()), 1.0)
            mean = float(np.mean(values / scale)) * scale
            std = float(np.std(values / scale, ddof=0)) * scale
            target = mean - self.config.tolerance_k * std
            after = self.config.ema_alpha * before + (1 - self.config.ema_alpha) * target
            if not math.isfinite(after):
                raise ValueError("Adaptive threshold update is nonfinite")
        if update:
            self.decision_count += 1
            if next_history is not None:
                self.history = deque(next_history, maxlen=self.config.history_size)
                self.accepted_count += 1
                self.threshold = after
        return {
            "decision": "fake" if fake else "real",
            "predicted_label": int(fake),
            "threshold_before": before,
            "threshold_after": self.threshold,
            "history_length": len(self.history),
            "updated": bool(update and not fake),
            "stream_id": self.stream_id,
        }

    def state_dict(self):
        return {
            "version": 1,
            "config_signature": self.config.signature(),
            "stream_id": self.stream_id,
            "threshold": self.threshold,
            "model_id": self.model_id,
            "history": list(self.history),
            "accepted_count": self.accepted_count,
            "decision_count": self.decision_count,
        }

    def save(self, path):
        atomic_json(path, self.state_dict())

    def load(self, path):
        state = read_json(path)
        if not isinstance(state, dict) or type(state.get("version")) is not int or state.get("version") != 1 or state.get("config_signature") != self.config.signature():
            raise ValueError("Threshold state configuration mismatch")
        if state.get("stream_id") != self.stream_id:
            raise ValueError("Threshold state belongs to a different stream")
        if state.get("model_id") != self.model_id:
            raise ValueError("Threshold state belongs to different encoder weights")
        history = state.get("history")
        threshold = state.get("threshold")
        if not isinstance(history, list) or len(history) > self.config.history_size:
            raise ValueError("Invalid threshold state")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("Invalid threshold value")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0 for item in history):
            raise ValueError("Invalid historical synchronization score")
        accepted = state.get("accepted_count")
        decisions = state.get("decision_count")
        if type(accepted) is not int or type(decisions) is not int or accepted < len(history) or decisions < accepted:
            raise ValueError("Invalid state counters")
        # Commit only after every state field is valid.
        self.threshold = float(threshold)
        self.history = deque(map(float, history), maxlen=self.config.history_size)
        self.accepted_count = accepted
        self.decision_count = decisions
        return self

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
