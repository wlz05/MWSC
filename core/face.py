# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from pathlib import Path
import importlib.util
import numpy as np
from support.storage import relative_path

def color_frames(path, start, duration, origin=None):
    import cv2
    path = relative_path(path, True)
    if importlib.util.find_spec("av") is not None:
        import av
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    raise ValueError("Video has no color stream")
                stream = container.streams.video[0]
                native_origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
                begin = native_origin + start
                if start:
                    container.seek(int(begin / stream.time_base), stream=stream, backward=True)
                for frame in container.decode(video=0):
                    if frame.pts is None:
                        raise ValueError("Color video frame lacks a presentation timestamp")
                    timestamp = float(frame.pts * frame.time_base)
                    if timestamp >= begin + duration - 1e-8:
                        break
                    if timestamp >= begin - 1e-8:
                        yield frame.to_ndarray(format="bgr24"), timestamp if origin is None else timestamp - native_origin + origin
        except av.FFmpegError as error:
            raise ValueError("PyAV could not decode the color stream") from error
        return
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Color video has no valid frame rate")
        first = int(np.ceil(start * fps - 1e-8))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        while index / fps < start + duration - 1e-8:
            ok, frame = capture.read()
            if not ok:
                break
            if index / fps >= start - 1e-8:
                yield frame, (origin or 0.0) + index / fps
            index += 1
    finally:
        capture.release()

def validate_box(box):
    values = np.asarray(box)
    if values.shape != (4,) or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("Face box requires four finite coordinates")
    left, top, right, bottom = map(float, values)
    if min(left, top) < 0 or right <= left or bottom <= top:
        raise ValueError("Face box requires left, top, right, bottom coordinates")
    return left, top, right, bottom

def crop_face(frame, box):
    import cv2
    left, top, right, bottom = validate_box(box)
    if right > frame.shape[1] or bottom > frame.shape[0]:
        raise ValueError("Face box exceeds the video frame")
    size = max(right - left, bottom - top) / 2
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    border = int(np.ceil(1.8 * size)) + 2
    padded = np.pad(frame, ((border, border), (border, border), (0, 0)), constant_values=110)
    x, y = center_x + border, center_y + border
    face = padded[int(y - size):int(y + 1.8 * size), int(x - 1.4 * size):int(x + 1.4 * size)]
    if min(face.shape[:2]) < 1:
        raise ValueError("Face crop is empty")
    return cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)

class FaceReader:
    def __init__(self, box=None, cropped=False):
        import cv2
        if box is not None and cropped:
            raise ValueError("Face box and precropped mode are mutually exclusive")
        self.box = validate_box(box) if box is not None else None
        self.cropped = bool(cropped)
        self.mode = "precropped" if cropped else "manual_box" if box is not None else "haar_single_face"
        self.detector = None
        if self.mode == "haar_single_face":
            resource = Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
            factory = getattr(cv2, "CascadeClassifier", None) or getattr(getattr(cv2, "xobjdetect", None), "CascadeClassifier", None) or getattr(cv2, "xobjdetect_CascadeClassifier", None)
            if factory is None:
                raise RuntimeError("Automatic faces require OpenCV 4.x or OpenCV 5 contrib; run install.py --install --lipsync or provide a face box")
            self.detector = factory(str(resource))
            if self.detector.empty():
                raise RuntimeError("OpenCV frontal face resource is unavailable; provide a face box")

    def read(self, path, start, duration, origin=None):
        import cv2
        crops, times, validity = [], [], []
        previous = None
        for frame, timestamp in color_frames(path, start, duration, origin):
            if len(crops) >= 6000:
                raise ValueError("Color decode exceeds the bounded frame count")
            if not np.isfinite(timestamp) or times and timestamp <= times[-1]:
                raise ValueError("Color timestamps must be finite and increasing")
            box = self.box
            valid = True
            if self.cropped:
                crop = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                if box is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    factor = min(1.0, 640 / max(gray.shape))
                    small = cv2.resize(gray, None, fx=factor, fy=factor)
                    found = self.detector.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    valid = len(found) == 1
                    if valid:
                        x, y, width, height = found[0] / factor
                        box = (x, y, min(x + width, frame.shape[1]), min(y + height, frame.shape[0]))
                        if previous is not None:
                            overlap = max(0, min(box[2], previous[2]) - max(box[0], previous[0])) * max(0, min(box[3], previous[3]) - max(box[1], previous[1]))
                            area = (box[2] - box[0]) * (box[3] - box[1]) + (previous[2] - previous[0]) * (previous[3] - previous[1]) - overlap
                            valid = overlap / max(area, 1e-12) >= 0.3
                        previous = box
                    else:
                        previous = None
                crop = crop_face(frame, box) if valid else np.zeros((224, 224, 3), dtype=np.uint8)
            crops.append(crop)
            times.append(timestamp)
            validity.append(valid)
        if len(times) < 2:
            raise ValueError("Too few color frames for lip synchronization")
        return np.asarray(crops), np.asarray(times), np.asarray(validity, dtype=bool)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
