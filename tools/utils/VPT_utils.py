import cv2, numpy as np

def load_video_frames(path, fps=20, resize=(128,128)):
    cap = cv2.VideoCapture(path)
    frames, orig_fps = [], cap.get(cv2.CAP_PROP_FPS)
    sample_interval = int(round(orig_fps / fps)) if orig_fps > 0 else 1
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
        idx += 1
    cap.release()
    return np.stack(frames, axis=0)