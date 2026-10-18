import os, subprocess, cv2
def normalise(src, dst):
    """Re-encode once to a local, OpenCV-safe h264 file. Returns dst."""
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        # 18 Oct (worker, N0): encode to a side file and rename at the end, so a failed or killed ffmpeg never leaves a
        # half-written dst that the next call (and modal_app's "if not os.path.exists(video)") would take as done
        tmp = os.path.splitext(dst)[0] + ".part.mp4"
        try: subprocess.run(["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", tmp], check=True, capture_output=True)
        except BaseException:
            if os.path.exists(tmp): os.remove(tmp)
            raise
        os.replace(tmp, dst)
    return dst
def info(path):
    cap = cv2.VideoCapture(path); n = int(cap.get(7)); fps = cap.get(5); w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
    return {"n": n, "fps": fps, "width": w, "height": h}
def frames(path):
    cap = cv2.VideoCapture(path); k = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        yield k, f; k += 1
    cap.release()
def frame_at(path, k):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); cap.release(); return f if ok else None
