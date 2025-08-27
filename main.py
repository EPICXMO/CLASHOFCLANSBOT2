import time, subprocess, numpy as np, cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- ADB helpers ---
def adb(*args):
    return subprocess.run(["adb", *args], capture_output=True)

def get_frame_bgr():
    out = adb("exec-out", "screencap", "-p").stdout
    if not out:
        raise RuntimeError("No frame from ADB. Is BlueStacks ADB enabled and connected?")
    arr = np.frombuffer(out, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screencap PNG")
    return img

def tap(x, y):
    adb("shell", "input", "tap", str(int(x)), str(int(y)))

# --- Perception ---
yolo = YOLO("yolo11n.pt")  # downloads on first run
ocr = PaddleOCR(lang="en")

def find_and_tap_text(frame_bgr, keywords=("train", "train troops", "ok", "continue")):
    res = ocr.ocr(frame_bgr, cls=True)
    for block in res:
        for line in block:
            box, (txt, conf) = line
            if conf is None:
                continue
            t = (txt or "").strip().lower()
            if any(k in t for k in keywords):
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                cx, cy = sum(xs)/4.0, sum(ys)/4.0
                tap(cx, cy)
                return True
    return False

def click_first_detection(frame_bgr, min_conf=0.5):
    r = yolo.predict(source=frame_bgr, imgsz=640, conf=min_conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return False
    idx = int(r.boxes.conf.argmax())
    x1, y1, x2, y2 = map(float, r.boxes.xyxy[idx].cpu().numpy())
    tap((x1 + x2) / 2, (y1 + y2) / 2)
    return True

def skill_open_train_troops():
    frame = get_frame_bgr()
    if find_and_tap_text(frame, ("train troops", "train")):
        time.sleep(0.6)
        frame2 = get_frame_bgr()
        _ = find_and_tap_text(frame2, ())  # parse only; no extra tap
        return True
    if click_first_detection(frame, min_conf=0.6):
        time.sleep(0.6)
        frame2 = get_frame_bgr()
        if find_and_tap_text(frame2, ("train troops", "train")):
            return True
    return False

def skill_clear_simple_popups():
    frame = get_frame_bgr()
    return find_and_tap_text(frame, ("ok", "continue", "confirm", "close"))

def main_loop():
    print("Starting v0 loop. CTRL+C to stop.")
    while True:
        if skill_clear_simple_popups():
            time.sleep(0.5); continue
        if skill_open_train_troops():
            time.sleep(1.0); continue
        time.sleep(0.3)

if __name__ == "__main__":
    main_loop()
