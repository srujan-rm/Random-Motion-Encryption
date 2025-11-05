import cv2
import numpy as np
import pandas as pd
import time
from skimage.feature import local_binary_pattern

# ==============================================
# === COLOR CONFIGURATION (HSV + LAB fusion) ===
# ==============================================

BLOB_BGR = np.array([
    # Blue / Cyan
    [17, 55, 246], [86, 116, 250], [5, 5, 166],
    # Magenta / Pink
    [255, 100, 251], [252, 14, 241], [120, 2, 122],
    [247, 1, 25], [230, 2, 47], [155, 1, 23],
    # Orange / Yellow
    [255, 213, 51], [132, 32, 5], [255, 219, 56], [253, 145, 13],
    # Green
    [71, 255, 25], [36, 178, 28], [16, 192, 17], [17, 121, 13]
], dtype=np.uint8)

SEGMENT_BGR = np.array([
    [25, 24, 59], [35, 7, 37], [97, 64, 6],
    [5, 22, 5], [59, 12, 25], [19, 26, 20]
], dtype=np.uint8)

HSV_TOLERANCE_DEFAULT = (15, 90, 90)
HSV_TOLERANCE_PINK = (25, 120, 120)
HSV_TOLERANCE_BG = (15, 80, 80)

# ==============================================
# === COLOR RANGE HELPERS ======================
# ==============================================

def bgr_to_hsv(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]

def bgr_to_lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0]

def build_hsv_ranges(bgr_list):
    ranges = []
    for bgr in bgr_list:
        h, s, v = map(int, bgr_to_hsv(bgr))
        th, ts, tv = HSV_TOLERANCE_PINK if (h < 15 or h > 165) else HSV_TOLERANCE_DEFAULT
        low = np.array([max(h - th, 0), max(s - ts, 0), max(v - tv, 0)], np.uint8)
        high = np.array([min(h + th, 179), min(s + ts, 255), min(v + tv, 255)], np.uint8)
        ranges.append((low, high))
    return ranges

def build_lab_ranges(bgr_list, tol_L=25, tol_A=25, tol_B=25):
    ranges = []
    for bgr in bgr_list:
        L, A, B = map(int, bgr_to_lab(bgr))
        low = np.array([max(L - tol_L, 0), max(A - tol_A, 0), max(B - tol_B, 0)], np.uint8)
        high = np.array([min(L + tol_L, 255), min(A + tol_A, 255), min(B + tol_B, 255)], np.uint8)
        ranges.append((low, high))
    return ranges

BLOB_HSV_RANGES = build_hsv_ranges(BLOB_BGR)
BLOB_LAB_RANGES = build_lab_ranges(BLOB_BGR)
SEGMENT_HSV_RANGES = build_hsv_ranges(SEGMENT_BGR)

# ==============================================
# === SEGMENTATION =============================
# ==============================================

def segment_mask_by_ranges(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mask_hsv = np.zeros(frame.shape[:2], np.uint8)
    for low, high in BLOB_HSV_RANGES:
        mask_hsv |= cv2.inRange(hsv, low, high)
    mask_lab = np.zeros(frame.shape[:2], np.uint8)
    for low, high in BLOB_LAB_RANGES:
        mask_lab |= cv2.inRange(lab, low, high)
    mask_bg = np.zeros(frame.shape[:2], np.uint8)
    for low, high in SEGMENT_HSV_RANGES:
        mask_bg |= cv2.inRange(hsv, low, high)
    mask = cv2.bitwise_and(cv2.bitwise_or(mask_hsv, mask_lab), cv2.bitwise_not(mask_bg))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask

# ==============================================
# === FEATURE EXTRACTION =======================
# ==============================================

def mean_hsv_drift(prev, curr):
    hsv1 = cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    return np.linalg.norm(np.array(cv2.mean(hsv2)[:3]) - np.array(cv2.mean(hsv1)[:3]))

def blob_centroid_speed(prev_mask, curr_mask):
    def centroid(mask):
        M = cv2.moments(mask)
        return (M["m10"]/M["m00"], M["m01"]/M["m00"]) if M["m00"] else (0, 0)
    c1, c2 = centroid(prev_mask), centroid(curr_mask)
    return np.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)

def optical_flow_entropy(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 3, 5, 1.3, 0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return np.mean(mag) * np.std(mag), flow, mag

def frame_shannon_entropy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = hist / (np.sum(hist) + 1e-9)
    return -np.sum(p * np.log2(p + 1e-9))

def lbp_texture_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257))
    p = hist / (np.sum(hist) + 1e-9)
    return np.std(p)

# ==============================================
# === VISUALIZATION HELPERS ====================
# ==============================================

def draw_optical_flow(flow, mask, frame, step=10):
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    vis = frame.copy()
    avg_mag = np.mean(np.sqrt(fx**2 + fy**2))
    scale = 1.8 if avg_mag < 1 else 1.2
    for (xi, yi, dxi, dyi) in zip(x, y, fx, fy):
        if mask[yi, xi] == 0:
            continue
        x2, y2 = int(xi + dxi * scale), int(yi + dyi * scale)
        cv2.arrowedLine(vis, (xi, yi), (x2, y2), (255, 255, 255), 1, tipLength=0.35)
    return vis

def visualize_heatmap_color_blobs(mag, frame, mask):
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = np.zeros_like(frame)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 40:
            continue
        roi_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        mean_color = np.array(cv2.mean(frame[y:y+h, x:x+w], mask=roi_mask)[:3], np.uint8)
        local_mag = mag_norm[y:y+h, x:x+w]
        boosted = cv2.pow(local_mag.astype(np.float32)/255.0, 0.7)*200
        boosted = np.clip(boosted, 0, 255).astype(np.uint8)
        color_tint = cv2.multiply(np.tile(mean_color, (h, w, 1)), cv2.merge([boosted]*3), scale=1/255.0)
        color_tint = np.clip(color_tint, 0, 255).astype(np.uint8)
        existing = heat[y:y+h, x:x+w]
        heat[y:y+h, x:x+w] = cv2.addWeighted(existing, 0.6, color_tint, 0.4, 0)
    return cv2.addWeighted(frame, 0.7, heat, 0.6, 0)

# ==============================================
# === MAIN PROCESS LOOP ========================
# ==============================================

def load_frames_from_video(path, max_frames=120, step=3, resize_scale=0.3, visualize=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    frames, entropy_scores = [], []
    prev_frame = prev_mask = prev_gray = None
    frame_idx = 0
    start_time = time.time()

    print("\n🎥 Enhanced Lava Lamp capture with flow + entropy metrics...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, (int(frame.shape[1]*resize_scale), int(frame.shape[0]*resize_scale)))
        mask = segment_mask_by_ranges(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            drift = mean_hsv_drift(prev_frame, frame)
            centroid_speed = blob_centroid_speed(prev_mask, mask)
            flow_entropy, flow, mag = optical_flow_entropy(prev_gray, gray)
            tex_entropy = frame_shannon_entropy(frame)
            lbp_var = lbp_texture_variance(frame)

            entropy_total = (0.3*drift + 0.25*centroid_speed + 0.25*flow_entropy +
                             0.1*tex_entropy + 0.1*lbp_var)
            entropy_scores.append(entropy_total)

            vis = draw_optical_flow(flow, mask, frame)
            vis = visualize_heatmap_color_blobs(mag, vis, mask)

            # Overlay metrics
            cv2.putText(vis, f"Frame {frame_idx}/{max_frames}", (15,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(vis, f"HSV Drift: {drift:.3f}", (15,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(vis, f"Centroid: {centroid_speed:.3f}", (15,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(vis, f"Flow: {flow_entropy:.3f}", (15,90),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(vis, f"Shannon: {tex_entropy:.3f}", (15,110),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(vis, f"LBP: {lbp_var:.6f}", (15,130),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(vis, f"Combined Entropy: {entropy_total:.3f}", (15,155),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            if visualize:
                cv2.imshow("Lava Lamp - Flow + Entropy Visualization", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        prev_frame, prev_mask, prev_gray = frame, mask, gray
        frames.append(frame)
        frame_idx += 1
        if len(frames) >= max_frames: break

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    pd.DataFrame({"Frame": range(len(entropy_scores)), "EntropyScore": entropy_scores}).to_csv("results/entropy_features.csv", index=False)
    print(f"\n✅ {len(frames)} frames processed. Entropy data saved.\n")
    return frames
