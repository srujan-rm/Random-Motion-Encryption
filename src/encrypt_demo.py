import argparse
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from capture import load_frames_from_video
from entropy import aggregate_entropy_bits, sha256_digest_verbose
from kdf_aes import derive_key_iv, encrypt_bytes
from utils import (
    image_to_bytes,
    shuffle_pixels,
    visualize_cipher_bytes_as_image
)
from evaluate import npcr_uaci


def resolve_path(path):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, path)


def encrypt_image_with_motion(video_path, target_path, out_cipher, out_encimg, fast=False):
    print("\n🚀 [START] FULL TRANSPARENT RANDOM MOTION ENCRYPTION")
    print("=====================================================\n")

    for f in [video_path, target_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"❌ Missing required file: {f}")

    results_dir = os.path.dirname(out_cipher)
    os.makedirs(results_dir, exist_ok=True)

    # === STEP 1: Frame Capture + Entropy Extraction ===
    print("🎥 [STEP 1] Capturing frames and extracting motion entropy features...\n")
    frames = load_frames_from_video(video_path, max_frames=120, resize_scale=0.3, visualize=not fast)

    # === STEP 2: Read computed entropy file ===
    print("🧮 [STEP 2] Loading entropy values from CSV...")
    entropy_vals_full = np.loadtxt("results/entropy_features.csv", delimiter=",", skiprows=1, usecols=1)
    print(f"   ➤ Total frames processed: {len(entropy_vals_full)}")

    # skip warm-up
    entropy_vals = entropy_vals_full[25:] if len(entropy_vals_full) > 25 else entropy_vals_full
    print(f"   ➤ Stable entropy region frames: {len(entropy_vals)}")
    print(f"   ➤ Range: {entropy_vals.min():.4f} – {entropy_vals.max():.4f}")
    print(f"   ➤ Mean: {entropy_vals.mean():.4f}\n")

    # === STEP 3: Plot entropy variation ===
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(25, 25 + len(entropy_vals)), entropy_vals, color='orange', linewidth=2, label='Entropy')
    plt.fill_between(np.arange(25, 25 + len(entropy_vals)), entropy_vals, color='orange', alpha=0.3)
    plt.title("Entropy Variation Across Frames (Stable Region)")
    plt.xlabel("Frame Index")
    plt.ylabel("Combined Entropy Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "entropy_vs_frame.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"✅ Entropy evolution plot saved → {plot_path}\n")

    # === STEP 4: Aggregate entropy ===
    seed_material = aggregate_entropy_bits(entropy_vals)

    # === STEP 5: SHA-256 Conditioning ===
    seed = sha256_digest_verbose(seed_material)

    # === STEP 6: Derive AES key + IV ===
    print("============================================================")
    print("🔐 [STEP 6] HKDF KEY DERIVATION (AES-256-GCM)")
    print("============================================================")
    key, iv = derive_key_iv(seed)
    print(f"   ▪ Derived AES-256 key (hex): {key.hex()}")
    print(f"   ▪ Derived IV (hex): {iv.hex()}")
    print("============================================================\n")

    # === STEP 7: Encryption ===
    print("🖼️ [STEP 7] Encrypting target image using motion-derived key...")
    img = cv2.imread(target_path)
    shuffled, idx = shuffle_pixels(img, seed)
    cipher = encrypt_bytes(image_to_bytes(shuffled), key, iv)
    cipher_img = visualize_cipher_bytes_as_image(cipher, img.shape)

    # save cipher
    meta = {'shape': img.shape, 'idx': idx.tolist(), 'seed': seed.hex()}
    with open(out_cipher, 'wb') as f:
        f.write(json.dumps(meta).encode() + b'\n--CIPHER--\n' + cipher)
    cv2.imwrite(out_encimg, cipher_img)

    print(f"   ▪ Cipher file: {out_cipher}")
    print(f"   ▪ Encrypted image saved: {out_encimg}")
    print("============================================================\n")

    # === STEP 8: Display side-by-side comparison ===
    print("📸 [STEP 8] Displaying Original and Encrypted Image Comparison...")
    orig, enc = cv2.imread(target_path), cv2.imread(out_encimg)

    h = 400
    orig_r = cv2.resize(orig, (int(orig.shape[1] * h / orig.shape[0]), h))
    enc_r = cv2.resize(enc, (int(enc.shape[1] * h / enc.shape[0]), h))

    def label(img, text):
        labeled = img.copy()
        cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return labeled

    vis = np.hstack([label(orig_r, "Original Image"), label(enc_r, "Encrypted Image")])
    comparison_path = os.path.join(results_dir, "comparison.png")
    cv2.imwrite(comparison_path, vis)
    print(f"✅ Comparison image saved → {comparison_path}\n")

    cv2.imshow("Comparison: Original vs Encrypted", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # === STEP 9: Evaluation ===
    print("📊 [STEP 9] Evaluating Encryption Strength (NPCR & UACI)...")
    npcr, uaci = npcr_uaci(target_path, out_encimg)
    print(f"   ▪ NPCR = {npcr:.2f}%")
    print(f"   ▪ UACI = {uaci:.2f}%")
    print("============================================================")
    print("🏁 [COMPLETE] RANDOM MOTION ENCRYPTION PIPELINE FINISHED.")
    print("============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Motion-Based Image Encryption Demo")
    parser.add_argument("--frames", default="data/lava_lamp_video.mp4")
    parser.add_argument("--target", default="data/test_image.png")
    parser.add_argument("--cipher", default="results/cipher.bin")
    parser.add_argument("--encimg", default="results/encrypted.png")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    encrypt_image_with_motion(
        resolve_path(args.frames),
        resolve_path(args.target),
        resolve_path(args.cipher),
        resolve_path(args.encimg),
        fast=args.fast
    )
