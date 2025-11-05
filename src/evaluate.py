# src/evaluate.py
import numpy as np, cv2, sys

def npcr_uaci(a_path, b_path):
    a, b = cv2.imread(a_path), cv2.imread(b_path)
    if a.shape != b.shape:
        raise ValueError("Image shapes differ.")
    npcr = (a != b).mean() * 100
    uaci = (np.abs(a - b).sum() / (255 * a.size)) * 100
    return npcr, uaci

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py img1 img2")
        exit()
    npcr, uaci = npcr_uaci(sys.argv[1], sys.argv[2])
    print(f"NPCR: {npcr:.2f}%  UACI: {uaci:.2f}%")
