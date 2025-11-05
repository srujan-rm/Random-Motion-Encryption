# src/utils.py
import numpy as np
import cv2

def image_to_bytes(img):
    return img.tobytes()

def bytes_to_image(b, shape):
    return np.frombuffer(b, dtype=np.uint8).reshape(shape)

def shuffle_pixels(img, seed_bytes):
    h, w, c = img.shape
    rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], 'big'))
    flat = img.reshape(-1, c).copy()
    idx = np.arange(flat.shape[0])
    rng.shuffle(idx)
    return flat[idx].reshape(h, w, c), idx

def unshuffle_pixels(img, idx):
    h, w, c = img.shape
    flat = img.reshape(-1, c).copy()
    unflat = np.zeros_like(flat)
    unflat[idx] = flat
    return unflat.reshape(h, w, c)

def visualize_cipher_bytes_as_image(cipher, shape):
    h, w, c = shape
    arr = np.frombuffer(cipher, dtype=np.uint8)
    size = h * w * c
    arr = np.tile(arr, int(size/len(arr)) + 1)[:size] if len(arr) < size else arr[:size]
    return arr.reshape((h, w, c))
