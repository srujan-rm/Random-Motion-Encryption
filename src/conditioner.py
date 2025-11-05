# src/conditioner.py
import hashlib

def sha256_digest(seed_bytes: bytes) -> bytes:
    return hashlib.sha256(seed_bytes).digest()
