# src/kdf_aes.py
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def derive_key_iv(seed_bytes, key_len=32, iv_len=12, info=b'rme'):
    hkdf = HKDF(algorithm=hashes.SHA256(), length=key_len+iv_len, salt=None, info=info)
    okm = hkdf.derive(seed_bytes)
    return okm[:key_len], okm[key_len:key_len+iv_len]

def encrypt_bytes(data, key, iv):
    return AESGCM(key).encrypt(iv, data, None)

def decrypt_bytes(cipher, key, iv):
    return AESGCM(key).decrypt(iv, cipher, None)
