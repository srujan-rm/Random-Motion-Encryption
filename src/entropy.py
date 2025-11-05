import numpy as np
import hashlib

def aggregate_entropy_bits(entropy_values):
    print("============================================================")
    print("🧩 [STEP 4A] ENTROPY AGGREGATION & TRACEABILITY REPORT")
    print("============================================================")

    # --- Raw Entropy ---
    e_min, e_max = np.min(entropy_values), np.max(entropy_values)
    print(f"   ▪ Raw entropy stats: min={e_min:.4f}, max={e_max:.4f}, mean={np.mean(entropy_values):.4f}")

    # --- Normalize to 0–255 ---
    normalized = np.interp(entropy_values, (e_min, e_max), (0, 255)).astype(np.uint8)
    print(f"   ▪ Normalized sample (first 10): {normalized[:10].tolist()}")
    print(f"   ▪ Normalized range: {normalized.min()} – {normalized.max()}")

    # --- Convert to Bitstream ---
    bits = []
    for v in normalized:
        bits.extend([int(b) for b in format(v, '08b')])
    print(f"   ▪ Bitstream length: {len(bits)} bits")
    print(f"   ▪ Bitstream (first 64 bits): {''.join(map(str, bits[:64]))}...")

    # --- Convert bits → bytes ---
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | bit
        out.append(byte)
    final_bytes = bytes(out[:64])
    print(f"   ▪ Final entropy stream length: {len(final_bytes)} bytes")
    print(f"   ▪ First 32 bytes (hex): {final_bytes[:32].hex()}...")

    # --- Shannon Entropy Estimate ---
    hist, _ = np.histogram(np.frombuffer(final_bytes, dtype=np.uint8),
                           bins=256, range=(0, 256))
    p = hist / (np.sum(hist) + 1e-9)
    shannon_bits = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    print(f"   ▪ Shannon entropy ≈ {shannon_bits:.2f} bits/byte "
          f"(~{shannon_bits * len(final_bytes):.1f} bits total)")

    print("============================================================\n")
    return final_bytes


def sha256_digest_verbose(seed_bytes: bytes) -> bytes:
    print("============================================================")
    print("🔬 [STEP 5A] CONDITIONING ENTROPY VIA SHA-256")
    print("============================================================")
    print(f"   ▪ Input seed (first 32 bytes): {seed_bytes[:32].hex()}...")
    digest = hashlib.sha256(seed_bytes).digest()
    print(f"   ▪ SHA-256 Digest (256-bit output): {digest.hex()}")
    print("============================================================\n")
    return digest
