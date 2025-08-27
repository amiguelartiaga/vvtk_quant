# gen_float_formats_tables.py
# - No CLI. Just run: python gen_float_formats_tables.py
# - Produces CSVs and prints tables for:
#     * FP8 E4M3  (IEEE-like and FN)
#     * FP8 E5M2  (IEEE-like and FN)
#     * FP4 E2M1  (IEEE-like and FN)
#     * FP4 E3M0  (IEEE-like and FN)
#
# "IEEE-like" = all-ones exponent is reserved for Inf/NaN (we EXCLUDE those here)
# "FN"        = finite-numbers-only; all exponent patterns are finite (no Inf/NaN)

import math
import csv
from typing import List, Tuple

# -------------------------------
# Generic utilities for (e, m)
# -------------------------------

def bias_for(e: int) -> int:
    if e <= 0:
        raise ValueError("Exponent bits e must be >= 1")
    return (1 << (e - 1)) - 1

def nbits_total(e: int, m: int) -> int:
    return 1 + e + m  # sign + exponent + mantissa

def hex_width_for_bits(nbits: int) -> int:
    # 4 bits per hex digit, round up
    return (nbits + 3) // 4

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

# Round-to-nearest, ties-to-even via Python's round()
def rnte(x: float) -> int:
    return int(round(x))

# -------------------------------
# Encode / Decode (generic)
# -------------------------------

def encode_scalar(x: float, e: int, m: int, mode: str = "ieee") -> int:
    """
    Encode a Python float into an integer bit pattern for a (1+e+m)-bit float.
    mode:
      - 'ieee': all-ones exponent is reserved (we SATURATE to max finite).
      - 'fn'  : finite-numbers-only; all exponents are finite (we SATURATE to max finite).
    Notes:
      * Subnormals included.
      * We emit +0 for zero (single numeric zero).
    """
    if e < 1 or m < 0:
        raise ValueError("e must be >=1, m must be >=0")
    mode = mode.lower()
    if mode not in ("ieee", "fn"):
        raise ValueError("mode must be 'ieee' or 'fn'")

    b = bias_for(e)
    M = 1 << m
    Emax_field = (1 << e) - 1
    # Highest finite exponent field (for saturation)
    E_hi = (Emax_field - 1) if mode == "ieee" else Emax_field

    xf = float(x)
    if xf == 0.0:
        return 0  # +0 only

    s = 1 if xf < 0.0 else 0
    ax = abs(xf)

    # Unbiased exponent k = floor(log2(|x|))
    k = math.floor(math.log2(ax))

    # Normal range (in unbiased terms):
    #   IEEE: k in [1-b, (Emax-1)-b]
    #   FN  : k in [1-b, Emax-b]
    k_min = 1 - b
    k_max = E_hi - b

    if k < k_min:
        # Subnormal
        # value = +/- (mant / 2^m) * 2^(1 - b)
        # mant = round(|x| / (2^(1-b)) * 2^m) = round(|x| * 2^(m + b - 1))
        mant_sub = rnte(ax * (2.0 ** (m + b - 1)))
        mant_sub = clamp(mant_sub, 0, M - 1)
        if mant_sub == 0:
            return 0  # underflow to zero
        code = (s << (e + m)) | (0 << m) | mant_sub
        return code

    if k > k_max:
        # Overflow -> saturate to max finite
        E_field = E_hi
        mant = M - 1
        code = (s << (e + m)) | (E_field << m) | mant
        return code

    # Normal
    E_field = k + b  # in [1, E_hi]
    # mantissa rounding: m = round((|x|/2^k - 1) * 2^m)
    m_real = ax / (2.0 ** k) - 1.0
    mant = rnte(m_real * M)
    mant = clamp(mant, 0, M - 1)
    code = (s << (e + m)) | (E_field << m) | mant
    return code

def decode_scalar(code: int, e: int, m: int, mode: str = "ieee") -> float:
    """
    Decode an integer bit pattern into a Python float for a (1+e+m)-bit float.
    (We handle IEEE 'specials' conceptually, but this script never emits them.)
    """
    if e < 1 or m < 0:
        raise ValueError("e must be >=1, m must be >=0")
    mode = mode.lower()
    if mode not in ("ieee", "fn"):
        raise ValueError("mode must be 'ieee' or 'fn'")

    b = bias_for(e)
    M = 1 << m
    Emax_field = (1 << e) - 1

    s = (code >> (e + m)) & 0x1
    E_field = (code >> m) & ((1 << e) - 1)
    mant = code & (M - 1 if M > 0 else 0)

    if E_field == 0:
        # subnormal or zero
        if m == 0 or mant == 0:
            v = 0.0
        else:
            v = (mant / M) * (2.0 ** (1 - b))
    else:
        if mode == "ieee" and E_field == Emax_field:
            # Would be Inf/NaN; we don't generate these in this script.
            # Return NaN to flag if encountered accidentally.
            return float("nan")
        # normal
        v = (1.0 + (mant / M if M > 0 else 0.0)) * (2.0 ** (E_field - b))

    return -v if s else v

# -------------------------------
# Enumerate all finite values (numeric)
# -------------------------------

def enumerate_finite_values(e: int, m: int, mode: str) -> List[float]:
    """
    All finite representable numeric values (including subnormals), sorted.
    Zero appears once (numeric).
    """
    mode = mode.lower()
    if mode not in ("ieee", "fn"):
        raise ValueError("mode must be 'ieee' or 'fn'")
    b = bias_for(e)
    M = 1 << m
    values = set()

    # subnormals (E=0, mant=1..M-1)
    for mant in range(1, M):
        v = (mant / M) * (2.0 ** (1 - b))
        values.add(v); values.add(-v)

    # zero
    values.add(0.0)

    # normals: E in 1..E_hi
    Emax_field = (1 << e) - 1
    E_hi = (Emax_field - 1) if mode == "ieee" else Emax_field
    for E_field in range(1, E_hi + 1):
        scale = 2.0 ** (E_field - b)
        for mant in range(0, M if M > 0 else 1):
            v = (1.0 + (mant / M if M > 0 else 0.0)) * scale
            values.add(v); values.add(-v)

    return sorted(values)

# -------------------------------
# Tabulation + CSV
# -------------------------------

def bit_fields(code: int, e: int, m: int) -> Tuple[int, int, int]:
    s = (code >> (e + m)) & 0x1
    E_field = (code >> m) & ((1 << e) - 1)
    mant = code & ((1 << m) - 1 if m > 0 else 0)
    return s, E_field, mant

def bitstring(code: int, e: int, m: int) -> str:
    nbits = nbits_total(e, m)
    return format(code, f"0{nbits}b")

def write_and_print_table(fmt_name: str, e: int, m: int, mode: str, csv_path: str):
    vals = enumerate_finite_values(e, m, mode)
    nbits = nbits_total(e, m)
    hexw = hex_width_for_bits(nbits)

    rows = []
    for idx, x in enumerate(vals):
        code = encode_scalar(x, e, m, mode)
        y = decode_scalar(code, e, m, mode)
        s, E_field, mant = bit_fields(code, e, m)
        rows.append({
            "idx": idx,
            "value": x,
            "code": code,
            "hex": f"0x{code:0{hexw}X}",
            "bits": bitstring(code, e, m),
            "sign": s,
            "exp": E_field,
            "mant": mant,
            "decoded": y,
            "abs_err": abs(x - y),
        })

    # print header
    title = f"{fmt_name} ({mode.upper()}): e={e}, m={m}, total_values={len(rows)}"
    print("\n" + title)
    print("-" * len(title))
    print(f"{'idx':>4} {'value':>14} {'code':>6} {'hex':>6} {'bits':>{1+e+m}} {'s':>2} {'E':>4} {'M':>4} {'decoded':>14} {'abs_err':>12}")
    for r in rows:
        print(f"{r['idx']:4d} {r['value']:14.6g} {r['code']:6d} {r['hex']:>6} {r['bits']:>{1+e+m}} {r['sign']:2d} {r['exp']:4d} {r['mant']:4d} {r['decoded']:14.6g} {r['abs_err']:12.6g}")

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV -> {csv_path}")

# -------------------------------
# Main: generate all tables
# -------------------------------

def main():
    tasks = [
        # name, e, m, mode, out_csv
        ("FP8_E4M3", 4, 3, "ieee", "all_fp8_e4m3_ieee.csv"),
        ("FP8_E4M3", 4, 3, "fn",   "all_fp8_e4m3_fn.csv"),
        ("FP8_E5M2", 5, 2, "ieee", "all_fp8_e5m2_ieee.csv"),
        ("FP8_E5M2", 5, 2, "fn",   "all_fp8_e5m2_fn.csv"),
        ("FP4_E2M1", 2, 1, "ieee", "all_fp4_e2m1_ieee.csv"),
        ("FP4_E2M1", 2, 1, "fn",   "all_fp4_e2m1_fn.csv"),
        ("FP4_E3M0", 3, 0, "ieee", "all_fp4_e3m0_ieee.csv"),
        ("FP4_E3M0", 3, 0, "fn",   "all_fp4_e3m0_fn.csv"),
    ]

    for name, e, m, mode, csv_path in tasks:
        write_and_print_table(name, e, m, mode, csv_path)

if __name__ == "__main__":
    main()

