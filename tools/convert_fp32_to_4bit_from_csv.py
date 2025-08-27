# convert_fp32_to_4bit_packed_from_table_relaxed.py
# Packs two 4-bit codes per byte, using a (possibly sparse) 4-bit CSV table (code,decoded).
# FIX: Unsqueeze now uses axes *input* for opset >= 13 (attribute for older opsets).

import argparse, csv, math
from typing import Dict, List, Tuple
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto, shape_inference

# ---------- table I/O (relaxed) ----------
def load_4bit_table_relaxed(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    present: Dict[int, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV appears empty or malformed.")
        for row in r:
            if "code" not in row:
                raise ValueError("CSV missing 'code' column.")
            code_str = str(row["code"]).strip()
            if not code_str:
                continue
            try:
                c = int(code_str, 0)  # allow '0x..'
            except ValueError:
                c = int(code_str)
            if not (0 <= c <= 15):
                continue

            d_str = row.get("decoded", row.get("value"))
            if d_str is None:
                continue
            try:
                d = float(d_str)
            except ValueError:
                continue
            if not math.isfinite(d):
                # Skip Inf/NaN rows
                continue

            if c not in present:
                present[c] = float(np.float32(d))

    if len(present) == 0:
        raise ValueError("No finite entries found in CSV (code->decoded).")

    # Build 16-entry LUT (NaN for missing codes)
    lut = np.full((16,), np.nan, dtype=np.float32)
    for c, d in present.items():
        lut[c] = np.float32(d)

    # Quantization set: unique decoded values + preferred code (smallest code for duplicates)
    val2code: Dict[float, int] = {}
    for c, d in present.items():
        if d not in val2code or c < val2code[d]:
            val2code[d] = c
    pairs = sorted(((float(v), int(c)) for v, c in val2code.items()), key=lambda t: t[0])
    uniq_vals = np.array([p[0] for p in pairs], dtype=np.float32)
    codes_for_vals = np.array([p[1] for p in pairs], dtype=np.int64)
    return lut, uniq_vals, codes_for_vals

# ---------- quantize & pack ----------
def quantize_to_codes_4bit(x: np.ndarray, uniq_vals: np.ndarray, codes_for_vals: np.ndarray) -> np.ndarray:
    xv = x.astype(np.float32, copy=False).ravel()
    pos = np.searchsorted(uniq_vals, xv, side="left")
    left  = np.clip(pos - 1, 0, len(uniq_vals) - 1)
    right = np.clip(pos,     0, len(uniq_vals) - 1)
    pick_right = np.abs(xv - uniq_vals[right]) < np.abs(xv - uniq_vals[left])
    idx = np.where(pick_right, right, left)
    codes = codes_for_vals[idx].astype(np.uint8)
    return codes.reshape(x.shape)

def pack_nibbles(codes: np.ndarray) -> Tuple[np.ndarray, int]:
    flat = codes.astype(np.uint8, copy=False).ravel()
    N = flat.size
    if N % 2 != 0:
        flat = np.concatenate([flat, np.array([0], dtype=np.uint8)])  # pad one nibble
    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    packed = (lo | hi).astype(np.uint8)
    return packed, int(N)

# ---------- graph helpers ----------
def collect_weight_slots(g: onnx.GraphProto):
    init_names = {i.name for i in g.initializer}
    slots = []
    for n in g.node:
        if n.op_type == "Conv":
            if len(n.input) >= 2 and n.input[1] in init_names: slots.append((n, 1))
        elif n.op_type == "Gemm":
            if len(n.input) >= 2 and n.input[1] in init_names: slots.append((n, 1))
        elif n.op_type == "MatMul":
            if len(n.input) >= 2 and n.input[1] in init_names: slots.append((n, 1))
    return slots

def ensure_lut(graph: onnx.GraphProto, lut: np.ndarray, name="LUT4B") -> str:
    for ini in graph.initializer:
        if ini.name == name:
            return name
    graph.initializer.append(onh.from_array(lut.astype(np.float32), name=name))
    graph.value_info.append(oh.make_tensor_value_info(name, TensorProto.FLOAT, [lut.shape[0]]))
    return name

def add_unsqueeze(g: onnx.GraphProto, tensor_name: str, axes: List[int], out_name: str, base_name: str, opset: int):
    """
    Unsqueeze helper that uses:
      - opset >= 13: axes as second input tensor
      - opset <  13: axes attribute
    """
    if opset >= 13:
        axes_name = base_name + "_axes"
        g.initializer.append(onh.from_array(np.array(axes, dtype=np.int64), name=axes_name))
        g.node.append(oh.make_node("Unsqueeze", [tensor_name, axes_name], [out_name], name=base_name+"_unsq"))
    else:
        g.node.append(oh.make_node("Unsqueeze", [tensor_name], [out_name], axes=axes, name=base_name+"_unsq"))

def add_unpack_decode_subgraph(g: onnx.GraphProto, u8_name: str, out_name: str, N: int, out_shape: List[int], lut_name: str, opset: int):
    """
    Build:
      u8 -> Cast(INT32) -> {low=Mod 16, high=Div 16} -> Unsqueeze both -> Concat(axis=1)
         -> Reshape([-1]) -> Slice(0:N) -> Reshape(out_shape)
         -> Cast(INT64) -> Gather(lut) -> out_name
    """
    i32 = u8_name + "_i32"
    g.node.append(oh.make_node("Cast", [u8_name], [i32], to=TensorProto.INT32, name=u8_name+"_cast_u8_to_i32"))

    c16 = u8_name + "_c16"
    g.initializer.append(onh.from_array(np.array([16], dtype=np.int32), name=c16))
    low  = u8_name + "_low"
    high = u8_name + "_high"
    g.node.append(oh.make_node("Mod", [i32, c16], [low],  name=u8_name+"_mod16"))
    g.node.append(oh.make_node("Div", [i32, c16], [high], name=u8_name+"_div16"))

    # Unsqueeze to [K,1] then Concat -> [K,2]
    low_u  = low  + "_u"
    high_u = high + "_u"
    add_unsqueeze(g, low,  [1], low_u,  base_name=low,  opset=opset)
    add_unsqueeze(g, high, [1], high_u, base_name=high, opset=opset)

    inter2 = u8_name + "_inter2"
    g.node.append(oh.make_node("Concat", [low_u, high_u], [inter2], axis=1, name=u8_name+"_concat_low_high"))

    # Reshape [K,2] -> [-1]  (row-major => [low0, high0, low1, high1, ...])
    shape_m1 = u8_name + "_shape_m1"
    g.initializer.append(onh.from_array(np.array([-1], dtype=np.int64), name=shape_m1))
    flat2 = u8_name + "_flat2"
    g.node.append(oh.make_node("Reshape", [inter2, shape_m1], [flat2], name=u8_name+"_reshape_flat2"))

    # Slice first N (drop padding nibble if it existed)
    start0 = u8_name + "_c0"; endN = u8_name + "_cN"; axes0 = u8_name + "_axes0"
    g.initializer.append(onh.from_array(np.array([0], dtype=np.int64), name=start0))
    g.initializer.append(onh.from_array(np.array([N], dtype=np.int64), name=endN))
    g.initializer.append(onh.from_array(np.array([0], dtype=np.int64), name=axes0))
    flatN = u8_name + "_flatN"
    g.node.append(oh.make_node("Slice", [flat2, start0, endN, axes0], [flatN], name=u8_name+"_sliceN"))

    # Reshape to original weight shape and decode via LUT
    out_shape_name = u8_name + "_outshape"
    g.initializer.append(onh.from_array(np.array(out_shape, dtype=np.int64), name=out_shape_name))
    codes_full = u8_name + "_codes_full"
    g.node.append(oh.make_node("Reshape", [flatN, out_shape_name], [codes_full], name=u8_name+"_reshape_out"))

    idx64 = u8_name + "_idx64"
    g.node.append(oh.make_node("Cast", [codes_full], [idx64], to=TensorProto.INT64, name=u8_name+"_cast_i32_to_i64"))
    g.node.append(oh.make_node("Gather", [lut_name, idx64], [out_name], name=u8_name+"_gather_decode", axis=0))
    g.value_info.append(oh.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape))

# ---------- main conversion ----------
def convert(table_csv: str, in_model: str, out_model: str, lut_name="LUT4B"):
    lut, uniq_vals, codes_for_vals = load_4bit_table_relaxed(table_csv)
    model = onnx.load(in_model)
    g = model.graph

    # Detect opset (default to 13 if multiple domains).
    opset_imports = {imp.domain: imp.version for imp in model.opset_import}
    opset = opset_imports.get("", max(opset_imports.values()))  # default to highest if not found
    if opset is None:
        opset = 13  # safe default

    init_by_name = {i.name: i for i in g.initializer}
    slots = collect_weight_slots(g)
    lut_name = ensure_lut(g, lut, name=lut_name)

    converted = 0
    for node, w_idx in slots:
        w_name = node.input[w_idx]
        t = init_by_name[w_name]
        if t.data_type != TensorProto.FLOAT:
            continue

        arr = onh.to_array(t).astype(np.float32)
        codes = quantize_to_codes_4bit(arr, uniq_vals, codes_for_vals)
        packed, N = pack_nibbles(codes)

        u8_name = w_name + "_packed_u8"
        u8_init = onh.from_array(packed.view(np.uint8), name=u8_name)
        u8_init.data_type = TensorProto.UINT8
        g.initializer.append(u8_init)

        dec_name = w_name + "_decoded_f32"
        add_unpack_decode_subgraph(
            g, u8_name=u8_name, out_name=dec_name, N=N,
            out_shape=list(t.dims), lut_name=lut_name, opset=opset
        )

        node.input[w_idx] = dec_name
        for ini in list(g.initializer):
            if ini.name == w_name:
                g.initializer.remove(ini)
                break

        converted += 1

    # (Optional but nice) run shape inference
    model = shape_inference.infer_shapes(model)
    onnx.save(model, out_model)
    print(f"[OK] Packed {converted} weight tensor(s) into 4-bit (2 per byte).")
    print(f"     LUT has {np.sum(np.isfinite(lut))} finite entries out of 16 (missing filled with NaN).")
    print(f"     Opset detected: {opset}")
    print(f"     Wrote -> {out_model}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_csv", help="CSV with subset of codes 0..15 (columns: code, decoded or value).")
    ap.add_argument("in_fp32",   help="Input FP32 ONNX.")
    ap.add_argument("out_onnx",  help="Output ONNX with packed 4-bit + unpack/LUT.")
    ap.add_argument("--lut_name", default="LUT4B")
    args = ap.parse_args()
    convert(args.table_csv, args.in_fp32, args.out_onnx, lut_name=args.lut_name)

if __name__ == "__main__":
    main()

