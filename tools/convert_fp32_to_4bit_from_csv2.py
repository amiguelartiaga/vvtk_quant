# convert_fp32_to_lut_from_table.py
#
# Usage for 8-bit:
#   python convert_fp32_to_lut_from_table.py table8.csv in_fp32.onnx out_lut8.onnx --bits 8
#
# Usage for 4-bit:
#   python convert_fp32_to_lut_from_table.py table4.csv in_fp32.onnx out_lut4.onnx --bits 4
#
# What it does:
#   * For 4-bit mode, it correctly generates a model with opset version 18 or higher
#     to support the necessary BitwiseAnd operator.

import argparse
import csv
import math
from typing import Dict, Tuple, List

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto, shape_inference


def load_table(csv_path: str) -> Tuple[Dict[int, float], List[float], List[int]]:
    """Reads the code-to-value mapping from the CSV."""
    code_to_decoded: Dict[int, float] = {}
    value_to_code: Dict[float, int] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code_str, dec_str = row.get("code"), row.get("decoded", row.get("value"))
            if code_str is None or dec_str is None: continue
            try:
                code, decoded = int(code_str, 0), float(dec_str)
            except (ValueError, TypeError):
                continue
            if math.isnan(decoded) or math.isinf(decoded): continue

            code_to_decoded[code] = np.float32(decoded).item()
            if decoded not in value_to_code or code < value_to_code[decoded]:
                value_to_code[decoded] = code

    if not code_to_decoded: raise ValueError("No valid (code, decoded) rows found in CSV.")
    items = sorted(((float(v), c) for v, c in value_to_code.items()), key=lambda t: t[0])
    return code_to_decoded, [v for v, _ in items], [c for _, c in items]


def build_lut_from_table(code_to_decoded: Dict[int, float]) -> np.ndarray:
    """Builds a float32 LUT from the code-to-value mapping."""
    max_code = max(code_to_decoded.keys()) if code_to_decoded else -1
    lut = np.full((max(max_code + 1, 1),), np.nan, dtype=np.float32)
    for c, v in code_to_decoded.items():
        if 0 <= c < len(lut): lut[c] = np.float32(v)
    return lut


def quantize_to_codes(arr: np.ndarray, values_sorted: List[float], codes_sorted: List[int]) -> np.ndarray:
    """Quantizes a float array to integer codes via nearest neighbor search."""
    vals, codes = np.asarray(values_sorted, dtype=np.float32), np.asarray(codes_sorted, dtype=np.int64)
    x = arr.astype(np.float32, copy=False).ravel()
    pos = np.searchsorted(vals, x, side="left")
    left_idx, right_idx = np.clip(pos - 1, 0, len(vals) - 1), np.clip(pos, 0, len(vals) - 1)
    idx = np.where(np.abs(x - vals[right_idx]) < np.abs(x - vals[left_idx]), right_idx, left_idx)
    return codes[idx].reshape(arr.shape)


def pack_4bit_nibbles(codes_4bit: np.ndarray) -> np.ndarray:
    """Packs an array of 4-bit integer codes into a uint8 array."""
    flat_codes = codes_4bit.ravel().astype(np.uint8)
    if flat_codes.size % 2 != 0: flat_codes = np.append(flat_codes, 0)
    pairs = flat_codes.reshape(-1, 2)
    return ((pairs[:, 0] << 4) | pairs[:, 1]).astype(np.uint8)


def collect_weight_slots(graph: onnx.GraphProto) -> List[Tuple[onnx.NodeProto, int]]:
    """Finds all weights in Conv, Gemm, and MatMul nodes to be quantized."""
    init_names = {i.name for i in graph.initializer}
    return [(n, 1) for n in graph.node if n.op_type in {"Conv", "Gemm", "MatMul"} and len(n.input) >= 2 and n.input[1] in init_names]


def ensure_lut_initializer(graph: onnx.GraphProto, lut: np.ndarray, name: str) -> str:
    """Adds the LUT as an initializer to the graph if it doesn't exist."""
    if any(ini.name == name for ini in graph.initializer): return name
    graph.initializer.append(onh.from_array(lut.astype(np.float32), name=name))
    graph.value_info.append(oh.make_tensor_value_info(name, TensorProto.FLOAT, [len(lut)]))
    return name


def convert_with_csv_table(csv_path: str, in_model: str, out_model: str, bits: int, lut_name: str):
    if bits not in [4, 8]: raise ValueError("Unsupported bit width. Must be 4 or 8.")
    
    code_to_decoded, values_sorted, codes_sorted = load_table(csv_path)
    lut = build_lut_from_table(code_to_decoded)

    max_code = max(codes_sorted) if codes_sorted else -1
    if bits == 4 and max_code > 15: raise ValueError(f"4-bit mode, but CSV has codes > 15 (max: {max_code}).")
    if bits == 8 and max_code > 255: raise ValueError(f"8-bit mode, but CSV has codes > 255 (max: {max_code}).")

    model = onnx.load(in_model)
    g = model.graph
    init_by_name = {i.name: i for i in g.initializer}

    weight_slots = collect_weight_slots(g)
    lut_name = ensure_lut_initializer(g, lut, lut_name)
    converted = 0

    for node, w_idx in weight_slots:
        w_name = node.input[w_idx]
        t = init_by_name.get(w_name)
        if not t or t.data_type != TensorProto.FLOAT: continue

        arr = onh.to_array(t)
        codes = quantize_to_codes(arr, values_sorted, codes_sorted)
        dec_name = w_name + "_decoded_f32"

        if bits == 8:
            u8_name = w_name + "_codes_u8"
            g.initializer.append(onh.from_array(codes.astype(np.uint8), name=u8_name))
            i64_name = w_name + "_idx_i64"
            g.node.extend([
                oh.make_node("Cast", [u8_name], [i64_name], to=TensorProto.INT64),
                oh.make_node("Gather", [lut_name, i64_name], [dec_name], axis=0)
            ])
        elif bits == 4:
            packed_name = w_name + "_packed_u4_as_u8"
            g.initializer.append(onh.from_array(pack_4bit_nibbles(codes), name=packed_name))

            shift_const = onh.from_array(np.array(4, dtype=np.uint8), name=f"{w_name}_shift_const")
            mask_const = onh.from_array(np.array(0x0F, dtype=np.uint8), name=f"{w_name}_mask_const")
            g.initializer.extend([shift_const, mask_const])

            high_nibbles, low_nibbles = f"{w_name}_high_nibbles", f"{w_name}_low_nibbles"
            g.node.extend([
                oh.make_node("BitShift", [packed_name, shift_const.name], [high_nibbles], direction="RIGHT"),
                oh.make_node("BitwiseAnd", [packed_name, mask_const.name], [low_nibbles])
            ])

            unsqueeze_axes = onh.from_array(np.array([1], dtype=np.int64), name=f"{w_name}_unsqueeze_axes")
            g.initializer.append(unsqueeze_axes)
            unpacked_flat = f"{w_name}_unpacked_flat"
            g.node.extend([
                oh.make_node("Unsqueeze", [high_nibbles, unsqueeze_axes.name], [f"{high_nibbles}_u"]),
                oh.make_node("Unsqueeze", [low_nibbles, unsqueeze_axes.name], [f"{low_nibbles}_u"]),
                oh.make_node("Concat", [f"{high_nibbles}_u", f"{low_nibbles}_u"], [f"{w_name}_interleaved"], axis=1),
            ])
            
            flatten_shape = onh.from_array(np.array([-1], dtype=np.int64), name=f"{w_name}_flatten_shape")
            g.initializer.append(flatten_shape)
            g.node.append(oh.make_node("Reshape", [f"{w_name}_interleaved", flatten_shape.name], [unpacked_flat]))

            indices_u8 = unpacked_flat
            if arr.size % 2 != 0:
                indices_u8 = f"{w_name}_indices_u8_sliced"
                slice_ends = onh.from_array(np.array([arr.size], dtype=np.int64), name=f"{w_name}_slice_ends")
                g.initializer.append(slice_ends)
                g.node.append(oh.make_node("Slice", [unpacked_flat, "", slice_ends.name, ""], [indices_u8]))

            orig_shape = onh.from_array(np.array(arr.shape, dtype=np.int64), name=f"{w_name}_orig_shape")
            g.initializer.append(orig_shape)
            i64_name = w_name + "_idx_i64"
            g.node.extend([
                oh.make_node("Reshape", [indices_u8, orig_shape.name], [f"{w_name}_reshaped"]),
                oh.make_node("Cast", [f"{w_name}_reshaped"], [i64_name], to=TensorProto.INT64),
                oh.make_node("Gather", [lut_name, i64_name], [dec_name], axis=0)
            ])

        g.value_info.append(oh.make_tensor_value_info(dec_name, TensorProto.FLOAT, t.dims))
        node.input[w_idx] = dec_name
        g.initializer.remove(t)
        converted += 1

    # --- THIS IS THE CRUCIAL FIX ---
    # The BitwiseAnd operator, used in 4-bit mode, requires opset 18 or higher.
    # We must update the model's opset version before saving to make it valid.
    if bits == 4:
        opset_updated = False
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset.version = 18
                print(f"[INFO] Updated ONNX opset version to {opset.version} to support BitwiseAnd operator.")
                opset_updated = True
                break
        if not opset_updated:
             print("[WARNING] Could not find the default ONNX opset to update. The model may be invalid.")

    model = shape_inference.infer_shapes(model)
    onnx.save(model, out_model)
    print(f"[OK] Converted {converted} weight tensor(s) to {bits}-bit using CSV table '{csv_path}'.")
    print(f"     Wrote -> {out_model}")


def main():
    ap = argparse.ArgumentParser(description="Convert FP32 ONNX weights to table-coded integers (4 or 8-bit) + LUT decode.")
    ap.add_argument("csv", help="CSV table with columns 'code' and 'decoded'.")
    ap.add_argument("in_fp32", help="Input FP32 ONNX model.")
    ap.add_argument("out_lut", help="Output ONNX with packed codes + LUT decode.")
    ap.add_argument("--bits", type=int, choices=[4, 8], default=8, help="Target bit width (default: 8).")
    ap.add_argument("--lut_name", default="FP_LUT_FROM_CSV", help="Name for the LUT initializer.")
    args = ap.parse_args()
    convert_with_csv_table(args.csv, args.in_fp32, args.out_lut, args.bits, args.lut_name)


if __name__ == "__main__":
    main()

