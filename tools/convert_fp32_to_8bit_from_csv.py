# convert_fp32_to_lut_from_table.py
# Usage:
#   python convert_fp32_to_lut_from_table.py table.csv in_fp32.onnx out_lut.onnx
#
# The CSV must have at least the columns:
#   idx,value,code,hex,bits,sign,exp,mant,decoded,abs_err
# We use ONLY 'code' (int) and 'decoded' (float). Others are ignored.
#
# What it does:
#   * Builds one global LUT where LUT[code] = decoded (float32). Any missing codes are set to NaN.
#   * Quantizes weights by nearest neighbor over the set of distinct 'decoded' values in the table:
#         x  -> argmin_v |x - v|  -> emit the 'code' associated with v (ties break toward the lower value)
#   * Converts ONLY weights (Conv W, Gemm B, MatMul weight). Biases remain FP32.
#   * Adds:
#         uint8_weight --Cast--> int64_indices --Gather(LUT)--> float32_weight
#     so compute stays fully FP32.
#
# Notes:
#   * Works for FP8 (e4m3/e5m2), FP4, or any custom format as long as the CSV maps code->decoded.
#   * The quantizer never emits codes that aren't present in the CSV.
#   * Opset 17–18 recommended for broad compatibility.

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
    """
    Read the CSV table and return:
      - code_to_decoded: dict[code] = decoded float
      - values_sorted:   sorted unique list of decoded values
      - codes_sorted:    codes aligned with values_sorted (preferred code per value)
    If multiple codes map to the same decoded (e.g., +0/-0), the smaller code is preferred.
    """
    code_to_decoded: Dict[int, float] = {}
    # We also build a preferred value->code map (smallest code chosen on duplicates)
    value_to_code: Dict[float, int] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # be robust to column name variants; we prefer 'decoded', else fallback to 'value'
        for row in reader:
            code_str = row.get("code")
            if code_str is None:
                raise ValueError("CSV is missing 'code' column")
            dec_str = row.get("decoded", row.get("value"))
            if dec_str is None:
                raise ValueError("CSV is missing both 'decoded' and 'value' columns")

            try:
                code = int(code_str, 0)  # allow "0x.." or decimal
            except ValueError:
                code = int(code_str)

            try:
                decoded = float(dec_str)
            except ValueError:
                # skip rows that don't parse
                continue

            if math.isnan(decoded) or math.isinf(decoded):
                # We ignore specials; we won't emit these codes.
                continue

            code_to_decoded[code] = np.float32(decoded).item()

            # pick a preferred code for this decoded value (smallest code)
            if decoded not in value_to_code or code < value_to_code[decoded]:
                value_to_code[decoded] = code

    if not code_to_decoded:
        raise ValueError("No valid (code, decoded) rows found in CSV.")

    # Build sorted unique decoded values and aligned preferred codes
    items = sorted(((float(v), c) for v, c in value_to_code.items()), key=lambda t: t[0])
    values_sorted = [v for v, _ in items]
    codes_sorted = [c for _, c in items]

    return code_to_decoded, values_sorted, codes_sorted


def build_lut_from_table(code_to_decoded: Dict[int, float]) -> np.ndarray:
    """
    Build a 1D float32 LUT where lut[code] = decoded.
    Any codes not present in the table get NaN (we never emit them).
    The LUT length = max_code + 1 (min 1).
    """
    max_code = max(code_to_decoded.keys())
    length = max(max_code + 1, 1)
    lut = np.full((length,), np.nan, dtype=np.float32)
    for c, v in code_to_decoded.items():
        if 0 <= c < length:
            lut[c] = np.float32(v)
    return lut


def quantize_to_codes(arr: np.ndarray, values_sorted: List[float], codes_sorted: List[int]) -> np.ndarray:
    """
    Quantize a float32 array to UINT8 codes using nearest neighbor over values_sorted.
    - values_sorted: sorted list of decoded floats (unique)
    - codes_sorted:  matching codes for each value in values_sorted
    Ties break to the lower value (left neighbor).
    """
    vals = np.asarray(values_sorted, dtype=np.float32)
    codes = np.asarray(codes_sorted, dtype=np.int64)
    x = arr.astype(np.float32, copy=False).ravel()

    # vectorized nearest neighbor via searchsorted
    pos = np.searchsorted(vals, x, side="left")
    left_idx = np.clip(pos - 1, 0, len(vals) - 1)
    right_idx = np.clip(pos,     0, len(vals) - 1)

    left_diff = np.abs(x - vals[left_idx])
    right_diff = np.abs(x - vals[right_idx])

    choose_right = right_diff < left_diff
    idx = np.where(choose_right, right_idx, left_idx)

    chosen_codes = codes[idx].astype(np.uint8)
    return chosen_codes.reshape(arr.shape)


def collect_weight_slots(graph: onnx.GraphProto) -> List[Tuple[onnx.NodeProto, int]]:
    """
    Return list of (node, input_index) pairs for weights to convert:
      - Conv:  input[1] (W)
      - Gemm:  input[1] (B)
      - MatMul:input[1] (weight)
    """
    init_names = {i.name for i in graph.initializer}
    slots = []
    for n in graph.node:
        if n.op_type == "Conv":
            if len(n.input) >= 2 and n.input[1] in init_names:
                slots.append((n, 1))
        elif n.op_type == "Gemm":
            if len(n.input) >= 2 and n.input[1] in init_names:
                slots.append((n, 1))
        elif n.op_type == "MatMul":
            if len(n.input) >= 2 and n.input[1] in init_names:
                slots.append((n, 1))
    return slots


def ensure_lut_initializer(graph: onnx.GraphProto, lut: np.ndarray, name: str = "FP_LUT_FROM_CSV") -> str:
    # Reuse if same name exists; else add it.
    for ini in graph.initializer:
        if ini.name == name:
            return name
    graph.initializer.append(onh.from_array(lut.astype(np.float32), name=name))
    graph.value_info.append(oh.make_tensor_value_info(name, TensorProto.FLOAT, [lut.shape[0]]))
    return name


def convert_with_csv_table(csv_path: str, in_model: str, out_model: str, lut_name: str = "FP_LUT_FROM_CSV"):
    # 1) Read table and build LUT + quantizer
    code_to_decoded, values_sorted, codes_sorted = load_table(csv_path)
    lut = build_lut_from_table(code_to_decoded)

    # 2) Load model
    model = onnx.load(in_model)
    g = model.graph
    init_by_name = {i.name: i for i in g.initializer}

    # 3) Find weights
    weight_slots = collect_weight_slots(g)

    # 4) Add LUT initializer (once)
    lut_name = ensure_lut_initializer(g, lut, name=lut_name)

    converted = 0
    for node, w_idx in weight_slots:
        w_name = node.input[w_idx]
        t = init_by_name[w_name]
        if t.data_type != TensorProto.FLOAT:
            # Only convert FP32 weights; leave others untouched
            continue

        # 5) Quantize to codes using ONLY the CSV value set
        arr = onh.to_array(t).astype(np.float32)
        codes = quantize_to_codes(arr, values_sorted, codes_sorted)

        # 6) Replace initializer with UINT8 codes
        u8_name = w_name + "_codes_u8"
        u8_init = onh.from_array(codes.view(np.uint8), name=u8_name)
        u8_init.data_type = TensorProto.UINT8
        g.initializer.append(u8_init)

        # 7) Decode subgraph: UINT8 -> INT64 -> GATHER(LUT) -> FLOAT
        i64_name = w_name + "_idx_i64"
        dec_name = w_name + "_decoded_f32"
        g.node.append(oh.make_node("Cast",   [u8_name],      [i64_name], to=TensorProto.INT64, name=w_name+"_cast_u8_to_i64"))
        g.node.append(oh.make_node("Gather", [lut_name, i64_name], [dec_name], name=w_name+"_gather_decode", axis=0))
        g.value_info.append(oh.make_tensor_value_info(dec_name, TensorProto.FLOAT, t.dims))

        # Redirect user node to decoded
        node.input[w_idx] = dec_name

        # Remove original FP32 weight initializer
        for ini in list(g.initializer):
            if ini.name == w_name:
                g.initializer.remove(ini)
                break

        converted += 1

    model = shape_inference.infer_shapes(model)
    onnx.save(model, out_model)
    print(f"[OK] Converted {converted} weight tensor(s) using CSV table '{csv_path}'.")
    print(f"     Wrote -> {out_model}")
    print(f"     LUT length = {lut.shape[0]} (float32).")


def main():
    ap = argparse.ArgumentParser(description="Convert FP32 ONNX weights to table-coded UINT8 + LUT decode using ONLY a CSV table.")
    ap.add_argument("csv", help="CSV table with columns including 'code' and 'decoded'.")
    ap.add_argument("in_fp32", help="Input FP32 ONNX model.")
    ap.add_argument("out_lut", help="Output ONNX with UINT8 codes + LUT decode.")
    ap.add_argument("--lut_name", default="FP_LUT_FROM_CSV", help="Name for the LUT initializer.")
    args = ap.parse_args()

    convert_with_csv_table(args.csv, args.in_fp32, args.out_lut, lut_name=args.lut_name)


if __name__ == "__main__":
    main()


