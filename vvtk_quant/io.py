
import cbitstruct as bs                # C-accelerated bitstruct clone

# (Optional) tiny cache so we only parse/compile a format once
_fmt_cache: dict[tuple[int, int], bs.CompiledFormat] = {}

def _get_cf(n: int, count: int) -> bs.CompiledFormat:
    """
    Return a cached compiled-format object for `count` unsigned ints,
    each `n` bits wide.
    """
    key = (n, count)
    cf = _fmt_cache.get(key)
    if cf is None:
        fmt = ("u%d" % n) * count    # e.g. 'u5u5u5u5'
        cf = bs.compile(fmt)
        _fmt_cache[key] = cf
    return cf


# -------------------------------------------------------------------
# Public API – mirroring your original signatures
# -------------------------------------------------------------------

def pack(values: list[int], n: int) -> bytes:
    """
    values : iterable of non-negative ints, each < 2**n
    n      : bit-width per value (1 ≤ n ≤ 64)
    returns: packed bytes produced by cbitstruct
    """
    if not (1 <= n <= 64):
        raise ValueError("n must be between 1 and 64")
    count = len(values)
    cf = _get_cf(n, count)

    # cbitstruct raises its own ValueError if a value won't fit,
    # so we don't need the mask/check loop from the pure-Python version.
    return cf.pack(*values)


def unpack(data: bytes, n: int, count: int) -> list[int]:
    """
    data  : bytes previously produced by `pack`
    n     : same bit-width that was used to pack
    count : number of integers expected
    """
    if not (1 <= n <= 64):
        raise ValueError("n must be between 1 and 64")
    cf = _get_cf(n, count)

    # cf.unpack() returns a tuple; convert to list to match original API
    result = list(cf.unpack(data))

    if len(result) != count:
        raise ValueError("Data truncated or count wrong")
    return result

