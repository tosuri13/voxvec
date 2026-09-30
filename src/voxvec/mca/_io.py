from io import Reader
from struct import Struct
from typing import Any


def read_exact(f: Reader[bytes], size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    return data


def unpack(f: Reader[bytes], fmt: Struct) -> tuple[Any, ...]:
    return fmt.unpack(read_exact(f, fmt.size))
