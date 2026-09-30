import gzip
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO, Reader
from struct import Struct

from voxvec.mca import _io

__all__ = [
    "Compression",
    "Region",
    "RegionChunk",
    "load",
    "loads",
]

SECTOR_SIZE = 4096
REGION_WIDTH = 32

# NOTE: チャンクヘッダーの圧縮方式における最上位ビット。フラグがONの場合にはチャンクデータが外部ファイルに格納される
EXTERNAL_FLAG = 0x80


class Format:
    LOCATIONS = Struct(f">{REGION_WIDTH**2}I")
    TIMESTAMPS = Struct(f">{REGION_WIDTH**2}i")

    CHUNK_LENGTH = Struct(">i")
    CHUNK_COMPRESSION = Struct(">B")


class Compression(IntEnum):
    GZIP = 1
    ZLIB = 2
    NONE = 3
    LZ4 = 4


@dataclass
class RegionChunk:
    compression: Compression
    data: bytes
    timestamp: int

    def decompress(self) -> bytes:
        match self.compression:
            case Compression.GZIP:
                return gzip.decompress(self.data)
            case Compression.ZLIB:
                return zlib.decompress(self.data)
            case Compression.NONE:
                return self.data
            case _:
                raise NotImplementedError(
                    f"unsupported compression: {self.compression!r}"
                )


@dataclass
class Region:
    chunks: dict[int, RegionChunk] = field(default_factory=dict)

    def get(self, x: int, z: int) -> bytes | None:
        chunk = self.chunks.get(_index(x, z))
        return None if chunk is None else chunk.decompress()


def load(f: Reader[bytes]) -> Region:
    raw = f.read()
    region = Region()

    # NOTE: 空のファイルは、チャンクが1つも存在しないリージョンとして扱う
    if not raw:
        return region

    data = BytesIO(raw)

    locations = _io.unpack(data, Format.LOCATIONS)
    timestamps = _io.unpack(data, Format.TIMESTAMPS)

    for idx, location in enumerate(locations):
        offset, sector_count = location >> 8, location & 0xFF
        if offset == 0:
            continue

        data.seek(offset * SECTOR_SIZE)
        (length,) = _io.unpack(data, Format.CHUNK_LENGTH)
        (compression,) = _io.unpack(data, Format.CHUNK_COMPRESSION)

        # NOTE: 容量が大き過ぎるチャンクの場合、データは外部の`.mcc`ファイルに書き込まれる。Voxvecでは考慮しない
        if compression & EXTERNAL_FLAG:
            raise NotImplementedError(f"chunk #{idx} is stored in an external file")

        # NOTE: 確保したセクタ領域とチャンクデータに記載された長さの整合性をチェックする
        if Format.CHUNK_LENGTH.size + length > sector_count * SECTOR_SIZE:
            raise ValueError(f"chunk #{idx} overflows its sectors")

        region.chunks[idx] = RegionChunk(
            compression=Compression(compression),
            data=_io.read_exact(data, length - Format.CHUNK_COMPRESSION.size),
            timestamp=timestamps[idx],
        )

    return region


def loads(data: bytes) -> Region:
    return load(BytesIO(data))


def _index(x: int, z: int) -> int:
    if not (0 <= x < REGION_WIDTH and 0 <= z < REGION_WIDTH):
        raise ValueError(f"chunk ({x}, {z}) is out of region")
    return x + z * REGION_WIDTH
