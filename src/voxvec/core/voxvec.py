from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class VoxvecHeader:
    MAGIC: ClassVar[bytes] = b"VOXVEC"

    version: int

    def to_bytes(self):
        pass

    @classmethod
    def from_bytes(self):
        pass


@dataclass(frozen=True)
class VoxvecCodebook:
    def __init__(self):
        pass


class Voxvec:
    def __init__(self):
        pass

    def init(self):
        pass
