from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO, Reader, Writer
from struct import Struct
from types import MappingProxyType
from typing import ClassVar, Self, override

import numpy as np

__all__ = [
    "ByteArrayTag",
    "ByteTag",
    "CompoundTag",
    "DoubleTag",
    "FloatTag",
    "IntArrayTag",
    "IntTag",
    "ListTag",
    "LongArrayTag",
    "LongTag",
    "ShortTag",
    "StringTag",
    "Tag",
    "TagType",
    "dump",
    "dumps",
    "load",
    "loads",
]


class Format:
    BYTE = Struct(">b")
    SHORT = Struct(">h")
    USHORT = Struct(">H")
    INT = Struct(">i")
    LONG = Struct(">q")
    FLOAT = Struct(">f")
    DOUBLE = Struct(">d")


class TagType(IntEnum):
    END = 0
    BYTE = 1
    SHORT = 2
    INT = 3
    LONG = 4
    FLOAT = 5
    DOUBLE = 6
    BYTE_ARRAY = 7
    STRING = 8
    LIST = 9
    COMPOUND = 10
    INT_ARRAY = 11
    LONG_ARRAY = 12


@dataclass
class Tag(ABC):
    TYPE: ClassVar[TagType]

    @classmethod
    @abstractmethod
    def read(cls, f: Reader[bytes]) -> Self: ...

    @abstractmethod
    def write(self, f: Writer[bytes]) -> None: ...


@dataclass
class ScalarTag[T: (int, float)](Tag):
    FORMAT: ClassVar[Struct]

    value: T

    @override
    @classmethod
    def read(cls, f: Reader[bytes]):
        (value,) = _unpack(f, cls.FORMAT)
        return cls(value)

    @override
    def write(self, f: Writer[bytes]):
        f.write(self.FORMAT.pack(self.value))


class ByteTag(ScalarTag[int]):
    TYPE = TagType.BYTE
    FORMAT = Format.BYTE


class ShortTag(ScalarTag[int]):
    TYPE = TagType.SHORT
    FORMAT = Format.SHORT


class IntTag(ScalarTag[int]):
    TYPE = TagType.INT
    FORMAT = Format.INT


class LongTag(ScalarTag[int]):
    TYPE = TagType.LONG
    FORMAT = Format.LONG


class FloatTag(ScalarTag[float]):
    TYPE = TagType.FLOAT
    FORMAT = Format.FLOAT


class DoubleTag(ScalarTag[float]):
    TYPE = TagType.DOUBLE
    FORMAT = Format.DOUBLE


@dataclass
class ArrayTag(Tag):
    DTYPE: ClassVar[np.dtype]

    value: np.ndarray

    @override
    @classmethod
    def read(cls, f: Reader[bytes]):
        (length,) = _unpack(f, Format.INT)
        data = _read_exact(f, length * cls.DTYPE.itemsize)
        return cls(np.frombuffer(data, dtype=cls.DTYPE))

    @override
    def write(self, f: Writer[bytes]):
        array = np.asarray(self.value, dtype=self.DTYPE)
        f.write(Format.INT.pack(len(array)))
        f.write(array.tobytes())


class ByteArrayTag(ArrayTag):
    TYPE = TagType.BYTE_ARRAY
    DTYPE = np.dtype(">i1")


class IntArrayTag(ArrayTag):
    TYPE = TagType.INT_ARRAY
    DTYPE = np.dtype(">i4")


class LongArrayTag(ArrayTag):
    TYPE = TagType.LONG_ARRAY
    DTYPE = np.dtype(">i8")


@dataclass
class StringTag(Tag):
    TYPE = TagType.STRING

    value: str

    @override
    @classmethod
    def read(cls, f: Reader[bytes]):
        return cls(_read_string(f))

    @override
    def write(self, f: Writer[bytes]):
        _write_string(f, self.value)


@dataclass
class ListTag(Tag):
    TYPE = TagType.LIST

    item_type: TagType
    items: list[Tag]

    def __post_init__(self):
        self._check_items()

    def __getitem__(self, index: int):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    @override
    @classmethod
    def read(cls, f: Reader[bytes]):
        item_type = _read_type(f)
        (length,) = _unpack(f, Format.INT)
        return cls(item_type, [TAG_BY_TYPE[item_type].read(f) for _ in range(length)])

    @override
    def write(self, f: Writer[bytes]):
        self._check_items()
        f.write(bytes([self.item_type]))
        f.write(Format.INT.pack(len(self.items)))
        for item in self.items:
            item.write(f)

    def _check_items(self):
        for item in self.items:
            if item.TYPE != self.item_type:
                raise TypeError(
                    f"{type(item).__name__} in ListTag of {self.item_type.name}"
                )


@dataclass
class CompoundTag(Tag):
    TYPE = TagType.COMPOUND

    value: dict[str, Tag]

    def __getitem__(self, key: str):
        return self.value[key]

    def get[T: Tag](self, key: str, tag_class: type[T]):
        tag = self.value[key]
        if not isinstance(tag, tag_class):
            raise TypeError(
                f"{key!r} is {type(tag).__name__}, not {tag_class.__name__}"
            )
        return tag

    @override
    @classmethod
    def read(cls, f: Reader[bytes]):
        value = {}
        while (tag_type := _read_type(f)) != TagType.END:
            name = _read_string(f)
            value[name] = TAG_BY_TYPE[tag_type].read(f)
        return cls(value)

    @override
    def write(self, f: Writer[bytes]):
        for name, tag in self.value.items():
            f.write(bytes([tag.TYPE]))
            _write_string(f, name)
            tag.write(f)
        f.write(bytes([TagType.END]))


TAGS = [
    ByteTag,
    ShortTag,
    IntTag,
    LongTag,
    FloatTag,
    DoubleTag,
    ByteArrayTag,
    IntArrayTag,
    LongArrayTag,
    StringTag,
    ListTag,
    CompoundTag,
]
TAG_BY_TYPE = MappingProxyType({t.TYPE: t for t in TAGS})


def load(f: Reader[bytes]) -> CompoundTag:
    if (tag_type := _read_type(f)) != TagType.COMPOUND:
        raise ValueError(f"root must be COMPOUND: {tag_type!r}")
    _read_string(f)
    return CompoundTag.read(f)


def loads(data: bytes) -> CompoundTag:
    return load(BytesIO(data))


def dump(tag: CompoundTag, f: Writer[bytes]) -> None:
    f.write(bytes([TagType.COMPOUND]))
    _write_string(f, "")
    tag.write(f)


def dumps(tag: CompoundTag) -> bytes:
    f = BytesIO()
    dump(tag, f)
    return f.getvalue()


def _read_exact(f: Reader[bytes], size: int):
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    return data


def _unpack(f: Reader[bytes], fmt: Struct):
    return fmt.unpack(_read_exact(f, fmt.size))


def _read_type(f: Reader[bytes]):
    return TagType(_read_exact(f, 1)[0])


def _read_string(f: Reader[bytes]):
    (length,) = _unpack(f, Format.USHORT)
    return _read_exact(f, length).decode("utf-8")


def _write_string(f: Writer[bytes], value: str):
    data = value.encode("utf-8")
    f.write(Format.USHORT.pack(len(data)))
    f.write(data)
