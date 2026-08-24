from enum import IntEnum


class AutoNumber(IntEnum):
    def __new__(cls):
        value = 2 * len(cls.__members__)  # note no + 1
        obj = int.__new__(cls)
        obj._value_ = value
        return obj


class Flag(AutoNumber):
    DECOMPRESSED = ()
    ZIGZAG_DCT_1D = ()
