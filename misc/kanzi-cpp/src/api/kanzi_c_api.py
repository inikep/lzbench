import ctypes
import sys
import os

# -----------------------------------------------------------------------------
# Platform detection
# -----------------------------------------------------------------------------

if sys.platform.startswith("win"):
    KANZI_LIB_NAME = "kanzi.dll"
    LIBC_NAME = "msvcrt.dll"
elif sys.platform == "darwin":
    KANZI_LIB_NAME = "libkanzi.dylib"
    LIBC_NAME = "libc.dylib"
else:
    KANZI_LIB_NAME = "libkanzi.so"
    LIBC_NAME = "libc.so.6"

# -----------------------------------------------------------------------------
# Load shared libraries
# -----------------------------------------------------------------------------

_lib = ctypes.CDLL(KANZI_LIB_NAME)
_libc = ctypes.CDLL(LIBC_NAME)

# -----------------------------------------------------------------------------
# libc FILE*
# -----------------------------------------------------------------------------

FILE_p = ctypes.c_void_p

_libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_libc.fopen.restype = FILE_p

_libc.fclose.argtypes = [FILE_p]
_libc.fclose.restype = ctypes.c_int

# -----------------------------------------------------------------------------
# Opaque contexts
# -----------------------------------------------------------------------------

class cContext(ctypes.Structure):
    pass


class dContext(ctypes.Structure):
    pass


cContext_p = ctypes.POINTER(cContext)
dContext_p = ctypes.POINTER(dContext)

# -----------------------------------------------------------------------------
# Compression parameters
# -----------------------------------------------------------------------------

class cData(ctypes.Structure):
    _fields_ = [
        ("transform", ctypes.c_char * 64),
        ("entropy", ctypes.c_char * 16),
        ("blockSize", ctypes.c_size_t),
        ("jobs", ctypes.c_uint),
        ("checksum", ctypes.c_int),
        ("headerless", ctypes.c_int),
    ]


class dData(ctypes.Structure):
    _fields_ = [
        # Required
        ("bufferSize", ctypes.c_size_t),
        ("jobs", ctypes.c_uint),
        ("headerless", ctypes.c_int),

        # Headerless-only
        ("transform", ctypes.c_char * 64),
        ("entropy", ctypes.c_char * 16),
        ("blockSize", ctypes.c_uint),
        ("originalSize", ctypes.c_size_t),
        ("checksum", ctypes.c_int),
        ("bsVersion", ctypes.c_int),
    ]

# -----------------------------------------------------------------------------
# Function prototypes - Compressor
# -----------------------------------------------------------------------------

_lib.getCompressorVersion.argtypes = []
_lib.getCompressorVersion.restype = ctypes.c_uint

_lib.initCompressor.argtypes = [
    ctypes.POINTER(cData),
    FILE_p,
    ctypes.POINTER(cContext_p),
]
_lib.initCompressor.restype = ctypes.c_int

_lib.compress.argtypes = [
    cContext_p,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.compress.restype = ctypes.c_int

_lib.disposeCompressor.argtypes = [
    ctypes.POINTER(cContext_p),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.disposeCompressor.restype = ctypes.c_int

# -----------------------------------------------------------------------------
# Function prototypes - Decompressor
# -----------------------------------------------------------------------------

_lib.getDecompressorVersion.argtypes = []
_lib.getDecompressorVersion.restype = ctypes.c_uint

_lib.initDecompressor.argtypes = [
    ctypes.POINTER(dData),
    FILE_p,
    ctypes.POINTER(dContext_p),
]
_lib.initDecompressor.restype = ctypes.c_int

_lib.decompress.argtypes = [
    dContext_p,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.decompress.restype = ctypes.c_int

_lib.disposeDecompressor.argtypes = [
    ctypes.POINTER(dContext_p),
]
_lib.disposeDecompressor.restype = ctypes.c_int

# -----------------------------------------------------------------------------
# Optional helpers (recommended for kanzi.py)
# -----------------------------------------------------------------------------

def fopen(path: bytes, mode: bytes) -> FILE_p:
    return _libc.fopen(path, mode)


def fclose(fp: FILE_p) -> int:
    return _libc.fclose(fp)


# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------

__all__ = [
    # libraries
    "_lib",
    "_libc",

    # FILE*
    "FILE_p",
    "fopen",
    "fclose",

    # contexts
    "cContext",
    "dContext",
    "cContext_p",
    "dContext_p",

    # params
    "cData",
    "dData",

    # functions (via _lib)
]

