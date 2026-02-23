import ctypes
from kanzi_c_api import _lib, _libc, cContext_p, dContext_p, cData, dData


class KanziError(RuntimeError):
    pass


def _check(rc, msg):
    if rc != 0:
        raise KanziError(f"{msg} (error code {rc})")


# -----------------------------------------------------------------------------
# Compressor
# -----------------------------------------------------------------------------

class Compressor:
    def __init__(
        self,
        dst_path,
        transform=b"LZ",
        entropy=b"Huffman",
        block_size=1 << 20,
        jobs=1,
        checksum=0,
        headerless=0,
    ):
        self._file = _libc.fopen(dst_path.encode(), b"wb")
        if not self._file:
            raise OSError("fopen failed")

        self._ctx = cContext_p()

        params = cData()
        params.transform = transform
        params.entropy = entropy
        params.blockSize = block_size
        params.jobs = jobs
        params.checksum = checksum
        params.headerless = headerless

        rc = _lib.initCompressor(
            ctypes.byref(params),
            self._file,
            ctypes.byref(self._ctx),
        )
        _check(rc, "initCompressor failed")

    def compress(self, data: bytes) -> int:
        src = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        out_size = ctypes.c_size_t(0)

        rc = _lib.compress(
            self._ctx,
            src,
            len(data),
            ctypes.byref(out_size),
        )
        _check(rc, "compress failed")
        return out_size.value

    def close(self) -> int:
        out_size = ctypes.c_size_t(0)
        rc = _lib.disposeCompressor(
            ctypes.byref(self._ctx),
            ctypes.byref(out_size),
        )
        _check(rc, "disposeCompressor failed")

        _libc.fclose(self._file)
        self._file = None
        return out_size.value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# -----------------------------------------------------------------------------
# Decompressor
# -----------------------------------------------------------------------------

class Decompressor:
    def __init__(
        self,
        src_path,
        buffer_size,
        jobs=1,
        headerless=0,
        **headerless_params,
    ):
        self._file = _libc.fopen(src_path.encode(), b"rb")
        if not self._file:
            raise OSError("fopen failed")

        self._ctx = dContext_p()

        params = dData()
        params.bufferSize = buffer_size
        params.jobs = jobs
        params.headerless = headerless

        if headerless:
            params.transform = headerless_params["transform"]
            params.entropy = headerless_params["entropy"]
            params.blockSize = headerless_params["blockSize"]
            params.originalSize = headerless_params["originalSize"]
            params.checksum = headerless_params["checksum"]
            params.bsVersion = headerless_params["bsVersion"]

        rc = _lib.initDecompressor(
            ctypes.byref(params),
            self._file,
            ctypes.byref(self._ctx),
        )
        _check(rc, "initDecompressor failed")

    def decompress_block(self, max_output: int) -> bytes:
        dst = (ctypes.c_ubyte * max_output)()
        in_size = ctypes.c_size_t(0)
        out_size = ctypes.c_size_t(max_output)

        rc = _lib.decompress(
            self._ctx,
            dst,
            ctypes.byref(in_size),
            ctypes.byref(out_size),
        )
        _check(rc, "decompress failed")

        return bytes(dst[: out_size.value])

    def close(self):
        rc = _lib.disposeDecompressor(ctypes.byref(self._ctx))
        _check(rc, "disposeDecompressor failed")
        _libc.fclose(self._file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

