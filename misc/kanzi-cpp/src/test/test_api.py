import os
import tempfile

from kanzi import Compressor, Decompressor, KanziError


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def make_params():
    return dict(
        transform=b"LZX",
        entropy=b"HUFFMAN",
        block_size=1024,
        jobs=1,
        checksum=0,
        headerless=0,
    )


def fill_buffer(size):
    return bytes((i * 17 + 3) & 0xFF for i in range(size))


def write_file(path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_init_invalid():
    print("TEST: initCompressor invalid params...")

    # Our Python wrapper validates parameters eagerly,
    # so most invalid cases are raised as Python exceptions.

    try:
        Compressor(
            dst_path="/dev/null",
            transform=None,   # invalid
        )
        assert False, "init should fail on invalid transform"
    except Exception:
        pass


def test_init_dispose():
    print("TEST: init + dispose...")

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        with Compressor(tmp.name, **make_params()) as c:
            assert c is not None


def test_compress_small():
    print("TEST: small compression...")

    data = fill_buffer(256)

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        with Compressor(tmp.name, **make_params()) as c:
            written = c.compress(data)
            assert written >= 0


def test_compress_too_big():
    print("TEST: oversized block handling...")

    params = make_params()
    params["block_size"] = 1024

    big = fill_buffer(4096)

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        with Compressor(tmp.name, **params) as c:
            try:
                c.compress(big)
                assert False, "compress should fail on oversized input"
            except KanziError:
                pass


def test_compress_two_blocks():
    print("TEST: two-block compression...")

    params = make_params()
    params["block_size"] = 1024

    a = fill_buffer(300)
    b = fill_buffer(500)

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        with Compressor(tmp.name, **params) as c:
            out1 = c.compress(a)
            out2 = c.compress(b)
            assert out1 >= 0
            assert out2 >= 0


def test_basic_decompression():
    print("TEST: basic decompression...")

    input_data = b"Hello Kanzi! Hello Compression!"

    with tempfile.NamedTemporaryFile(delete=False) as comp:
        comp_name = comp.name

    # Compress
    with Compressor(
        comp_name,
        transform=b"LZ",
        entropy=b"ANS0",
        block_size=1 << 16,
        jobs=1,
        checksum=32,
        headerless=0,
    ) as c:
        c.compress(input_data)

    # Decompress
    with Decompressor(
        comp_name,
        buffer_size=1 << 16,
        jobs=1,
        headerless=0,
    ) as d:
        out = d.decompress_block(1024)

    assert out == input_data, "decompressed data mismatch"

    os.remove(comp_name)


def test_large_multi_block():
    print("TEST: large multi blocks")

    size = 2 * 1024 * 1024
    data = bytes((i * 7) & 0xFF for i in range(size))

    with tempfile.NamedTemporaryFile(delete=False) as comp:
        comp_name = comp.name

    # Compress
    with Compressor(
        comp_name,
        transform=b"LZ",
        entropy=b"FPAQ",
        block_size=256 * 1024,
        jobs=1,
        checksum=64,
        headerless=0,
    ) as c:
        offset = 0
        while offset < size:
            chunk = min(256 * 1024, size - offset)
            c.compress(data[offset:offset + chunk])
            offset += chunk

    # Decompress
    out = bytearray()

    with Decompressor(
        comp_name,
        buffer_size=256 * 1024,
        jobs=1,
        headerless=0,
    ) as d:
        while True:
            try:
                block = d.decompress_block(256 * 1024)
                if not block:
                    break
                out.extend(block)
            except KanziError:
                break  # expected EOF

    assert bytes(out) == data, "large decompression mismatch"

    os.remove(comp_name)


def test_headerless():
    print("TEST: headerless")

    input_data = b"HEADERLESS MODE IS ACTIVE"

    with tempfile.NamedTemporaryFile(delete=False) as comp:
        comp_name = comp.name

    # Compress (headerless)
    with Compressor(
        comp_name,
        transform=b"LZ",
        entropy=b"ANS0",
        block_size=1 << 15,
        jobs=1,
        checksum=0,
        headerless=1,
    ) as c:
        c.compress(input_data)

    # Decompress (headerless)
    with Decompressor(
        comp_name,
        buffer_size=1 << 15,
        jobs=1,
        headerless=1,
        transform=b"LZ",
        entropy=b"ANS0",
        blockSize=1 << 15,
        originalSize=len(input_data),
        checksum=0,
        bsVersion=1,
    ) as d:
        out = d.decompress_block(256)

    assert out == input_data, "headerless decompression mismatch"

    os.remove(comp_name)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    test_init_invalid()
    test_init_dispose()
    test_compress_small()
    test_compress_too_big()
    test_compress_two_blocks()

    test_basic_decompression()
    test_large_multi_block()
    test_headerless()

    print("All Python API tests passed.")


if __name__ == "__main__":
    main()

