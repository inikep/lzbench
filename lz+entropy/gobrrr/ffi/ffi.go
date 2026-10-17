// C ABI of go-brrr for lzbench, built with mk/gobrrr.mk:
//
//	go build -buildmode=c-archive -o libgobrrr.a ./ffi
//
// lzbench calls gobrrr_init once per thread and level, outside the timed
// loops, and gobrrr_deinit after the level. The state between them is one
// Writer and one Reader, the same as a zstd CCtx and DCtx.
package main

/*
#include <stdint.h>
#include <stddef.h>
*/
import "C"

import (
	"bytes"
	"io"
	"runtime"
	"runtime/cgo"
	"runtime/debug"
	"unsafe"

	brrr "github.com/molecule-man/go-brrr"
)

func main() {}

// At this level and above, a Writer starts worker goroutines when GOMAXPROCS
// is 2 or more. Compress never does. lzbench compares single-threaded codecs.
const minWorkerLevel = 10

type state struct {
	level int
	w     *brrr.Writer // nil at minWorkerLevel and above
	r     *brrr.Reader
	src   bytes.Reader
	dst   fixedBuf
}

// fixedBuf writes into the lzbench output buffer.
type fixedBuf struct {
	b []byte
	n int
}

func (f *fixedBuf) Write(p []byte) (int, error) {
	if f.n+len(p) > len(f.b) {
		return 0, io.ErrShortBuffer
	}
	f.n += copy(f.b[f.n:], p)
	return len(p), nil
}

// liveStates counts active codec states. lzbench calls init and deinit
// from its main thread.
var liveStates int

// gobrrr_init returns a state handle, or 0 if initialization fails.
//
// lzbench creates one state per benchmark thread before the timed loops.
// This function sets GOMAXPROCS to the current state count. Fewer Ps
// reduce Go runtime memory use when lzbench uses fewer threads than CPUs.
//
//export gobrrr_init
func gobrrr_init(level C.int) C.uintptr_t {
	if level < 0 || level > 11 {
		return 0
	}
	liveStates++
	runtime.GOMAXPROCS(liveStates)

	s := &state{level: int(level)}
	if level < minWorkerLevel {
		w, err := brrr.NewWriter(&s.dst, int(level))
		if err != nil {
			liveStates--
			return 0
		}
		s.w = w
	}
	s.r = brrr.NewReader(&s.src)
	return C.uintptr_t(cgo.NewHandle(s))
}

// gobrrr_deinit frees the state and returns the Go heap to the OS, so that
// the codecs that lzbench runs next do not share the process with it.
//
//export gobrrr_deinit
func gobrrr_deinit(h C.uintptr_t) {
	cgo.Handle(h).Delete()
	liveStates--
	debug.FreeOSMemory()
}

// A panic must not cross into C: it would abort the whole lzbench run.
// recoverAsError makes the call return 0, which lzbench reports as an error.
func recoverAsError(n *C.int64_t) {
	if recover() != nil {
		*n = 0
	}
}

// gobrrr_compress returns the compressed size, or 0 on error or when the
// output does not fit into dst.
//
//export gobrrr_compress
func gobrrr_compress(h C.uintptr_t, src *C.uint8_t, srcLen C.size_t, dst *C.uint8_t, dstCap C.size_t) (n C.int64_t) {
	defer recoverAsError(&n)
	s := cgo.Handle(h).Value().(*state)
	in := unsafe.Slice((*byte)(unsafe.Pointer(src)), int(srcLen))
	out := unsafe.Slice((*byte)(unsafe.Pointer(dst)), int(dstCap))

	if s.w == nil {
		c, err := brrr.Compress(in, s.level)
		if err != nil || len(c) > len(out) {
			return 0
		}
		return C.int64_t(copy(out, c))
	}

	s.dst = fixedBuf{b: out}
	// Same hint as C BrotliEncoderCompress: the exact input size.
	s.w.ResetWithSizeHint(&s.dst, uint(srcLen))
	if _, err := s.w.Write(in); err != nil {
		return 0
	}
	if err := s.w.Close(); err != nil {
		return 0
	}
	size := s.dst.n
	s.dst = fixedBuf{} // do not keep the C buffer after the call
	return C.int64_t(size)
}

// gobrrr_decompress returns the decompressed size, or 0 on error or when the
// output does not fit into dst.
//
//export gobrrr_decompress
func gobrrr_decompress(h C.uintptr_t, src *C.uint8_t, srcLen C.size_t, dst *C.uint8_t, dstCap C.size_t) (n C.int64_t) {
	defer recoverAsError(&n)
	s := cgo.Handle(h).Value().(*state)
	out := unsafe.Slice((*byte)(unsafe.Pointer(dst)), int(dstCap))
	s.src.Reset(unsafe.Slice((*byte)(unsafe.Pointer(src)), int(srcLen)))
	defer s.src.Reset(nil)
	s.r.Reset(&s.src)

	got, err := io.ReadFull(s.r, out)
	switch err {
	case io.EOF, io.ErrUnexpectedEOF:
	case nil:
		// dst is full. The stream must end here.
		var one [1]byte
		if m, err := s.r.Read(one[:]); m != 0 || err != io.EOF {
			return 0
		}
	default:
		return 0
	}
	return C.int64_t(got)
}
