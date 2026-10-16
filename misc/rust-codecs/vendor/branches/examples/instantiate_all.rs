//! Instantiates every public function of the crate.
//!
//! Generic functions are only fully checked (including their inline assembly)
//! when they are monomorphized, so cross-compilation CI builds this example to
//! make sure the architecture-specific code paths actually assemble on every
//! supported target.

use branches::{assume, likely, mark_unlikely, unlikely};

fn main() {
    let flag = std::env::args().count() > 1;
    if likely(!flag) {
        println!("expected path");
    }
    if unlikely(flag) {
        // On 1.95+ `mark_unlikely` is `cold_path`; clippy can't see the cfg.
        #[allow(clippy::incompatible_msrv)]
        mark_unlikely();
        println!("unexpected path");
    }
    unsafe { assume(usize::MAX > 0) };

    // Reference abort() without calling it.
    let _abort: extern "C" fn() -> ! = branches::abort;

    #[cfg(feature = "prefetch")]
    {
        let buf = [0u8; 64];
        let p = buf.as_ptr();
        branches::prefetch_read_data::<_, 0>(p);
        branches::prefetch_read_data::<_, 1>(p);
        branches::prefetch_read_data::<_, 2>(p);
        branches::prefetch_read_data::<_, 3>(p);
        branches::prefetch_read_data::<_, { -1 }>(p);
        branches::prefetch_write_data::<_, 0>(p);
        branches::prefetch_write_data::<_, 1>(p);
        branches::prefetch_write_data::<_, 2>(p);
        branches::prefetch_write_data::<_, 3>(p);
        branches::prefetch_write_data::<_, { -1 }>(p);
    }
}
