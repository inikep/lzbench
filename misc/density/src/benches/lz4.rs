use crate::utils::file_path;

mod utils;

fn main() {
    file_path(true);
    divan::main();
}

#[divan::bench_group(sample_count = 25)]
mod default {
    use crate::utils::file_path;
    use divan::counter::BytesCount;
    use divan::Bencher;
    use lz4_flex::{compress_into, decompress_into};
    use std::fs::read;

    #[divan::bench(name = "compress/raw")]
    fn encode_raw(bencher: Bencher) {
        let file_mem = read(file_path(false)).unwrap();
        let mut encoded_mem = vec![0_u8; file_mem.len() << 1];

        print!("\r\t\t\t(\x1b[1m\x1b[34m{:.3}x\x1b[0m)   ", file_mem.len() as f64 / compress_into(&file_mem, &mut encoded_mem).unwrap() as f64);

        bencher
            .counter(BytesCount::of_slice(&file_mem))
            .bench_local(|| { compress_into(&file_mem, &mut encoded_mem) });
    }

    #[divan::bench(name = "decompress/raw             ")]
    fn decode_raw(bencher: Bencher) {
        let file_mem = read(file_path(false)).unwrap();
        let mut encoded_mem = vec![0_u8; file_mem.len() << 1];
        let encoded_size = compress_into(&file_mem, &mut encoded_mem).unwrap();
        let mut decoded_mem = vec![0_u8; file_mem.len() << 1];

        let decoded_size = decompress_into(&encoded_mem[0..encoded_size], &mut decoded_mem).unwrap();
        assert_eq!(file_mem.len(), decoded_size);
        for i in 0..decoded_size {
            assert_eq!(file_mem[i], decoded_mem[i]);
        }

        bencher
            .counter(BytesCount::of_slice(&file_mem))
            .bench_local(|| { decompress_into(&encoded_mem[0..encoded_size], &mut decoded_mem) });
    }
}