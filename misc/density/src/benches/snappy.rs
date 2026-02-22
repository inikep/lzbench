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
    use snap::read::FrameDecoder;
    use snap::write::FrameEncoder;
    use std::fs::read;
    use std::io;
    use std::io::Write;

    #[divan::bench(name = "compress/stream            ")]
    fn encode_stream(bencher: Bencher) {
        let file_mem = read(file_path(false)).unwrap();
        let capacity = file_mem.len() << 1;
        let mut encoded_mem = Vec::with_capacity(capacity);

        let mut encoder = FrameEncoder::new(&mut encoded_mem);
        io::copy(&mut file_mem.as_slice(), &mut encoder).unwrap();
        encoder.flush().unwrap();

        print!("\r\t\t\t(\x1b[1m\x1b[34m{:.3}x\x1b[0m)   ", file_mem.len() as f64 / encoder.get_ref().len() as f64);

        bencher
            .counter(BytesCount::of_slice(&file_mem))
            .with_inputs(|| { (file_mem.as_slice(), Vec::with_capacity(capacity)) })
            .bench_local_values(|(mut input, mut output)| {
                io::copy(&mut input, &mut FrameEncoder::new(&mut output))
            });
    }

    #[divan::bench(name = "decompress/stream          ")]
    fn decode_raw(bencher: Bencher) {
        let file_mem = read(file_path(false)).unwrap();
        let capacity = file_mem.len() << 1;
        let mut encoded_mem = Vec::with_capacity(capacity);
        let mut decoded_mem = Vec::with_capacity(capacity);

        let encoded_size = {
            let mut encoder = FrameEncoder::new(&mut encoded_mem);
            io::copy(&mut file_mem.as_slice(), &mut encoder).unwrap();
            encoder.flush().unwrap();
            encoder.get_ref().len()
        };

        let decoded_size = {
            let mut decoder = FrameDecoder::new(&encoded_mem[0..encoded_size]);
            io::copy(&mut decoder, &mut decoded_mem).unwrap() as usize
        };

        assert_eq!(file_mem.len(), decoded_size);
        for i in 0..decoded_size {
            assert_eq!(file_mem[i], decoded_mem[i]);
        }

        bencher
            .counter(BytesCount::of_slice(&file_mem))
            .with_inputs(|| { (&encoded_mem[0..encoded_size], Vec::with_capacity(capacity)) })
            .bench_local_values(|(mut input, mut output)| {
                io::copy(&mut FrameDecoder::new(&mut input), &mut output)
            });
    }
}