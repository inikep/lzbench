use std::fs;
use std::time::Instant;
use pulsar::{pulsar_decode, pulsar_encode, version};
use pulsar::bwt_ans;
use pulsar::own_lz;

fn magic_of(blob: &[u8]) -> &'static str {
    if blob.len() >= 4 {
        match &blob[..4] {
            b"BW22" => "BW22",
            b"BW23" => "BW23",
            b"OZL2" => "OZL2",
            b"PZ22" => "PZ22",
            _ => "????",
        }
    } else {
        "none"
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: pulsar-bench <file>");
        return;
    }
    let file = &args[1];
    let only_bw = args.get(2).map(|s| s=="bw").unwrap_or(false);
    let data = fs::read(file).expect("read file");
    println!("pulsar {}  file={} raw={}", version(), file, data.len());

    let t0 = Instant::now();
    let bw = bwt_ans::compress(&data);
    let tbw = t0.elapsed();
    match bwt_ans::decompress(&bw) {
        Ok(back) if back == data => println!(
            "BW22: {} ratio {:.4} {:.3}s OK",
            bw.len(),
            bw.len() as f64 / data.len() as f64,
            tbw.as_secs_f64()
        ),
        Ok(_) => println!("BW22: {} BAD rt mismatch", bw.len()),
        Err(e) => println!("BW22: {} DECODE_FAIL {}", bw.len(), e),
    }

    if only_bw { println!("skip ozl2/best"); return; }
    if let Some(oz) = own_lz::own_lz_encode(&data) {
        match own_lz::own_lz_decode(&oz) {
            Ok(back) if back == data => println!("OZL2: {} OK", oz.len()),
            _ => println!("OZL2: {} BAD", oz.len()),
        }
    } else {
        println!("OZL2: none");
    }

    let t1 = Instant::now();
    match pulsar_encode(&data) {
        Some(e) => {
            let t = t1.elapsed();
            println!(
                "BEST: {} {} ratio {:.4} {:.3}s",
                magic_of(&e),
                e.len(),
                e.len() as f64 / data.len() as f64,
                t.as_secs_f64()
            );
            let back = pulsar_decode(&e).expect("decode");
            assert_eq!(back, data);
            println!("BEST_RT ok");
        }
        None => println!("BEST: not compressible"),
    }
}
