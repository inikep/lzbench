use std::fs;
use std::time::Instant;
use pulsar::{pulsar_decode, pulsar_encode, version};

fn print_usage() {
    eprintln!("pulsar {}", version());
    eprintln!("Usage:");
    eprintln!("  pulsar encode <input> [-o <output.pulsar>]");
    eprintln!("  pulsar decode <input.pulsar> [-o <output>]");
    eprintln!("  pulsar bench <file>");
    eprintln!("  pulsar version");
}

fn flag_o(args: &[String], default: String) -> String {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
        i += 1;
    }
    default
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }
    match args[1].as_str() {
        "encode" => {
            if args.len() < 3 {
                eprintln!("need input");
                return;
            }
            let input = &args[2];
            let output = flag_o(&args[3..], format!("{}.pulsar", input));
            let data = fs::read(input).expect("read input");
            let start = Instant::now();
            match pulsar_encode(&data) {
                Some(enc) => {
                    let elapsed = start.elapsed();
                    fs::write(&output, &enc).expect("write");
                    println!(
                        "PULSAR {} -> {} raw={} coded={} ratio={:.4} {:.3}s saved={}",
                        input,
                        output,
                        data.len(),
                        enc.len(),
                        enc.len() as f64 / data.len() as f64,
                        elapsed.as_secs_f64(),
                        data.len() - enc.len()
                    );
                }
                None => {
                    eprintln!("not compressible (ratio >= 1.0)");
                    std::process::exit(1);
                }
            }
        }
        "decode" => {
            if args.len() < 3 {
                eprintln!("need input");
                return;
            }
            let input = &args[2];
            let output = flag_o(&args[3..], format!("{}.orig", input));
            let data = fs::read(input).expect("read");
            let start = Instant::now();
            let dec = pulsar_decode(&data).expect("decode failed");
            let elapsed = start.elapsed();
            fs::write(&output, &dec).expect("write");
            println!(
                "decoded {} -> {} {} bytes in {:.3}s",
                input,
                output,
                dec.len(),
                elapsed.as_secs_f64()
            );
        }
        "bench" => {
            if args.len() < 3 {
                eprintln!("need file");
                return;
            }
            let file = &args[2];
            let data = fs::read(file).expect("read bench file");
            println!("bench file {} {} bytes version {}", file, data.len(), version());
            let start = Instant::now();
            match pulsar_encode(&data) {
                Some(enc) => {
                    let enc_time = start.elapsed();
                    println!(
                        "PULSAR: {} -> {} ratio {:.4} encode {:.3}s",
                        data.len(),
                        enc.len(),
                        enc.len() as f64 / data.len() as f64,
                        enc_time.as_secs_f64()
                    );
                    let dec = pulsar_decode(&enc).expect("rt");
                    assert_eq!(dec, data);
                    println!("roundtrip ok");
                }
                None => println!("not compressible by PULSAR (ratio >=1)"),
            }
        }
        "version" => println!("{}", version()),
        _ => print_usage(),
    }
}
