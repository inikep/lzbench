use std::env;
use std::fs::metadata;

const DEFAULT_FILE_PATH: &str = "./benches/data/dickens.txt";

pub fn file_path(display: bool) -> String {
    let file_path = match env::var("FILE") {
        Ok(file_path) => { file_path }
        Err(_) => { String::from(DEFAULT_FILE_PATH) }
    };

    if display {
        let size = metadata(&file_path).unwrap().len();
        println!("Using file \x1b[1m{}\x1b[0m (\x1b[37m{} bytes\x1b[0m)", &file_path, size);
    }

    file_path
}