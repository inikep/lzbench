pub mod codec;
pub mod algorithms;
pub mod buffer;
pub mod errors;
pub mod io;

pub(crate) const BYTE_SIZE_U16: usize = size_of::<u16>();
pub(crate) const BYTE_SIZE_U32: usize = size_of::<u32>();
pub(crate) const BYTE_SIZE_U128: usize = size_of::<u128>();
pub(crate) const BIT_SIZE_U16: usize = BYTE_SIZE_U16 << 3;
pub(crate) const BIT_SIZE_U32: usize = BYTE_SIZE_U32 << 3;

#[cfg(test)]
mod tests {
    use crate::algorithms::chameleon::chameleon::Chameleon;
    use crate::algorithms::cheetah::cheetah::Cheetah;
    use crate::algorithms::lion::lion::Lion;

    const TEST_DATA: &str = "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttestt";

    #[test]
    fn chameleon() {
        let mut out_mem = vec![0; TEST_DATA.len()];

        // Encoding
        match Chameleon::encode(TEST_DATA.as_bytes(), &mut out_mem) {
            Ok(size) => {
                assert_eq!(&out_mem[0..size], [0xfe, 0xff, 0xff, 0x7f, 0, 0, 0, 0, 116, 101, 115, 116, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 112, 251, 116]);

                // Decoding
                let mut dec_mem = vec![0; TEST_DATA.len()];
                match Chameleon::decode(&out_mem[0..size], &mut dec_mem) {
                    Ok(size) => {
                        assert_eq!(&dec_mem[0..size], TEST_DATA.as_bytes());
                    }
                    Err(_) => { assert!(false); }
                }
            }
            Err(_) => { assert!(false); }
        }
    }

    #[test]
    fn cheetah() {
        let mut out_mem = vec![0; TEST_DATA.len()];

        // Encoding
        match Cheetah::encode(TEST_DATA.as_bytes(), &mut out_mem) {
            Ok(size) => {
                assert_eq!(&out_mem[0..size], [244, 255, 255, 255, 255, 255, 255, 63, 116, 101, 115, 116, 112, 251, 116]);

                // Decoding
                let mut dec_mem = vec![0; TEST_DATA.len()];
                match Cheetah::decode(&out_mem[0..size], &mut dec_mem) {
                    Ok(size) => {
                        assert_eq!(&dec_mem[0..size], TEST_DATA.as_bytes());
                    }
                    Err(_) => { assert!(false); }
                }
            }
            Err(_) => { assert!(false); }
        }
    }

    #[test]
    fn lion() {
        let mut out_mem = vec![0; TEST_DATA.len()];

        // Encoding
        match Lion::encode(TEST_DATA.as_bytes(), &mut out_mem) {
            Ok(size) => {
                assert_eq!(&out_mem[0..size], [112, 146, 36, 73, 146, 36, 116, 101, 115, 116, 112, 251, 73, 146, 36, 73, 146, 4, 116]);

                // Decoding
                let mut dec_mem = vec![0; TEST_DATA.len()];
                match Lion::decode(&out_mem[0..size], &mut dec_mem) {
                    Ok(size) => {
                        assert_eq!(&dec_mem[0..size], TEST_DATA.as_bytes());
                    }
                    Err(_) => { assert!(false); }
                }
            }
            Err(_) => { assert!(false); }
        }
    }
}

