
use crate::bwt;
pub fn bwt_big_encode(data: &[u8]) -> (Vec<u8>, usize) { bwt::bwt_encode(data) }
pub fn bwt_big_decode(l: &[u8], p: usize) -> Vec<u8> { bwt::bwt_decode(l,p) }
