
use crate::match_maker;
pub fn mm_tokens_to_bytes(tokens: &[match_maker::Token]) -> Vec<u8> {
    let mut out=Vec::new();
    for t in tokens {
        match *t {
            match_maker::Token::Lit(b) => { out.push(0x00); out.push(b); }
            match_maker::Token::Match{dist, len} => {
                let mut remain = len;
                while remain > 0 {
                    let l = remain.min(255);
                    if dist<=0xFFFF {
                        out.push(0x01);
                        out.extend_from_slice(&(dist as u16).to_le_bytes());
                        out.push(l as u8);
                    } else {
                        out.push(0x02);
                        out.push((dist & 0xFF) as u8);
                        out.push(((dist>>8)&0xFF) as u8);
                        out.push(((dist>>16)&0xFF) as u8);
                        out.push(l as u8);
                    }
                    remain -= l;
                }
            }
        }
    }
    out
}
pub fn bytes_to_mm_tokens(data: &[u8]) -> Vec<match_maker::Token> {
    let mut tokens=Vec::new(); let mut i=0;
    while i<data.len() {
        match data[i] {
            0x00 => { if i+1>=data.len() { break; } tokens.push(match_maker::Token::Lit(data[i+1])); i+=2; }
            0x01 => { if i+3>=data.len() { break; } let dist=u16::from_le_bytes([data[i+1], data[i+2]]) as usize; let len=data[i+3] as usize; tokens.push(match_maker::Token::Match{dist, len}); i+=4; }
            0x02 => { if i+4>=data.len() { break; } let dist=data[i+1] as usize | ((data[i+2] as usize)<<8) | ((data[i+3] as usize)<<16); let len=data[i+4] as usize; tokens.push(match_maker::Token::Match{dist,len}); i+=5; }
            _ => break,
        }
    }
    tokens
}
