use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct DecodeError {}

impl Display for DecodeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Decode error")
    }
}

impl Error for DecodeError {}