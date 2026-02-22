pub struct ProtectionState {
    pub(crate) copy_penalty: u8,
    pub(crate) copy_penalty_start: u8,
    pub(crate) previous_incompressible: bool,
    pub(crate) counter: u64,
}

impl ProtectionState {
    pub fn new() -> Self {
        ProtectionState {
            copy_penalty: 0,
            copy_penalty_start: 1,
            previous_incompressible: false,
            counter: 0,
        }
    }

    #[inline(always)]
    pub fn revert_to_copy(&mut self) -> bool {
        if self.counter & 0xf == 0 {
            if self.copy_penalty_start > 1 {
                self.copy_penalty_start >>= 1;
            }
        }
        self.counter += 1;
        self.copy_penalty > 0
    }

    #[inline(always)]
    pub fn decay(&mut self) {
        self.copy_penalty -= 1;
        if self.copy_penalty == 0 {
            self.copy_penalty_start += 1;
        }
    }

    #[inline(always)]
    pub fn update(&mut self, incompressible: bool) {
        if incompressible {
            if self.previous_incompressible {
                self.copy_penalty = self.copy_penalty_start;
            }
            self.previous_incompressible = true;
        } else {
            self.previous_incompressible = false;
        }
    }
}