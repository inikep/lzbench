use std::env;
use std::process::Command;
use std::str;

// rustc releases gating version-dependent code paths. Keep in sync with the
// cfg names used in src/lib.rs and the check-cfg list in Cargo.toml.
// 1.54: macro invocations in attribute values (`doc = include_str!(...)`).
// 1.57: `unreachable_unchecked` callable in const fn (`assume` emits a hint).
// 1.59: inline assembly on x86/x86_64, aarch64 and riscv64.
// 1.81: `core::hint::assert_unchecked`.
// 1.84: inline assembly on s390x.
// 1.95: inline assembly on powerpc/powerpc64, `core::hint::cold_path`.
const VERSION_CFGS: &[(u64, u64, &str)] = &[
    (1, 54, "rustc_ge_1_54_0"),
    (1, 57, "rustc_ge_1_57_0"),
    (1, 59, "rustc_ge_1_59_0"),
    (1, 81, "rustc_ge_1_81_0"),
    (1, 84, "rustc_ge_1_84_0"),
    (1, 95, "rustc_ge_1_95_0"),
];

fn main() {
    match rustc_release() {
        Some(release) => {
            if release.is_nightly() {
                println!("cargo:rustc-cfg=branches_nightly");
            } else {
                println!("cargo:rustc-cfg=branches_stable");
            }
            for &(major, minor, cfg) in VERSION_CFGS {
                if release.is_at_least(major, minor) {
                    println!("cargo:rustc-cfg={}", cfg);
                }
            }
            // rustc 1.84+ (LLVM 19) at -O2/-O3 converts calls to `#[cold]`
            // functions into `!prof` branch weights while inlining them, so
            // the empty hint helpers may drop `#[inline(never)]` there.
            // Other opt levels record no weights and need the call kept out
            // of line. Measured per stable release from 1.84 through 1.94.
            let opt_level = env::var("OPT_LEVEL").unwrap_or_default();
            if release.is_at_least(1, 84) && (opt_level == "2" || opt_level == "3") {
                println!("cargo:rustc-cfg=branches_cold_weights");
            }
        }
        // If the compiler version cannot be detected, emitting no cfg at all
        // would make the crate fail to compile with confusing type errors.
        // Fall back to the most conservative stable code path instead.
        None => println!("cargo:rustc-cfg=branches_stable"),
    }
}

struct Release {
    major: u64,
    minor: u64,
    patch: u64,
    /// Pre-release suffix, e.g. `nightly` or `beta.5`; empty on releases.
    pre: String,
}

impl Release {
    // `dev` (a compiler built from source) stays on the stable code path.
    fn is_nightly(&self) -> bool {
        self.pre.starts_with("nightly")
    }

    // Prerelease sorts before the release: 1.84.0-nightly < 1.84.0.
    fn is_at_least(&self, major: u64, minor: u64) -> bool {
        if (self.major, self.minor, self.patch) == (major, minor, 0) {
            self.pre.is_empty()
        } else {
            (self.major, self.minor, self.patch) > (major, minor, 0)
        }
    }
}

// Hand-rolled instead of the `rustc_version` crate: its `semver` dependency
// ships manifests that cargo older than 1.60 cannot parse, which would
// defeat the 1.51 MSRV.
fn rustc_release() -> Option<Release> {
    // Use the compiler cargo selected, via RUSTC_WRAPPER (sccache) when set.
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let wrapper = env::var_os("RUSTC_WRAPPER").filter(|w| !w.is_empty());
    let mut cmd = match wrapper {
        Some(wrapper) => {
            let mut cmd = Command::new(wrapper);
            cmd.arg(rustc);
            cmd
        }
        None => Command::new(rustc),
    };
    let output = cmd.arg("-vV").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = str::from_utf8(&output.stdout).ok()?;
    let release = stdout
        .lines()
        .find_map(|line| line.strip_prefix("release: "))?;
    parse_release(release.trim())
}

// `1.84.0`, `1.100.0-nightly` or `1.84.0-beta.5`, optionally followed by
// `+` build metadata on vendored compilers.
fn parse_release(release: &str) -> Option<Release> {
    let release = release.split('+').next()?;
    let mut parts = release.splitn(2, '-');
    let triple = parts.next()?;
    let pre = parts.next().unwrap_or("").to_string();
    let mut numbers = triple.split('.');
    let major = numbers.next()?.parse().ok()?;
    let minor = numbers.next()?.parse().ok()?;
    let patch = numbers.next()?.parse().ok()?;
    if numbers.next().is_some() {
        return None;
    }
    Some(Release {
        major,
        minor,
        patch,
        pre,
    })
}
