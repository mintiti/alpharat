//! Compiler probes for the access-model spike's negative guarantees.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn compile(source: &Path, output: &Path, cfg: Option<&str>) -> Output {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let mut command = Command::new(rustc);
    command
        .arg("--edition=2021")
        .arg("--test")
        .arg(source)
        .arg("-o")
        .arg(output);
    if let Some(cfg) = cfg {
        command.arg("--cfg").arg(cfg);
    }
    command.output().expect("failed to invoke rustc")
}

#[test]
fn lifetime_misuse_is_rejected_by_the_compiler() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest.join("tests/access_model_spike.rs");
    let output_dir =
        std::env::temp_dir().join(format!("alpharat-access-model-ui-{}", std::process::id()));
    fs::create_dir_all(&output_dir).expect("failed to create compiler-probe directory");

    let positive = compile(&source, &output_dir.join("positive"), None);
    assert!(
        positive.status.success(),
        "the positive access model must compile directly:\n{}",
        String::from_utf8_lossy(&positive.stderr)
    );

    for (cfg, expected_diagnostics) in [
        (
            "access_model_fail_view_escape",
            &["does not live long enough", "borrowed value"] as &[&str],
        ),
        (
            "access_model_fail_handle_escape",
            &["lifetime", "borrowed data escapes"],
        ),
        (
            "access_model_fail_wrong_tree",
            &["borrowed data escapes", "invariant"],
        ),
        (
            "access_model_fail_exclusive_handle_escape",
            &["lifetime", "borrowed data escapes"],
        ),
        (
            "access_model_fail_exclusive_view_escape",
            &["does not live long enough", "lifetime may not live long enough"],
        ),
        (
            "access_model_fail_exclusive_alias",
            &["cannot borrow `access` as mutable", "already borrowed"],
        ),
        (
            "access_model_fail_exclusive_wrong_tree",
            &["borrowed data escapes", "invariant"],
        ),
    ] {
        let result = compile(&source, &output_dir.join(cfg), Some(cfg));
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success(),
            "compiler unexpectedly accepted {cfg}"
        );
        assert!(
            expected_diagnostics
                .iter()
                .any(|diagnostic| stderr.contains(diagnostic)),
            "{cfg} failed for an unrelated reason:\n{stderr}"
        );
    }

    let _ = fs::remove_dir_all(output_dir);
}
