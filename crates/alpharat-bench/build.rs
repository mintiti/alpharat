use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const EMBEDDED_PATCH_FILE: &str = "alpharat-build-source.patch";

fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo must provide CARGO_MANIFEST_DIR"),
    );
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("alpharat-bench must live under <repo>/crates/<package>");

    track_build_inputs(repo_root);
    emit_build_identity();
    let mut features = env::vars()
        .filter(|(k, _)| k.starts_with("CARGO_FEATURE_"))
        .map(|(k, _)| k)
        .collect::<Vec<_>>();
    features.sort();
    emit_env("ALPHARAT_INFER_FEATURES", &features.join(","));
    let flags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    emit_env("ALPHARAT_INFER_RUSTFLAGS", &flags.replace('\x1f', " "));
    // sha256sum is only a provenance helper; unavailable remains explicit.
    let lock = command_output("sha256sum", [repo_root.join("Cargo.lock")])
        .ok()
        .and_then(|s| s.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "unavailable".into());
    emit_env("ALPHARAT_INFER_LOCK_SHA256", &lock);
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let provenance = capture_source(repo_root);
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    fs::write(out_dir.join(EMBEDDED_PATCH_FILE), &provenance.patch)
        .expect("failed to materialize embedded source patch");
    emit_env("ALPHARAT_BUILD_SOURCE_REVISION", &provenance.revision);
    emit_env("ALPHARAT_BUILD_SOURCE_STATE", provenance.state);
    emit_env("ALPHARAT_BUILD_SOURCE_REASON", &provenance.reason);
}

fn emit_build_identity() {
    let rustc = env::var_os("RUSTC")
        .and_then(|program| command_output(program, ["--version"]).ok())
        .unwrap_or_else(|| "unknown rustc".to_owned());
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_owned());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown-profile".to_owned());

    emit_env("ALPHARAT_BUILD_RUSTC_VERSION", &rustc);
    emit_env("ALPHARAT_BUILD_TARGET", &target);
    emit_env("ALPHARAT_BUILD_PROFILE", &profile);
}

struct BuildProvenance {
    revision: String,
    state: &'static str,
    reason: String,
    patch: Vec<u8>,
}

fn capture_source(repo_root: &Path) -> BuildProvenance {
    let revision = match git_output(repo_root, ["rev-parse", "HEAD"]) {
        Ok(revision) => revision,
        Err(error) => {
            return BuildProvenance {
                revision: "unknown".to_owned(),
                state: "non_reproducible",
                reason: format!("could not identify build source revision: {error}"),
                patch: Vec::new(),
            }
        }
    };
    let status = match git_output(repo_root, ["status", "--porcelain=v1"]) {
        Ok(status) => status,
        Err(error) => {
            return BuildProvenance {
                revision,
                state: "non_reproducible",
                reason: format!("could not inspect build source worktree: {error}"),
                patch: Vec::new(),
            }
        }
    };
    if status.trim().is_empty() {
        return BuildProvenance {
            revision,
            state: "clean",
            reason: String::new(),
            patch: Vec::new(),
        };
    }

    let untracked = status
        .lines()
        .filter(|line| line.starts_with("?? "))
        .count();
    if untracked > 0 {
        return BuildProvenance {
            revision,
            state: "non_reproducible",
            reason: format!(
                "build source contained {untracked} untracked path(s); commit or remove them before a comparable build"
            ),
            patch: Vec::new(),
        };
    }

    match git_output_bytes(repo_root, ["diff", "--binary", "HEAD"]) {
        Ok(patch) if !patch.is_empty() => BuildProvenance {
            revision,
            state: "patched",
            reason: String::new(),
            patch,
        },
        Ok(_) => BuildProvenance {
            revision,
            state: "non_reproducible",
            reason: "build source was dirty but produced no reconstructable patch".to_owned(),
            patch: Vec::new(),
        },
        Err(error) => BuildProvenance {
            revision,
            state: "non_reproducible",
            reason: format!("could not capture build source patch: {error}"),
            patch: Vec::new(),
        },
    }
}

fn track_build_inputs(repo_root: &Path) {
    if let Ok(files) = git_output_bytes(repo_root, ["ls-files", "-z"]) {
        for file in files
            .split(|byte| *byte == 0)
            .filter(|file| !file.is_empty())
        {
            let relative = String::from_utf8_lossy(file);
            println!(
                "cargo:rerun-if-changed={}",
                repo_root.join(relative.as_ref()).display()
            );
        }
    }

    for git_name in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = resolved_git_path(repo_root, git_name) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    if let Ok(reference) = git_output(repo_root, ["symbolic-ref", "HEAD"]) {
        if let Some(path) = resolved_git_path(repo_root, &reference) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn resolved_git_path(repo_root: &Path, name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(git_output(repo_root, ["rev-parse", "--git-path", name]).ok()?);
    Some(if path.is_absolute() {
        path
    } else {
        repo_root.join(path)
    })
}

fn emit_env(name: &str, value: &str) {
    let value = value.replace(['\r', '\n'], " ");
    println!("cargo:rustc-env={name}={value}");
}

fn command_output(
    program: impl AsRef<OsStr>,
    args: impl IntoIterator<Item = impl AsRef<OsStr>>,
) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to start command: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "command exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn git_output<const N: usize>(repo_root: &Path, args: [&str; N]) -> Result<String, String> {
    let output = git_command(repo_root, args)
        .output()
        .map_err(|error| format!("failed to run git in '{}': {error}", repo_root.display()))?;
    checked_git_stdout(output).map(|bytes| String::from_utf8_lossy(&bytes).trim().to_owned())
}

fn git_output_bytes<const N: usize>(repo_root: &Path, args: [&str; N]) -> Result<Vec<u8>, String> {
    let output = git_command(repo_root, args)
        .output()
        .map_err(|error| format!("failed to run git in '{}': {error}", repo_root.display()))?;
    checked_git_stdout(output)
}

fn git_command<const N: usize>(repo_root: &Path, args: [&str; N]) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(repo_root).args(args);
    command
}

fn checked_git_stdout(output: std::process::Output) -> Result<Vec<u8>, String> {
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(format!(
            "git exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}
