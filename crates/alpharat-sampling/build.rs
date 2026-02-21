fn main() {
    // When the tensorrt feature is active:
    // 1. Compile the C++ TRT shim (optimization profiles, session management)
    // 2. Link libcudart (raw CUDA FFI) and TRT-RTX shared libs
    #[cfg(feature = "tensorrt")]
    build_tensorrt();
}

#[cfg(feature = "tensorrt")]
fn build_tensorrt() {
    use std::path::Path;
    #[allow(unused_imports)]
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=TENSORRT_RTX_ROOT");
    println!("cargo:rerun-if-changed=cpp/trt_shim.cpp");

    let trt_root = std::env::var("TENSORRT_RTX_ROOT").unwrap_or_else(|_| {
        panic!(
            "TENSORRT_RTX_ROOT must be set to the TensorRT-RTX SDK root \
             (e.g. /path/to/TensorRT-RTX-1.3.0.35)"
        )
    });
    let trt_include = Path::new(&trt_root).join("include");

    // Find CUDA headers (cuda_runtime_api.h) — needed by TRT headers.
    // Search: CUDA_HOME/include, then nvidia pip packages in the venv.
    let cuda_include = find_cuda_include();

    // Compile the C++ shim
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("cpp/trt_shim.cpp")
        .include(&trt_include)
        .warnings(false);
    if let Some(ref inc) = cuda_include {
        build.include(inc);
    }
    build.compile("trt_shim");

    // TRT-RTX libs are loaded via dlopen at runtime (Python preloads them
    // with RTLD_GLOBAL). We only need libcudart linked for raw CUDA FFI.
    link_cudart();
}

#[cfg(feature = "tensorrt")]
fn find_cuda_include() -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};

    // 1. CUDA_HOME/include
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let inc = Path::new(&cuda_home).join("include");
        if inc.join("cuda_runtime_api.h").exists() {
            return Some(inc);
        }
    }

    // 2. nvidia pip packages — check VIRTUAL_ENV, then workspace .venv
    let venv_candidates: Vec<PathBuf> = {
        let mut v = vec![];
        if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
            v.push(PathBuf::from(venv));
        }
        // Also check workspace root's .venv (cargo sets CARGO_MANIFEST_DIR)
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            // manifest is crates/alpharat-sampling, workspace root is ../..
            let ws_root = Path::new(&manifest).parent().and_then(|p| p.parent());
            if let Some(root) = ws_root {
                v.push(root.join(".venv"));
            }
        }
        v
    };

    for venv in &venv_candidates {
        let lib = venv.join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib) {
            for entry in entries.filter_map(|e| e.ok()) {
                let pydir = entry.path();
                if pydir
                    .file_name()
                    .map_or(false, |n| n.to_string_lossy().starts_with("python"))
                {
                    for sub in &[
                        "triton/backends/nvidia/include",
                        "nvidia/cuda_runtime/include",
                        "nvidia/cu13/include",
                    ] {
                        let inc = pydir.join("site-packages").join(sub);
                        // Need crt/host_defines.h too (cuda_runtime_api.h includes it)
                        if inc.join("cuda_runtime_api.h").exists()
                            && inc.join("crt/host_defines.h").exists()
                        {
                            return Some(inc);
                        }
                    }
                }
            }
        }
    }

    // 3. System fallback
    let sys = PathBuf::from("/usr/local/cuda/include");
    if sys.join("cuda_runtime_api.h").exists() {
        return Some(sys);
    }

    None
}

#[cfg(feature = "tensorrt")]
fn link_cudart() {
    use std::path::{Path, PathBuf};

    // 1. Try CUDA_HOME (standard CUDA toolkit installation)
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let lib64 = Path::new(&cuda_home).join("lib64");
        if lib64.join("libcudart.so").exists() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
            println!("cargo:rustc-link-lib=dylib=cudart");
            return;
        }
    }

    // 2. Try to find libcudart from nvidia pip packages in the venv.
    //    These only ship versioned .so files (e.g. libcudart.so.13), so we
    //    create a symlink in OUT_DIR for the linker.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let venv = std::env::var("VIRTUAL_ENV").ok();

    let search_roots: Vec<PathBuf> = match &venv {
        Some(v) => std::fs::read_dir(Path::new(v).join("lib"))
            .into_iter()
            .flat_map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .map_or(false, |n| n.to_string_lossy().starts_with("python"))
            })
            .map(|p| p.join("site-packages/nvidia"))
            .collect(),
        None => vec![],
    };

    let cuda_lib_dirs = ["cu13/lib", "cuda_runtime/lib"];
    for root in &search_roots {
        for sub in &cuda_lib_dirs {
            let dir = root.join(sub);
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("libcudart.so.") && !name_str.ends_with(".a") {
                        // Create unversioned symlink in OUT_DIR
                        let symlink = out_dir.join("libcudart.so");
                        let _ = std::fs::remove_file(&symlink);
                        #[cfg(unix)]
                        std::os::unix::fs::symlink(entry.path(), &symlink).unwrap();
                        println!("cargo:rustc-link-search=native={}", out_dir.display());
                        println!("cargo:rustc-link-lib=dylib=cudart");
                        return;
                    }
                }
            }
        }
    }

    // 3. Fallback: try system default paths
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
