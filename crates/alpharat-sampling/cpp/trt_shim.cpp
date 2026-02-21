// trt_shim.cpp — Thin C++ wrapper around TensorRT-RTX API.
//
// Provides extern "C" functions for operations that the trtx Rust crate
// doesn't expose (optimization profiles, setInputShape).
// Compiled by the cc crate when the "tensorrt" feature is active.
//
// Uses dlopen/dlsym for TRT factory functions so the compiled binary has
// no hard link dependency on libtensorrt_rtx.so. TRT .so files must be
// preloaded (via Python ctypes with RTLD_GLOBAL) before calling these.

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

class SimpleLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            const char* level = "?";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
                case Severity::kERROR:          level = "ERROR"; break;
                case Severity::kWARNING:        level = "WARNING"; break;
                default: break;
            }
            fprintf(stderr, "[TensorRT/%s] %s\n", level, msg);
        }
    }
};

static SimpleLogger& get_logger() {
    static SimpleLogger logger;
    return logger;
}

// ---------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------

/// Load TRT-RTX shared libraries via dlopen. Call before any other trt_* function.
/// Returns 0 on success, -1 on failure.
/// Thread-safe via std::call_once (won't retry on failure — Rust side panics anyway).
extern "C" int trt_load_libs(const char* trt_lib_path, const char* parser_lib_path) {
    static std::once_flag flag;
    static int result = 0;

    std::call_once(flag, [&]() {
        void* h1 = dlopen(trt_lib_path ? trt_lib_path : "libtensorrt_rtx.so",
                           RTLD_NOW | RTLD_GLOBAL);
        if (!h1) {
            fprintf(stderr, "[TensorRT] dlopen(libtensorrt_rtx.so): %s\n", dlerror());
            result = -1;
            return;
        }

        void* h2 = dlopen(parser_lib_path ? parser_lib_path : "libtensorrt_onnxparser_rtx.so",
                           RTLD_NOW | RTLD_GLOBAL);
        if (!h2) {
            fprintf(stderr, "[TensorRT] dlopen(libtensorrt_onnxparser_rtx.so): %s\n", dlerror());
            result = -1;
            return;
        }
    });

    return result;
}

// ---------------------------------------------------------------------------
// Factory function resolution via dlsym
// ---------------------------------------------------------------------------
// TRT-RTX libs must be preloaded (via trt_load_libs or Python ctypes) before use.

using CreateBuilderFn = void* (*)(void*, int32_t);
using CreateRuntimeFn = void* (*)(void*, int32_t);
using CreateParserFn  = void* (*)(void*, void*, int);

static CreateBuilderFn resolve_create_builder() {
    auto fn = (CreateBuilderFn)dlsym(RTLD_DEFAULT, "createInferBuilder_INTERNAL");
    if (!fn) fprintf(stderr, "[TensorRT] dlsym(createInferBuilder_INTERNAL) failed: %s\n", dlerror());
    return fn;
}

static CreateRuntimeFn resolve_create_runtime() {
    auto fn = (CreateRuntimeFn)dlsym(RTLD_DEFAULT, "createInferRuntime_INTERNAL");
    if (!fn) fprintf(stderr, "[TensorRT] dlsym(createInferRuntime_INTERNAL) failed: %s\n", dlerror());
    return fn;
}

static CreateParserFn resolve_create_parser() {
    auto fn = (CreateParserFn)dlsym(RTLD_DEFAULT, "createNvOnnxParser_INTERNAL");
    if (!fn) fprintf(stderr, "[TensorRT] dlsym(createNvOnnxParser_INTERNAL) failed: %s\n", dlerror());
    return fn;
}

// Version constants (must match the headers we compiled against)
static constexpr int32_t TRT_VERSION = NV_TENSORRT_VERSION;
static constexpr int32_t ONNX_PARSER_VERSION = NV_ONNX_PARSER_VERSION;

// ---------------------------------------------------------------------------
// TrtSession — opaque handle for Rust
// ---------------------------------------------------------------------------

struct TrtSession {
    IRuntime*          runtime;
    ICudaEngine*       engine;
    IExecutionContext*  context;
};

// ---------------------------------------------------------------------------
// Engine building
// ---------------------------------------------------------------------------

extern "C" int trt_build_engine(
    const void* onnx_data, size_t onnx_len,
    int min_batch, int opt_batch, int max_batch,
    size_t workspace_mb,
    void** out_data, size_t* out_len
) {
    auto& logger = get_logger();

    auto create_builder = resolve_create_builder();
    auto create_parser = resolve_create_parser();
    if (!create_builder || !create_parser) return -1;

    auto* builder = static_cast<IBuilder*>(create_builder(&logger, TRT_VERSION));
    if (!builder) return -1;

    // kEXPLICIT_BATCH is deprecated/ignored in TRT 10+, but pass 0 for compat
    INetworkDefinition* network = builder->createNetworkV2(0);
    if (!network) { delete builder; return -2; }

    auto* parser = static_cast<nvonnxparser::IParser*>(
        create_parser(network, &logger, ONNX_PARSER_VERSION));
    if (!parser) { delete network; delete builder; return -3; }

    if (!parser->parse(onnx_data, onnx_len)) {
        int n = parser->getNbErrors();
        for (int i = 0; i < n; i++) {
            auto* err = parser->getError(i);
            fprintf(stderr, "[TensorRT] ONNX parse error: %s\n", err->desc());
        }
        delete parser; delete network; delete builder;
        return -4;
    }

    // --- Optimization profile for dynamic inputs ---
    // Iterate all network inputs; for any input with a dynamic dimension
    // (value -1), set optimization profile with min/opt/max batch.
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) { delete parser; delete network; delete builder; return -5; }

    int nb_inputs = network->getNbInputs();
    for (int i = 0; i < nb_inputs; i++) {
        ITensor* input = network->getInput(i);
        Dims shape = input->getDimensions();
        const char* name = input->getName();

        // Check if any dimension is dynamic (-1)
        bool has_dynamic = false;
        for (int d = 0; d < shape.nbDims; d++) {
            if (shape.d[d] == -1) { has_dynamic = true; break; }
        }
        if (!has_dynamic) continue;

        // Build min/opt/max dims — replace -1 in dim 0 with batch sizes,
        // keep all other dimensions as-is (they should be static).
        Dims dims_min = shape, dims_opt = shape, dims_max = shape;
        for (int d = 0; d < shape.nbDims; d++) {
            if (shape.d[d] == -1) {
                if (d == 0) {
                    dims_min.d[d] = min_batch;
                    dims_opt.d[d] = opt_batch;
                    dims_max.d[d] = max_batch;
                } else {
                    // Non-batch dynamic dim — unusual, use same value everywhere
                    fprintf(stderr, "[TensorRT] Warning: dynamic dim at axis %d for input '%s'\n", d, name);
                    dims_min.d[d] = 1;
                    dims_opt.d[d] = 1;
                    dims_max.d[d] = 1;
                }
            }
        }

        fprintf(stderr, "[TensorRT] Profile for '%s': min=[", name);
        for (int d = 0; d < shape.nbDims; d++) fprintf(stderr, "%s%ld", d?",":"", dims_min.d[d]);
        fprintf(stderr, "], opt=[");
        for (int d = 0; d < shape.nbDims; d++) fprintf(stderr, "%s%ld", d?",":"", dims_opt.d[d]);
        fprintf(stderr, "], max=[");
        for (int d = 0; d < shape.nbDims; d++) fprintf(stderr, "%s%ld", d?",":"", dims_max.d[d]);
        fprintf(stderr, "]\n");

        if (!profile->setDimensions(name, OptProfileSelector::kMIN, dims_min) ||
            !profile->setDimensions(name, OptProfileSelector::kOPT, dims_opt) ||
            !profile->setDimensions(name, OptProfileSelector::kMAX, dims_max)) {
            fprintf(stderr, "[TensorRT] Failed to set optimization profile for '%s'\n", name);
            delete parser; delete network; delete builder;
            return -6;
        }
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) { delete parser; delete network; delete builder; return -7; }

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace_mb << 20);
    if (config->addOptimizationProfile(profile) < 0) {
        fprintf(stderr, "[TensorRT] Failed to add optimization profile\n");
        delete config; delete parser; delete network; delete builder;
        return -8;
    }

    // Build serialized engine
    IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized || serialized->size() == 0) {
        fprintf(stderr, "[TensorRT] Engine build failed\n");
        delete config; delete parser; delete network; delete builder;
        return -9;
    }

    // Copy to caller-owned buffer (freed by trt_free_buffer)
    *out_len = serialized->size();
    *out_data = malloc(*out_len);
    memcpy(*out_data, serialized->data(), *out_len);

    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;
    return 0;
}

extern "C" void trt_free_buffer(void* data) {
    free(data);
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

extern "C" void* trt_create_session(const void* engine_data, size_t engine_len) {
    auto& logger = get_logger();

    auto create_runtime = resolve_create_runtime();
    if (!create_runtime) return nullptr;

    auto* runtime = static_cast<IRuntime*>(create_runtime(&logger, TRT_VERSION));
    if (!runtime) return nullptr;

    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data, engine_len);
    if (!engine) { delete runtime; return nullptr; }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) { delete engine; delete runtime; return nullptr; }

    auto* session = new TrtSession{runtime, engine, context};
    return static_cast<void*>(session);
}

extern "C" void trt_destroy_session(void* handle) {
    if (!handle) return;
    auto* s = static_cast<TrtSession*>(handle);
    delete s->context;
    delete s->engine;
    delete s->runtime;
    delete s;
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------

extern "C" int trt_set_tensor_address(void* handle, const char* name, void* ptr) {
    auto* s = static_cast<TrtSession*>(handle);
    return s->context->setTensorAddress(name, ptr) ? 0 : -1;
}

extern "C" int trt_set_input_shape(
    void* handle, const char* name, int ndims, const int64_t* shape
) {
    auto* s = static_cast<TrtSession*>(handle);
    Dims dims;
    dims.nbDims = ndims;
    for (int i = 0; i < ndims && i < Dims::MAX_DIMS; i++) {
        dims.d[i] = shape[i];
    }
    return s->context->setInputShape(name, dims) ? 0 : -1;
}

extern "C" int trt_enqueue_v3(void* handle, void* stream) {
    auto* s = static_cast<TrtSession*>(handle);
    return s->context->enqueueV3(static_cast<cudaStream_t>(stream)) ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Version query
// ---------------------------------------------------------------------------

extern "C" int32_t trt_get_version() {
    return NV_TENSORRT_VERSION;
}

// ---------------------------------------------------------------------------
// Engine inspection
// ---------------------------------------------------------------------------

extern "C" int trt_get_nb_io_tensors(void* handle) {
    auto* s = static_cast<TrtSession*>(handle);
    return s->engine->getNbIOTensors();
}

extern "C" const char* trt_get_tensor_name(void* handle, int index) {
    auto* s = static_cast<TrtSession*>(handle);
    return s->engine->getIOTensorName(index);
}
