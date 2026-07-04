// Shared helper for from-scratch ncnn vulkan candidate layers (方案A, vulkan).
//
// A vulkan candidate authors its compute shader as a SEPARATE .comp file (ncnn
// shader dialect). Instead of referencing the build-time-baked LayerShaderType
// enum (which is unavailable when the candidate is compiled standalone against a
// prebuilt libncnn), the candidate compiles its shader at RUNTIME via ncnn's
// online glslang path. The .comp path is injected by the oracle as the macro
// CANDIDATE_SHADER (mirrors -DCANDIDATE_HEADER / -DCANDIDATE_CLASS).
//
// Usage inside a candidate's create_pipeline(opt):
//     std::vector<uint32_t> spirv;
//     if (compile_candidate_shader(opt, spirv) != 0) return -1;
//     pipeline = new Pipeline(vkdev);
//     pipeline->set_optimal_local_size_xyz();
//     pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations);

#ifndef CAND_VULKAN_SHADER_H
#define CAND_VULKAN_SHADER_H

#include "gpu.h"      // ncnn::compile_spirv_module
#include "option.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

#ifndef CANDIDATE_SHADER
#error "CANDIDATE_SHADER (.comp path string literal) must be defined by the oracle compile step"
#endif

// Optional: directory holding ADDITIONAL .comp shaders. When set, the runner
// has copied every .comp emitted by the LLM into this directory, so the
// candidate can load pipelines beyond the primary CANDIDATE_SHADER by name
// (e.g. BinaryOp has separate broadcast + non-broadcast + pack4 variants).
// Falls back to the CANDIDATE_SHADER dir when the macro is absent.
#ifndef CANDIDATE_SHADER_DIR
#define CANDIDATE_SHADER_DIR ""
#endif

// Structured shader-side exit codes so the driver can differentiate compile vs
// pipeline vs dispatch failures WITHOUT the caller having to grep stderr.
// The runner main() propagates these verbatim as its own exit code, so
// VulkanLayerOracle sees them and can route repair prompts to the right role.
#define RC_SHADER_COMPILE_FAIL 51
#define RC_PIPELINE_CREATE_FAIL 52
#define RC_DISPATCH_FAIL 53

namespace ncnn {

// Sticky flag set from within compile_candidate_shader — the runner reads this
// AFTER the candidate's create_pipeline() returns to disambiguate the two
// failure modes reachable through that single non-zero return:
//   (a) shader source compile failure (glslang could not turn .comp into SPIR-V)
//   (b) pipeline object creation failure (Vulkan rejected the pipeline layout)
// The LLM writes the candidate's create_pipeline; requiring it to propagate a
// distinct error code would leak the RC scheme into every generated cpp. This
// flag keeps the divergence internal to our infra.
inline bool& _cand_shader_compile_failed() {
    static bool v = false;
    return v;
}

// Compile the CANDIDATE_SHADER .comp file to SPIR-V via ncnn's runtime glslang
// path. On failure we emit a MULTI-LINE, self-labeled block on stderr that the
// oracle preserves in run_log.txt — that block is what the debugger prompt
// slices out for shader-only error framing.
//
// The `stage` label ("shader_source_read" / "spirv_compile") lets us tell
// missing-file from a genuine glslang syntax error without parsing.
static inline int compile_candidate_shader(const Option& opt, std::vector<uint32_t>& spirv)
{
    const char* path = CANDIDATE_SHADER;   // a "..." string literal injected by the oracle

    FILE* fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "=== SHADER_COMPILE_FAIL stage=shader_source_read ===\n"
                        "candidate shader not found: %s\n"
                        "=== END_SHADER_COMPILE_FAIL ===\n", path);
        _cand_shader_compile_failed() = true;
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::string src;
    src.resize(n > 0 ? (size_t)n : 0);
    if (n > 0)
    {
        size_t r = fread(&src[0], 1, (size_t)n, fp);
        (void)r;
    }
    fclose(fp);

    // exact-size overload so we never drop the trailing char.
    // compile_spirv_module writes glslang error lines (with `.comp` line
    // numbers) directly to NCNN_LOGE → stderr. Framing them with our
    // fence markers lets the oracle isolate them for the debugger prompt
    // even when other subsystems log to stderr in the same run.
    fprintf(stderr, "=== SHADER_COMPILE stage=spirv_compile path=%s bytes=%zu ===\n",
            path, src.size());
    int ret = compile_spirv_module(src.c_str(), (int)src.size(), opt, spirv);
    if (ret != 0)
    {
        fprintf(stderr, "=== SHADER_COMPILE_FAIL stage=spirv_compile ret=%d path=%s ===\n"
                        "(see glslang errors above; they cite .comp line numbers)\n"
                        "=== END_SHADER_COMPILE_FAIL ===\n", ret, path);
        _cand_shader_compile_failed() = true;
    }
    else
    {
        fprintf(stderr, "=== SHADER_COMPILE_OK stage=spirv_compile spirv_words=%zu ===\n",
                spirv.size());
    }
    return ret;
}

// Load an ADDITIONAL shader by basename (without .comp) from CANDIDATE_SHADER_DIR
// and compile it to SPIR-V at runtime. Use this from a candidate's
// create_pipeline() when the op needs multiple pipelines (e.g. BinaryOp:
// pipeline_binaryop + pipeline_binaryop_broadcast + pipeline_binaryop_pack4).
//
// Naming rule: LLM emits `<name>.comp` in the code_book; the runner copies it
// into CANDIDATE_SHADER_DIR before compilation. Then in create_pipeline:
//   std::vector<uint32_t> spirv;
//   if (compile_candidate_shader_by_name(opt, "binaryop_broadcast", spirv) != 0)
//       return -1;
//   pipeline_broadcast->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs);
//
// Falls back to the CANDIDATE_SHADER file itself when name matches its stem
// (avoids duplicate-load bookkeeping for the primary shader).
static inline int compile_candidate_shader_by_name(const Option& opt, const char* name,
                                                   std::vector<uint32_t>& spirv)
{
    // If no dir is set, only the primary CANDIDATE_SHADER is reachable and we
    // require the caller to use compile_candidate_shader() for it.
    const char* dir = CANDIDATE_SHADER_DIR;
    if (!dir || !dir[0])
    {
        fprintf(stderr, "=== SHADER_COMPILE_FAIL stage=multi_shader_disabled ===\n"
                        "compile_candidate_shader_by_name(\"%s\") but "
                        "CANDIDATE_SHADER_DIR is not set; use compile_candidate_shader() "
                        "or ensure the oracle passed extra_shaders.\n"
                        "=== END_SHADER_COMPILE_FAIL ===\n", name ? name : "(null)");
        _cand_shader_compile_failed() = true;
        return -1;
    }
    std::string path;
    path.reserve(strlen(dir) + strlen(name) + 8);
    path += dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += name;
    path += ".comp";

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "=== SHADER_COMPILE_FAIL stage=shader_source_read ===\n"
                        "additional candidate shader not found: %s\n"
                        "=== END_SHADER_COMPILE_FAIL ===\n", path.c_str());
        _cand_shader_compile_failed() = true;
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::string src;
    src.resize(n > 0 ? (size_t)n : 0);
    if (n > 0)
    {
        size_t r = fread(&src[0], 1, (size_t)n, fp);
        (void)r;
    }
    fclose(fp);

    fprintf(stderr, "=== SHADER_COMPILE stage=spirv_compile path=%s bytes=%zu ===\n",
            path.c_str(), src.size());
    int ret = compile_spirv_module(src.c_str(), (int)src.size(), opt, spirv);
    if (ret != 0)
    {
        fprintf(stderr, "=== SHADER_COMPILE_FAIL stage=spirv_compile ret=%d path=%s ===\n"
                        "(see glslang errors above; they cite .comp line numbers)\n"
                        "=== END_SHADER_COMPILE_FAIL ===\n", ret, path.c_str());
        _cand_shader_compile_failed() = true;
    }
    else
    {
        fprintf(stderr, "=== SHADER_COMPILE_OK stage=spirv_compile spirv_words=%zu ===\n",
                spirv.size());
    }
    return ret;
}

} // namespace ncnn

#endif // CAND_VULKAN_SHADER_H
