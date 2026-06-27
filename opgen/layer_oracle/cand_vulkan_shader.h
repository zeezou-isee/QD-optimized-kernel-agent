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
#include <string>
#include <vector>

#ifndef CANDIDATE_SHADER
#error "CANDIDATE_SHADER (.comp path string literal) must be defined by the oracle compile step"
#endif

namespace ncnn {

static inline int compile_candidate_shader(const Option& opt, std::vector<uint32_t>& spirv)
{
    const char* path = CANDIDATE_SHADER;   // a "..." string literal injected by the oracle

    FILE* fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "candidate shader not found: %s\n", path);
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

    // exact-size overload so we never drop the trailing char
    int ret = compile_spirv_module(src.c_str(), (int)src.size(), opt, spirv);
    if (ret != 0)
        fprintf(stderr, "compile_spirv_module failed (ret=%d) for %s\n", ret, path);
    return ret;
}

} // namespace ncnn

#endif // CAND_VULKAN_SHADER_H
