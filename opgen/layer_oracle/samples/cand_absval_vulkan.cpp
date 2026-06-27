#include "cand_absval_vulkan.h"

#include "cand_vulkan_shader.h"   // compile_candidate_shader() — reads CANDIDATE_SHADER, online-compiles

#include <vector>

namespace ncnn {

Cand_AbsVal_vulkan::Cand_AbsVal_vulkan()
{
    support_vulkan = true;   // MUST be set — else the oracle refuses (no CPU fallback)
    support_inplace = true;
    one_blob_only = true;
    pipeline_absval = 0;
}

int Cand_AbsVal_vulkan::create_pipeline(const Option& opt)
{
    std::vector<uint32_t> spirv;
    if (compile_candidate_shader(opt, spirv) != 0)
        return -1;

    // The shader declares one specialization constant (constant_id = 0). We must
    // supply exactly that many; set it to 0 so psc(n) falls through to the
    // push-constant value at dispatch time (dynamic element count).
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = 0;

    // 1D dispatch -> 1D workgroup (matches dispatcher.w=n, h=1, c=1). Using the
    // default (4,4,4) here would make the shader's workgroup 3D and mismatch the
    // 1D dispatch, leaving most elements unprocessed.
    pipeline_absval = new Pipeline(vkdev);
    pipeline_absval->set_optimal_local_size_xyz(vkdev->info.subgroup_size(), 1, 1);
    return pipeline_absval->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations);
}

int Cand_AbsVal_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_absval;
    pipeline_absval = 0;
    return 0;
}

int Cand_AbsVal_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    // v1: the oracle forces elempack=1, so total() == scalar element count.
    const int n = (int)bottom_top_blob.total();

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(1);
    constants[0].i = n;

    VkMat dispatcher;
    dispatcher.w = n;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_absval, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
