// Hand-written ncnn VULKAN candidate layer: AbsVal (elementwise |x|).
//
// Sample for the vulkan layer oracle (方案A, vulkan). It is a self-contained
// ncnn::Layer subclass (not the built-in AbsVal) — the candidate authors its own
// compute shader in a separate .comp file and compiles it at runtime.

#ifndef CAND_ABSVAL_VULKAN_H
#define CAND_ABSVAL_VULKAN_H

#include "layer.h"
#include "pipeline.h"

namespace ncnn {

class Cand_AbsVal_vulkan : public Layer
{
public:
    Cand_AbsVal_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Layer::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_absval;
};

} // namespace ncnn

#endif // CAND_ABSVAL_VULKAN_H
