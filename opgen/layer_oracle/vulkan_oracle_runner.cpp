// Generic ncnn VULKAN layer "oracle" runner (方案A, vulkan).
//
// Compiled ONCE per candidate via -D macros together with the candidate vulkan
// layer .cpp, linked against a vulkan-enabled libncnn. It instantiates the
// candidate Layer DIRECTLY (new CANDIDATE_CLASS), uploads inputs to the GPU, runs
// the vulkan forward (VkMat/VkCompute), downloads the result, and writes it —
// NEVER going through ncnn::create_layer / Layer_final, so there is no silent CPU
// fallback that could mask a broken vulkan kernel (isolated instantiation, same
// philosophy as layer_oracle_runner.cpp for base/arm).
//
// Build (handled by VulkanLayerOracle via a generated CMake find_package(ncnn)):
//   -DCANDIDATE_HEADER='"cand_xxx_vulkan.h"' -DCANDIDATE_CLASS=Cand_Xxx_vulkan
//   -DCANDIDATE_SHADER='/abs/path/cand_xxx.comp'
//
// Run (same CLI + bin protocol as the base/arm runner):
//   runner --param "0=..." --input in0.bin [--input in1.bin] \
//          --weight w0.bin --out out.bin [--packing 4]
//
// Exit codes: 0 ok; 42 = no vulkan device (oracle treats as SKIPPED, not fail);
//             others = real failure.

#include "layer.h"
#include "mat.h"
#include "modelbin.h"
#include "paramdict.h"
#include "option.h"
#include "gpu.h"
#include "command.h"

#include CANDIDATE_HEADER

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ncnn;

#define RC_NO_VULKAN_DEVICE 42

// ---- bin protocol: [int ndim][int dims...][float data]  (matches MoKA / base runner) ----
static Mat read_mat(const char* path)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return Mat(); }
    int ndim = 0;
    if (fread(&ndim, sizeof(int), 1, fp) != 1) { fclose(fp); return Mat(); }
    std::vector<int> dims(ndim);
    if (ndim > 0) { size_t r = fread(dims.data(), sizeof(int), ndim, fp); (void)r; }

    Mat m;
    if (ndim == 1) m.create(dims[0]);
    else if (ndim == 2) m.create(dims[1], dims[0]);
    else if (ndim == 3) m.create(dims[2], dims[1], dims[0]);
    else if (ndim == 4) m.create(dims[3], dims[2], dims[1], dims[0]);
    else { fclose(fp); fprintf(stderr, "bad ndim %d\n", ndim); return Mat(); }

    for (int q = 0; q < m.c; q++)
    {
        float* ptr = m.channel(q);
        int per = m.w * m.h * m.d;
        size_t r = fread(ptr, sizeof(float), per, fp); (void)r;
    }
    fclose(fp);
    return m;
}

static void write_mat(const Mat& m, const char* path)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", path); return; }
    int ndim = m.dims;
    std::vector<int> dims;
    if (ndim == 1) dims = {m.w};
    else if (ndim == 2) dims = {m.h, m.w};
    else if (ndim == 3) dims = {m.c, m.h, m.w};
    else dims = {m.c, m.d, m.h, m.w};
    fwrite(&ndim, sizeof(int), 1, fp);
    fwrite(dims.data(), sizeof(int), ndim, fp);
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        int per = m.w * m.h * m.d;
        fwrite(ptr, sizeof(float), per, fp);
    }
    fclose(fp);
}

static Mat read_weight(const char* path)
{
    Mat m = read_mat(path);
    Mat flat = m.reshape(m.w * m.h * m.d * m.c);
    return flat.clone();
}

class ModelBinFromMatArrayStrict : public ModelBin
{
public:
    ModelBinFromMatArrayStrict(const Mat* _weights) : weights_(_weights) {}
    Mat load(int w, int /*type*/) const override
    {
        if (!weights_) return Mat();
        Mat m = weights_[index_];
        int actual = m.w * m.h * m.d * m.c;
        if (actual != w)
        {
            fprintf(stderr, "load_model size mismatch: requested %d, actual %d (weight index %d)\n",
                    w, actual, index_);
            return Mat();
        }
        index_++;
        return m;
    }
private:
    const Mat* weights_;
    mutable int index_ = 0;
};

static void parse_params(const std::string& s, ParamDict& pd)
{
    size_t i = 0;
    while (i < s.size())
    {
        size_t comma = s.find(',', i);
        std::string tok = s.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        i = (comma == std::string::npos) ? s.size() : comma + 1;
        size_t eq = tok.find('=');
        if (eq == std::string::npos) continue;
        int id = atoi(tok.substr(0, eq).c_str());
        std::string v = tok.substr(eq + 1);
        if (v.find('.') != std::string::npos || v.find('e') != std::string::npos || v.find('E') != std::string::npos)
            pd.set(id, (float)atof(v.c_str()));
        else
            pd.set(id, atoi(v.c_str()));
    }
}

int main(int argc, char** argv)
{
    std::vector<std::string> inputs, weights;
    std::string out = "out.bin", param_str;
    int packing = 0;   // reserved; v1 runs elempack=1 on GPU
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) inputs.push_back(argv[++i]);
        else if (a == "--weight" && i + 1 < argc) weights.push_back(argv[++i]);
        else if (a == "--param" && i + 1 < argc) param_str = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--packing" && i + 1 < argc) packing = atoi(argv[++i]);
    }
    (void)packing;

    // --- acquire a vulkan device; skip (not fail) if none ---
    int gpu_count = get_gpu_count();
    if (gpu_count <= 0)
    {
        fprintf(stderr, "NO_VULKAN_DEVICE: get_gpu_count()=%d\n", gpu_count);
        return RC_NO_VULKAN_DEVICE;
    }
    VulkanDevice* vkdev = get_gpu_device();
    if (!vkdev)
    {
        fprintf(stderr, "NO_VULKAN_DEVICE: get_gpu_device() returned null\n");
        return RC_NO_VULKAN_DEVICE;
    }

    ParamDict pd;
    parse_params(param_str, pd);

    Layer* op = new CANDIDATE_CLASS();
    op->vkdev = vkdev;

    if (op->load_param(pd) != 0) { fprintf(stderr, "load_param failed\n"); return 2; }

    std::vector<Mat> w;
    for (size_t i = 0; i < weights.size(); i++) w.push_back(read_weight(weights[i].c_str()));
    ModelBinFromMatArrayStrict mb(w.data());
    if (op->load_model(mb) != 0) { fprintf(stderr, "load_model failed\n"); return 3; }

    Option opt;
    opt.lightmode = false;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_packed = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_packing_layout = false;   // v1: elempack=1

    VkWeightAllocator weight_vkallocator(vkdev);
    VkWeightStagingAllocator weight_staging_vkallocator(vkdev);
    VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    int rc = 0;

    if (op->create_pipeline(opt) != 0)
    {
        fprintf(stderr, "create_pipeline failed\n");
        rc = 5;
    }

    // isolated-instantiation guarantee: the candidate MUST actually run on vulkan.
    if (rc == 0 && !op->support_vulkan)
    {
        fprintf(stderr, "candidate does not support_vulkan (would fall back to CPU) — refusing\n");
        rc = 6;
    }

    // upload weights (if any) on a transfer queue
    if (rc == 0 && !w.empty())
    {
        VkTransfer cmd(vkdev);
        Option opt_upload = opt;
        opt_upload.blob_vkallocator = &weight_vkallocator;
        opt_upload.workspace_vkallocator = &weight_vkallocator;
        opt_upload.staging_vkallocator = &weight_staging_vkallocator;
        op->upload_model(cmd, opt_upload);
        cmd.submit_and_wait();
    }

    if (rc == 0)
    {
        VkCompute cmd(vkdev);

        if (op->one_blob_only)
        {
            Mat in_cpu = read_mat(inputs[0].c_str());
            fprintf(stderr, "input dims=%d (w=%d h=%d d=%d c=%d)\n",
                    in_cpu.dims, in_cpu.w, in_cpu.h, in_cpu.d, in_cpu.c);

            VkMat in_gpu;
            cmd.record_upload(in_cpu, in_gpu, opt);
            // v1: record_upload auto-packs to elempack=4 when the pack dim % 4 == 0;
            // force back to elempack=1 so a scalar candidate shader is always valid.
            if (in_gpu.elempack != 1)
            {
                VkMat tmp;
                vkdev->convert_packing(in_gpu, tmp, 1, cmd, opt);
                in_gpu = tmp;
            }

            Mat out_cpu;
            if (op->support_inplace)
            {
                int ret = op->forward_inplace(in_gpu, cmd, opt);
                if (ret != 0) { fprintf(stderr, "forward_inplace failed ret=%d\n", ret); rc = 4; }
                else cmd.record_download(in_gpu, out_cpu, opt);
            }
            else
            {
                VkMat out_gpu;
                int ret = op->forward(in_gpu, out_gpu, cmd, opt);
                if (ret != 0) { fprintf(stderr, "forward failed ret=%d\n", ret); rc = 4; }
                else cmd.record_download(out_gpu, out_cpu, opt);
            }

            if (rc == 0)
            {
                cmd.submit_and_wait();
                fprintf(stderr, "output dims=%d (w=%d h=%d d=%d c=%d)\n",
                        out_cpu.dims, out_cpu.w, out_cpu.h, out_cpu.d, out_cpu.c);
                write_mat(out_cpu, out.c_str());
            }
        }
        else
        {
            std::vector<VkMat> in_gpu(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++)
            {
                Mat in_cpu = read_mat(inputs[i].c_str());
                cmd.record_upload(in_cpu, in_gpu[i], opt);
                if (in_gpu[i].elempack != 1)
                {
                    VkMat tmp;
                    vkdev->convert_packing(in_gpu[i], tmp, 1, cmd, opt);
                    in_gpu[i] = tmp;
                }
            }
            std::vector<VkMat> out_gpu(1);
            int ret;
            if (op->support_inplace) { out_gpu = in_gpu; ret = op->forward_inplace(out_gpu, cmd, opt); }
            else ret = op->forward(in_gpu, out_gpu, cmd, opt);

            if (ret != 0) { fprintf(stderr, "forward(multi) failed ret=%d\n", ret); rc = 4; }
            else
            {
                Mat out_cpu;
                cmd.record_download(out_gpu[0], out_cpu, opt);
                cmd.submit_and_wait();
                fprintf(stderr, "output[0] dims=%d (w=%d h=%d d=%d c=%d)\n",
                        out_cpu.dims, out_cpu.w, out_cpu.h, out_cpu.d, out_cpu.c);
                write_mat(out_cpu, out.c_str());
            }
        }
    }

    op->destroy_pipeline(opt);
    delete op;
    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    weight_vkallocator.clear();
    weight_staging_vkallocator.clear();

    if (rc != 0) return rc;
    printf("RUNNER_OK out=%s\n", out.c_str());
    return 0;
}
