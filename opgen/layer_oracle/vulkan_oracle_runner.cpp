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
#include "datareader.h"
#include "gpu.h"
#include "command.h"
#include "pipelinecache.h"

#include CANDIDATE_HEADER
// Read-only view of the shared shader-fail flag; definition lives in
// cand_vulkan_shader.h alongside compile_candidate_shader (function-local
// static, ODR-safe under inline linkage).
#ifdef CANDIDATE_SHADER
#include "cand_vulkan_shader.h"
#else
// Native-subclass builds compile with no CANDIDATE_SHADER (they reuse ncnn's
// baked SPIR-V registry). Provide a stub so the runner links either way.
namespace ncnn { inline bool& _cand_shader_compile_failed() { static bool v = false; return v; } }
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

using namespace ncnn;

#define RC_NO_VULKAN_DEVICE 42
// Structured runner exit codes — keep in sync with cand_vulkan_shader.h so
// the oracle-side driver can route repair prompts (shader-side vs host-side vs
// dispatch-side) to the correct role without grepping stderr.
#define RC_SHADER_COMPILE_FAIL 51
#define RC_PIPELINE_CREATE_FAIL 52
#define RC_DISPATCH_FAIL 53

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

// Pack weights into a ncnn-standard fp32-raw bin layout (in-memory). Used with
// DataReaderFromMemory + ModelBinFromDataReader (ncnn's real reader chain) so
// op->load_model sees the exact mb.load(w, type) semantics it gets in
// production / NetOracle. See layer_oracle_runner.cpp for the full rationale.
//
// Layout per weight: [4-byte flag = 0x00000000] [w * sizeof(float) raw fp32].
// Every weight gets a flag header, so the kernel must call mb.load(w, 0) for
// every weight (this matches the fp32-storage modelwriter convention).
// Per-weight bin layout, matching ncnn modelwriter (same rule as the base
// layer_oracle_runner): wflag==0 => PRIMARY (tagged, read with type 0) so we
// prepend a 4-byte tag=0; wflag==1 => SECONDARY (raw, read with type 1) so no
// tag. Flags come from the interface dict via --weight-flag. Default 0.
static std::vector<unsigned char> pack_weights_bin(const Mat* weights,
                                                   const int* wflags, int n_weights)
{
    std::vector<unsigned char> bin;
    for (int i = 0; i < n_weights; i++)
    {
        const Mat& m = weights[i];
        int n = m.w * m.h * m.d * m.c;
        int flag = wflags ? wflags[i] : 0;
        if (flag == 0)
        {
            unsigned int zero_flag = 0;
            const unsigned char* fp = (const unsigned char*)&zero_flag;
            bin.insert(bin.end(), fp, fp + sizeof(unsigned int));
        }
        const unsigned char* dp = (const unsigned char*)(const float*)m;
        bin.insert(bin.end(), dp, dp + (size_t)n * sizeof(float));
    }
    return bin;
}

// Parse the CLI --param string into a ParamDict.
//
// Handles two shapes:
//  (a) scalar:   "id=value"                       (int or float by '.' / 'e' sniff)
//  (b) array:    "-N=count,v0,v1,...,v_{count-1}" (ncnn negative-key trick; N = 23300+real_id)
//
// The previous naive impl split the whole string on commas and only looked for '=', which
// silently dropped every v_i in an array (they had no '='), so Reduction/Convolution/etc.
// got axes=Mat() and behaved like a no-op — that's the bug that made ReduceSum's native
// subclass return the input unchanged. Now we recognize a `-N=` opener and consume the
// exact count of comma-separated ints/floats that follow.
static bool _looks_float(const std::string& v)
{
    return v.find('.') != std::string::npos
        || v.find('e') != std::string::npos
        || v.find('E') != std::string::npos;
}

static void parse_params(const std::string& s, ParamDict& pd)
{
    // Split on comma into tokens first, then walk them so an array's tail values are
    // reachable via index-arithmetic rather than nested state.
    std::vector<std::string> toks;
    for (size_t p = 0, n; p <= s.size(); p = n + 1) {
        n = s.find(',', p);
        if (n == std::string::npos) { toks.push_back(s.substr(p)); break; }
        toks.push_back(s.substr(p, n - p));
    }
    for (size_t ti = 0; ti < toks.size(); ) {
        const std::string& tok = toks[ti];
        size_t eq = tok.find('=');
        if (eq == std::string::npos) { ti++; continue; }
        int id = atoi(tok.substr(0, eq).c_str());
        std::string v = tok.substr(eq + 1);
        if (id <= -23300) {
            int real_id = -id - 23300;
            int count = atoi(v.c_str());
            // consume the next `count` tokens as the array values
            if (count < 0 || ti + 1 + (size_t)count > toks.size()) { ti++; continue; }
            // decide int vs float array from the first value
            bool as_float = false;
            for (int k = 0; k < count; k++) if (_looks_float(toks[ti + 1 + k])) { as_float = true; break; }
            if (as_float) {
                Mat arr(count);
                float* p = arr;
                for (int k = 0; k < count; k++) p[k] = (float)atof(toks[ti + 1 + k].c_str());
                pd.set(real_id, arr);
            } else {
                Mat arr(count, (size_t)4u, 1);      // int Mat of length count
                int* p = arr;
                for (int k = 0; k < count; k++) p[k] = atoi(toks[ti + 1 + k].c_str());
                pd.set(real_id, arr);
            }
            ti += 1 + count;
        } else {
            if (_looks_float(v)) pd.set(id, (float)atof(v.c_str()));
            else pd.set(id, atoi(v.c_str()));
            ti++;
        }
    }
}

int main(int argc, char** argv)
{
    std::vector<std::string> inputs, weights;
    std::vector<int> wflags;   // per-weight bin layout: 0=tagged(type0), 1=raw(type1)
    std::string out = "out.bin", param_str;
    int packing = 0;   // reserved; v1 runs elempack=1 on GPU
    int bench_iters = 0;   // --bench N: after the correctness run, time N GPU
                           // forwards and print BENCH_{MIN,MAX,AVG}_MS (device latency)
    int bench_warmup = 10; // discarded warmup GPU forwards before timing (default 10)
    std::string layer_name;  // --layer <name>: instantiate a BUILT-IN ncnn VULKAN layer via
                             // create_layer_vulkan (native GPU baseline) instead of the
                             // compiled-in CANDIDATE_CLASS. Reuses the SAME runner + GPU
                             // harness (baked SPIR-V from libncnn-vk) -> native GPU latency,
                             // zero extra compile.
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) inputs.push_back(argv[++i]);
        else if (a == "--weight" && i + 1 < argc) weights.push_back(argv[++i]);
        else if (a == "--weight-flag" && i + 1 < argc) wflags.push_back(atoi(argv[++i]));
        else if (a == "--param" && i + 1 < argc) param_str = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--packing" && i + 1 < argc) packing = atoi(argv[++i]);
        else if (a == "--bench" && i + 1 < argc) bench_iters = atoi(argv[++i]);
        else if (a == "--bench-warmup" && i + 1 < argc) bench_warmup = atoi(argv[++i]);
        else if (a == "--layer" && i + 1 < argc) layer_name = argv[++i];
    }
    (void)packing;
    while (wflags.size() < weights.size()) wflags.push_back(0);

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

    // --layer <name> -> built-in ncnn VULKAN layer via create_layer_vulkan (native
    // GPU baseline); otherwise the compiled-in candidate class. Same GPU harness
    // (vkdev + create_pipeline uses ncnn's baked SPIR-V) for both. If the op has no
    // vulkan variant, create_layer_vulkan returns null -> RC_NO_VULKAN_DEVICE so the
    // caller skips the speedup (not a failure).
    Layer* op = layer_name.empty() ? (Layer*)(new CANDIDATE_CLASS())
                                   : create_layer_vulkan(layer_name.c_str());
    if (!op) { fprintf(stderr, "NO_VULKAN_NATIVE: create_layer_vulkan(%s) returned null\n",
                       layer_name.c_str()); return RC_NO_VULKAN_DEVICE; }
    op->vkdev = vkdev;

    if (op->load_param(pd) != 0) { fprintf(stderr, "load_param failed\n"); return 2; }

    // Read inputs up front so we can hand shape hints to create_pipeline before
    // load_model / create_pipeline — matches ncnn's Net order (net.cpp:1332-1409
    // sets bottom_shapes / top_shapes during load_param, well before
    // create_pipeline at net.cpp:1854). testutil.cpp:935-1001 packs each shape
    // according to `support_vulkan_packing` and (c/w/h % 4 == 0 ? 4 : 1).
    std::vector<Mat> in_cpus(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
        in_cpus[i] = read_mat(inputs[i].c_str());
    for (size_t i = 0; i < in_cpus.size(); i++)
        fprintf(stderr, "input[%zu] dims=%d (w=%d h=%d d=%d c=%d)\n",
                i, in_cpus[i].dims, in_cpus[i].w, in_cpus[i].h, in_cpus[i].d, in_cpus[i].c);

    // Compute a packed shape hint for one input (used for both bottom_shapes
    // and, for inplace one_blob_only ops, top_shapes). See testutil.cpp:940-999.
    auto pack_shape_hint = [&](const Mat& shape) -> Mat {
        int dims = shape.dims;
        int elempack = 1;
        if (op->support_vulkan_packing)
        {
            if (dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
            if (dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
            if (dims == 3 || dims == 4) elempack = shape.c % 4 == 0 ? 4 : 1;
        }
        size_t elemsize = (size_t)elempack * 4u;   // fp32 only (opt.use_fp16_storage=false)
        if (dims == 1) return Mat(shape.w / elempack, (void*)0, elemsize, elempack);
        if (dims == 2) return Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
        if (dims == 3) return Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
        if (dims == 4) return Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);
        return Mat();
    };

    // Hand shape hints to the layer so pack-aware ncnn <Op>_vulkan layers
    // (BatchNorm_vulkan, Convolution_vulkan, ...) pick the right pipeline in
    // create_pipeline(). Only set BOTH bottom_shapes and top_shapes together —
    // never one without the other. Setting `bottom_shapes` alone defeats the
    // `shape.dims == 0` short-circuit in ncnn's pipeline-selection macros
    // (e.g. Reshape_vulkan.cpp:95,103,111,119) but leaves out_shape empty, so
    // the branch `shape.elempack == 4 && out_shape.elempack == 4` fails and
    // pipeline_reshape_pack4 stays NULL → forward crashes with EXC_BAD_ACCESS
    // inside Pipeline::shader_info().
    //
    // We only know top_shape for inplace one_blob_only ops (top == bottom).
    // For non-inplace or multi-input, leave BOTH empty and let ncnn build all
    // pack variants unconditionally (via the `shape.dims == 0` short-circuit).
    if (op->one_blob_only && op->support_inplace) {
        Mat h = pack_shape_hint(in_cpus[0]);
        op->bottom_shapes = std::vector<Mat>{h};
        op->top_shapes = std::vector<Mat>{h};
    }

    std::vector<Mat> w;
    for (size_t i = 0; i < weights.size(); i++) w.push_back(read_weight(weights[i].c_str()));
    std::vector<unsigned char> mb_bin = pack_weights_bin(w.data(), wflags.data(), (int)w.size());
    const unsigned char* mem_ptr = mb_bin.data();
    DataReaderFromMemory dr(mem_ptr);
    ModelBinFromDataReader mb(dr);
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
    // The vulkan branch of convert_layout (net.cpp:586-622) reads
    // `layer->support_vulkan_packing` directly, not opt.use_packing_layout —
    // but testutil.cpp:874 gates test_layer_gpu on `_opt.use_packing_layout`,
    // so we set it true to match the reference harness.
    opt.use_packing_layout = true;

    VkWeightAllocator weight_vkallocator(vkdev);
    VkWeightStagingAllocator weight_staging_vkallocator(vkdev);
    VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;
    // Required by ncnn — Pipeline::create looks it up unconditionally; a NULL
    // cache crashes on some paths. testutil.cpp:1267-1270 always sets one.
    PipelineCache pipeline_cache(vkdev);
    opt.pipeline_cache = &pipeline_cache;

    int rc = 0;

    if (op->create_pipeline(opt) != 0)
    {
        // The candidate's create_pipeline calls compile_candidate_shader; if
        // THAT failed, we split the error attribution: shader-side (SPIR-V
        // compile fail, glslang error above) vs host-side (pipeline layout
        // rejection, spec-const mismatch, etc.).
        if (_cand_shader_compile_failed())
        {
            fprintf(stderr, "create_pipeline failed via shader compile fail\n");
            rc = RC_SHADER_COMPILE_FAIL;
        }
        else
        {
            fprintf(stderr, "create_pipeline failed (pipeline object, not shader)\n");
            rc = RC_PIPELINE_CREATE_FAIL;
        }
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

    // Match convert_layout in Net::do_forward_layer (net.cpp:586-622) exactly.
    // The `support_vulkan_packing` bool is set in each layer's ctor and decides
    // whether pack4 is even legal for this op. When true, pack to natural
    // elempack (elemcount % 4 == 0 ? 4 : 1); when false, force elempack=1.
    // This is what makes pack-aware layers (BatchNorm/Convolution/...) receive
    // the pipeline their create_pipeline actually compiled.
    auto pack_input = [&](VkMat& blob, VkCompute& cmd_local) {
        int dst_elempack = 1;
        if (op->support_vulkan_packing)
        {
            int elemcount = 0;
            if (blob.dims == 1) elemcount = blob.elempack * blob.w;
            if (blob.dims == 2) elemcount = blob.elempack * blob.h;
            if (blob.dims >= 3) elemcount = blob.elempack * blob.c;
            if (elemcount % 4 == 0) dst_elempack = 4;
        }
        if (blob.elempack != dst_elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(blob, tmp, dst_elempack, cmd_local, opt);
            blob = tmp;
        }
    };
    // We compare against PyTorch by dumping the CPU Mat as a flat fp32 stream
    // (write_mat walks m.c × w*h*d channels of `float` — it CANNOT read packed
    // vec4 storage). testutil.cpp compares via CompareMat which is elempack-
    // aware; ours isn't. So force any pack4/pack8 output back to elempack=1
    // before record_download, matching what ncnn's `ncnn2mem` / user-facing
    // Extractor::extract also does implicitly on the CPU side.
    // record_download itself re-packs the source blob to elempack=1 or 4 based
    // on opt.use_packing_layout (command.cpp:445-447): with packing_layout=true
    // and a c%4==0 output, it stays at elempack=4 → our flat-fp32 write_mat
    // then only reads m.c×w×h×d floats and drops 3/4 of the data. So we
    // download with a local opt that forces packing OFF; the input-side + op
    // forward still run with packing ON (matching each layer's compiled pipeline).
    Option opt_dl = opt;
    opt_dl.use_packing_layout = false;

    if (rc == 0)
    {
        VkCompute cmd(vkdev);

        if (op->one_blob_only)
        {
            VkMat in_gpu;
            cmd.record_upload(in_cpus[0], in_gpu, opt);
            pack_input(in_gpu, cmd);

            Mat out_cpu;
            if (op->support_inplace)
            {
                int ret = op->forward_inplace(in_gpu, cmd, opt);
                if (ret != 0) { fprintf(stderr, "forward_inplace failed ret=%d\n", ret); rc = RC_DISPATCH_FAIL; }
                else cmd.record_download(in_gpu, out_cpu, opt_dl);
            }
            else
            {
                VkMat out_gpu;
                int ret = op->forward(in_gpu, out_gpu, cmd, opt);
                if (ret != 0) { fprintf(stderr, "forward failed ret=%d\n", ret); rc = RC_DISPATCH_FAIL; }
                else cmd.record_download(out_gpu, out_cpu, opt_dl);
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
            std::vector<VkMat> in_gpu(in_cpus.size());
            for (size_t i = 0; i < in_cpus.size(); i++)
            {
                cmd.record_upload(in_cpus[i], in_gpu[i], opt);
                pack_input(in_gpu[i], cmd);
            }
            std::vector<VkMat> out_gpu(1);
            int ret;
            if (op->support_inplace) { out_gpu = in_gpu; ret = op->forward_inplace(out_gpu, cmd, opt); }
            else ret = op->forward(in_gpu, out_gpu, cmd, opt);

            if (ret != 0) { fprintf(stderr, "forward(multi) failed ret=%d\n", ret); rc = RC_DISPATCH_FAIL; }
            else
            {
                Mat out_cpu;
                cmd.record_download(out_gpu[0], out_cpu, opt_dl);
                cmd.submit_and_wait();
                fprintf(stderr, "output[0] dims=%d (w=%d h=%d d=%d c=%d elempack=%d)\n",
                        out_cpu.dims, out_cpu.w, out_cpu.h, out_cpu.d, out_cpu.c, out_cpu.elempack);
                write_mat(out_cpu, out.c_str());
            }
        }
    }

    // --- optional device-latency bench (--bench N): warmup + N timed GPU
    // forwards, min per-iter ms → BENCH_MIN_MS. Uploads inputs once; re-runs the
    // op on the GPU blob (for inplace ops this re-applies the op, which still
    // measures the kernel's per-launch cost). Host-timed around submit_and_wait. ---
    if (rc == 0 && bench_iters > 0)
    {
        std::vector<VkMat> gin(in_cpus.size());
        {
            VkCompute up(vkdev);
            for (size_t i = 0; i < in_cpus.size(); i++) { up.record_upload(in_cpus[i], gin[i], opt); pack_input(gin[i], up); }
            up.submit_and_wait();
        }
        const int warmup = bench_warmup;
        double best_ms = 1e30, worst_ms = 0.0, sum_ms = 0.0;
        for (int it = 0; it < warmup + bench_iters; it++)
        {
            VkCompute cmd(vkdev);
            auto t0 = std::chrono::steady_clock::now();
            int ret = 0;
            if (op->one_blob_only)
            {
                if (op->support_inplace) ret = op->forward_inplace(gin[0], cmd, opt);
                else { VkMat o; ret = op->forward(gin[0], o, cmd, opt); }
            }
            else
            {
                std::vector<VkMat> o(1);
                if (op->support_inplace) { o = gin; ret = op->forward_inplace(o, cmd, opt); }
                else ret = op->forward(gin, o, cmd, opt);
            }
            if (ret == 0) ret = cmd.submit_and_wait();
            auto t1 = std::chrono::steady_clock::now();
            if (ret != 0) { fprintf(stderr, "bench iter %d failed ret=%d\n", it, ret); break; }
            if (it >= warmup)
            {
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best_ms) best_ms = ms;
                if (ms > worst_ms) worst_ms = ms;
                sum_ms += ms;
            }
        }
        double avg_ms = bench_iters > 0 ? sum_ms / bench_iters : 0.0;
        printf("BENCH_MIN_MS=%.4f\n", best_ms);
        printf("BENCH_MAX_MS=%.4f\n", worst_ms);
        printf("BENCH_AVG_MS=%.4f\n", avg_ms);
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
