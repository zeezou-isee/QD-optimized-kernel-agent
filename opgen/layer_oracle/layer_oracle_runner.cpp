// Generic ncnn layer "oracle" runner (方案A).
//
// Compiled ONCE per candidate via -D macros together with the candidate layer
// .cpp, linked against libncnn. It instantiates the candidate Layer directly
// (so it exercises THAT .cpp), feeds inputs/params/weights, runs forward, and
// writes the output — no per-operator C++ test file needed.
//
// Build:
//   g++ -std=c++11 -I <ncnn/src> -I <ncnn/build_lib/src> \
//       layer_oracle_runner.cpp <candidate>.cpp -L <build_lib/src> -lncnn -fopenmp \
//       -DCANDIDATE_HEADER='"convolution1d.h"' -DCANDIDATE_CLASS=Convolution1D -o runner
//
// Run:
//   runner --param "0=4,1=3,..." --input in0.bin [--input in1.bin] \
//          --weight w0.bin [--weight w1.bin] --out out.bin
//
// Bin protocol (matches MoKA): [int ndim][int dims...][float data...]

#include "layer.h"
#include "mat.h"
#include "modelbin.h"
#include "paramdict.h"
#include "option.h"
#include "datareader.h"

#include CANDIDATE_HEADER

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ncnn;

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
    else if (ndim == 2) m.create(dims[1], dims[0]);              // w, h
    else if (ndim == 3) m.create(dims[2], dims[1], dims[0]);     // w, h, c
    else if (ndim == 4) m.create(dims[3], dims[2], dims[1], dims[0]); // w, h, d, c
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

// read a weight bin as a flat 1D Mat (length = product of dims)
static Mat read_weight(const char* path)
{
    Mat m = read_mat(path);
    Mat flat = m.reshape(m.w * m.h * m.d * m.c);
    return flat.clone();
}

// Pack a sequence of fp32 weight Mats into a ncnn-standard .bin byte stream
// (in-memory), so we can use the REAL ncnn ModelBinFromDataReader to feed
// op->load_model — making the LayerOracle's mb.load() semantics IDENTICAL to
// what the kernel sees inside ncnn::Net at production / NetOracle time.
//
// Layout per weight MUST match what ncnn modelwriter emits for THIS weight slot,
// because the kernel reads each slot with a fixed mb.load(w, type) and the real
// ModelBinFromDataReader advances the cursor differently per type:
//
//   * PRIMARY weight (modelwriter fwrite_weight_tag_data, e.g. Convolution /
//     InnerProduct / Gemm weight_data): bin = [4-byte tag=0][raw fp32].
//     Kernel reads it with type 0 (auto-detect); the reader consumes the tag.
//     -> we write this layout when wflag==0.
//
//   * SECONDARY weight (modelwriter fwrite_weight_data, e.g. bias_data, and ALL
//     of BatchNorm slope/mean/var/bias, Scale, PReLU slope): bin = [raw fp32]
//     with NO tag. Kernel reads it with type 1 (raw). -> we write this layout
//     when wflag==1.
//
// The per-weight flag comes from the ncnn interface dict (weights_load_order[i].
// flag) via --weight-flag, so the LayerOracle bin is byte-identical to the real
// .ncnn.bin for this layer. A kernel that reads the WRONG type for a slot
// misaligns the cursor here exactly as it would inside ncnn::Net -> the bug is
// caught at LayerOracle time, and a correct kernel passes both. (Before this,
// every slot got a tag, so a correct BatchNorm reading type 1 misread the tag
// as data and produced var=0 -> 1/sqrt(eps) blowups.)
//
// Default (no --weight-flag given) is 0 = tagged, preserving prior behavior for
// callers that don't pass flags.
//
// We use DataReaderFromMemory so no temp file is needed.
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
            // PRIMARY/tagged slot: 4-byte tag=0 (fp32-raw branch), then data.
            unsigned int zero_flag = 0;
            const unsigned char* fp = (const unsigned char*)&zero_flag;
            bin.insert(bin.end(), fp, fp + sizeof(unsigned int));
        }
        // SECONDARY/raw slot (flag==1): no tag, raw data only.
        const unsigned char* dp = (const unsigned char*)(const float*)m;
        bin.insert(bin.end(), dp, dp + (size_t)n * sizeof(float));
    }
    return bin;
}

static void parse_params(const std::string& s, ParamDict& pd)
{
    // comma-separated id=value ; value with '.'/'e' -> float else int
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
    std::vector<int> wflags;   // per-weight bin layout: 0=tagged (type 0), 1=raw (type 1)
    std::string out = "out.bin", param_str;
    int packing = 0;   // 0 = off (naive elempack=1); N>0 = pack inputs to elempack N (arm NC4HW4)
    int fp16_storage = 0;    // 1 = enable ncnn opt.use_fp16_storage (half-precision weights/blobs)
    int fp16_arith = 0;      // 1 = enable ncnn opt.use_fp16_arithmetic (requires HAS_ASIMDHP)
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) inputs.push_back(argv[++i]);
        else if (a == "--weight" && i + 1 < argc) weights.push_back(argv[++i]);
        else if (a == "--weight-flag" && i + 1 < argc) wflags.push_back(atoi(argv[++i]));
        else if (a == "--param" && i + 1 < argc) param_str = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--packing" && i + 1 < argc) packing = atoi(argv[++i]);
        else if (a == "--fp16-storage") fp16_storage = 1;
        else if (a == "--fp16-arith")   { fp16_storage = 1; fp16_arith = 1; }
    }
    // default any unspecified weight flags to 0 (tagged) so the bin layout is
    // well-defined even when the caller passes fewer --weight-flag than --weight.
    while (wflags.size() < weights.size()) wflags.push_back(0);

    ParamDict pd;
    parse_params(param_str, pd);

    Layer* op = new CANDIDATE_CLASS();
    if (op->load_param(pd) != 0) { fprintf(stderr, "load_param failed\n"); return 2; }

    std::vector<Mat> w;
    for (size_t i = 0; i < weights.size(); i++) w.push_back(read_weight(weights[i].c_str()));
    // Pack weights into ncnn-standard fp32-raw bin layout, then use the REAL
    // ncnn DataReader + ModelBinFromDataReader so op->load_model sees the
    // exact same mb.load(w, type) semantics it will see inside ncnn::Net at
    // production / NetOracle time. This way mb.load type misuse is caught
    // here, not silently slipped to NetOracle.
    std::vector<unsigned char> mb_bin = pack_weights_bin(w.data(), wflags.data(), (int)w.size());
    const unsigned char* mem_ptr = mb_bin.data();
    DataReaderFromMemory dr(mem_ptr);
    ModelBinFromDataReader mb(dr);
    if (op->load_model(mb) != 0) { fprintf(stderr, "load_model failed\n"); return 3; }

    Option opt;
    opt.lightmode = false;
    opt.num_threads = 1;
    opt.use_packing_layout = packing > 0;   // arm NC4HW4 path when --packing N
    // fp16 tiers (opt-in via --fp16-storage / --fp16-arith). Kernels that declare
    // support_fp16_storage in create_pipeline will consume half-precision blobs;
    // kernels without it stay fp32 even when the flag is on (ncnn's Layer base
    // handles the fp16<->fp32 conversion). Arithmetic requires ARMv8.2 FP16.
    opt.use_fp16_packed = fp16_storage != 0;
    opt.use_fp16_storage = fp16_storage != 0;
    opt.use_fp16_arithmetic = fp16_arith != 0;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;
    op->create_pipeline(opt);

    // pack to elempack N (e.g. 4 for arm NEON); identity when packing==0.
    auto pack = [&](const Mat& m) -> Mat {
        if (packing <= 0) return m;
        Mat p; convert_packing(m, p, packing, opt); return p;
    };
    // back to elempack=1 so the written .bin is plain row-major for the oracle.
    auto unpack = [&](const Mat& m) -> Mat {
        if (packing <= 0 || m.elempack == 1) return m;
        Mat u; convert_packing(m, u, 1, opt); return u;
    };

    int ret = 0;
    if (op->one_blob_only)
    {
        Mat in = pack(read_mat(inputs[0].c_str()));
        fprintf(stderr, "input dims=%d (w=%d h=%d d=%d c=%d elempack=%d)\n", in.dims, in.w, in.h, in.d, in.c, in.elempack);
        Mat o;
        if (op->support_inplace) { o = in.clone(); ret = op->forward_inplace(o, opt); }
        else ret = op->forward(in, o, opt);
        if (ret == 0)
        {
            o = unpack(o);
            fprintf(stderr, "output dims=%d (w=%d h=%d d=%d c=%d)\n", o.dims, o.w, o.h, o.d, o.c);
            write_mat(o, out.c_str());
        }
    }
    else
    {
        std::vector<Mat> ins;
        for (size_t i = 0; i < inputs.size(); i++) ins.push_back(pack(read_mat(inputs[i].c_str())));
        std::vector<Mat> outs(1);
        if (op->support_inplace) { outs = ins; ret = op->forward_inplace(outs, opt); }
        else ret = op->forward(ins, outs, opt);
        if (ret == 0 && !outs.empty())
        {
            Mat o = unpack(outs[0]);
            fprintf(stderr, "output[0] dims=%d (w=%d h=%d d=%d c=%d)\n", o.dims, o.w, o.h, o.d, o.c);
            write_mat(o, out.c_str());
        }
    }

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0) { fprintf(stderr, "forward failed ret=%d\n", ret); return 4; }
    printf("RUNNER_OK out=%s\n", out.c_str());
    return 0;
}
