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
    std::string out = "out.bin", param_str;
    int packing = 0;   // 0 = off (naive elempack=1); N>0 = pack inputs to elempack N (arm NC4HW4)
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) inputs.push_back(argv[++i]);
        else if (a == "--weight" && i + 1 < argc) weights.push_back(argv[++i]);
        else if (a == "--param" && i + 1 < argc) param_str = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--packing" && i + 1 < argc) packing = atoi(argv[++i]);
    }

    ParamDict pd;
    parse_params(param_str, pd);

    Layer* op = new CANDIDATE_CLASS();
    if (op->load_param(pd) != 0) { fprintf(stderr, "load_param failed\n"); return 2; }

    std::vector<Mat> w;
    for (size_t i = 0; i < weights.size(); i++) w.push_back(read_weight(weights[i].c_str()));
    ModelBinFromMatArray mb(w.data());
    if (op->load_model(mb) != 0) { fprintf(stderr, "load_model failed\n"); return 3; }

    Option opt;
    opt.lightmode = false;
    opt.num_threads = 1;
    opt.use_packing_layout = packing > 0;   // arm NC4HW4 path when --packing N
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
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
