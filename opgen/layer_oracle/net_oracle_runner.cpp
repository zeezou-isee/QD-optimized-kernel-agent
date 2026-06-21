// Generic ncnn Net runner: load a .ncnn.param/.bin model (which may reference a
// freshly-installed custom layer compiled into libncnn.a), feed named input
// blobs, run, and write the output blob — for end-to-end numeric verification of
// a graph conversion vs PyTorch.
//
// Build (against a libncnn.a that already contains the new layer):
//   g++ -std=c++11 -I <ncnn/src> -I <build_lib/src> net_oracle_runner.cpp \
//       <build_lib/src/libncnn.a> -o net_runner
//
// Run:
//   net_runner --param m.ncnn.param --bin m.ncnn.bin \
//              --in in0=in0.bin --in in1=in1.bin --out out0 --outfile out.bin
//
// Bin protocol: [int ndim][int dims...][float data...]

#include "net.h"
#include "mat.h"

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
    else if (ndim == 2) m.create(dims[1], dims[0]);
    else if (ndim == 3) m.create(dims[2], dims[1], dims[0]);
    else if (ndim == 4) m.create(dims[3], dims[2], dims[1], dims[0]);
    else { fclose(fp); return Mat(); }
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

int main(int argc, char** argv)
{
    std::string param, bin, out = "out0", outfile = "out.bin";
    std::vector<std::pair<std::string, std::string> > inputs;
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--param" && i + 1 < argc) param = argv[++i];
        else if (a == "--bin" && i + 1 < argc) bin = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--outfile" && i + 1 < argc) outfile = argv[++i];
        else if (a == "--in" && i + 1 < argc)
        {
            std::string kv = argv[++i];
            size_t eq = kv.find('=');
            inputs.push_back(std::make_pair(kv.substr(0, eq), kv.substr(eq + 1)));
        }
    }

    Net net;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;

    if (net.load_param(param.c_str()) != 0) { fprintf(stderr, "load_param failed\n"); return 2; }
    if (net.load_model(bin.c_str()) != 0) { fprintf(stderr, "load_model failed\n"); return 3; }

    Extractor ex = net.create_extractor();
    for (size_t i = 0; i < inputs.size(); i++)
    {
        Mat m = read_mat(inputs[i].second.c_str());
        if (ex.input(inputs[i].first.c_str(), m) != 0)
        { fprintf(stderr, "input '%s' failed\n", inputs[i].first.c_str()); return 4; }
    }
    Mat o;
    if (ex.extract(out.c_str(), o) != 0) { fprintf(stderr, "extract '%s' failed\n", out.c_str()); return 5; }
    fprintf(stderr, "net out dims=%d (w=%d h=%d d=%d c=%d)\n", o.dims, o.w, o.h, o.d, o.c);
    write_mat(o, outfile.c_str());
    printf("NET_RUNNER_OK\n");
    return 0;
}
