#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(BENCH_DUCC)
#include "ducc0/fft/fft.h"
#if BENCH_DUCC_CXX_STANDARD == 17
static_assert(__cplusplus == 201703L, "Expected C++17");
#elif BENCH_DUCC_CXX_STANDARD == 23
static_assert(__cplusplus >= 202100L && __cplusplus <= 202302L, "Expected C++23");
#else
#error Unsupported DUCC C++ standard
#endif
#elif defined(BENCH_FFTW)
#include <fftw3.h>
#elif defined(BENCH_ADMIRAL)
#include <admiral/admiral.hpp>
#endif

using Clock = std::chrono::steady_clock;
using Shape = std::vector<size_t>;
using Strides = std::vector<ptrdiff_t>;
using Call = std::function<void()>;

struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Config {
    std::string kind, mode, scale, planning, infile, outfile;
    Shape in_shape, out_shape, axes;
    Strides in_stride;
    size_t offset, threads;
    int type, samples;
    double target_ms;
};

template<class V>
V parse(const std::string& s) {
    V result;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ','))
        result.push_back(std::stoll(token));
    return result;
}

size_t size(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t(1),
                           std::multiplies<size_t>());
}

Strides contiguous(const Shape& shape) {
    Strides stride(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i)
        stride[i - 2] = stride[i - 1] * shape[i - 1];
    return stride;
}

template<class T>
T scale(const Config& c) {
    const auto& shape = c.kind == "irfft" ? c.out_shape : c.in_shape;
    T n = 1;
    for (auto a : c.axes) n *= T(shape[a]);
    if (c.scale == "inverse") return T(1) / n;
    if (c.scale == "ortho") return T(1) / std::sqrt(n);
    return T(1);
}

bool inverse(const Config& c) {
    return c.kind == "ifft" || c.kind == "irfft";
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::nano>(Clock::now() - start).count();
}

#if defined(BENCH_FFTW)
template<class T> struct Fftw;

#define FFTW_API(T, P, COMPLEX, PLAN, DIM)                                      \
template<> struct Fftw<T> {                                                    \
    using Plan = PLAN; using Dim = DIM;                                        \
    static void threads(int n) {                                              \
        if (!P##_init_threads()) throw std::runtime_error("FFTW threads");     \
        P##_plan_with_nthreads(n);                                             \
    }                                                                         \
    static Plan c2c(int r, const Dim* d, int b, const Dim* h,                    \
                    void* in, void* out, int sign, unsigned flags) {           \
        return P##_plan_guru64_dft(r,d,b,h,static_cast<COMPLEX*>(in),           \
                                  static_cast<COMPLEX*>(out),sign,flags);      \
    }                                                                         \
    static Plan r2c(int r, const Dim* d, int b, const Dim* h,                    \
                    T* in, void* out, unsigned flags) {                        \
        return P##_plan_guru64_dft_r2c(r,d,b,h,in,static_cast<COMPLEX*>(out),flags); \
    }                                                                         \
    static Plan c2r(int r, const Dim* d, int b, const Dim* h,                    \
                    void* in, T* out, unsigned flags) {                        \
        return P##_plan_guru64_dft_c2r(r,d,b,h,static_cast<COMPLEX*>(in),out,flags); \
    }                                                                         \
    static Plan r2r(int r, const Dim* d, int b, const Dim* h,                    \
                    T* in, T* out, const fftw_r2r_kind* kinds, unsigned flags) {\
        return P##_plan_guru64_r2r(r,d,b,h,in,out,kinds,flags);                  \
    }                                                                         \
    static void execute(Plan p) { P##_execute(p); }                             \
    static void destroy(Plan p) { P##_destroy_plan(p); }                        \
};

FFTW_API(float, fftwf, fftwf_complex, fftwf_plan, fftwf_iodim64)
FFTW_API(double, fftw, fftw_complex, fftw_plan, fftw_iodim64)
FFTW_API(long double, fftwl, fftwl_complex, fftwl_plan, fftwl_iodim64)
#undef FFTW_API
#endif

template<class T, class I, class O>
Call make_call(const Config& c, I* in, O* out) {
    using C = std::complex<T>;
    const T f = scale<T>(c);
    const bool forward = !inverse(c);
#if defined(BENCH_DUCC)
    if (c.planning != "estimate") throw Unsupported("DUCC has no explicit planning effort");
    auto a = ducc0::cfmav<I>(in, c.in_shape, c.in_stride);
    auto b = ducc0::vfmav<O>(out, c.out_shape);
    if constexpr (std::is_same_v<I,C> && std::is_same_v<O,C>) {
        return [=]{ ducc0::c2c(a,b,c.axes,forward,f,c.threads); };
    } else if constexpr (std::is_same_v<I,T> && std::is_same_v<O,C>) {
        return [=]{ ducc0::r2c(a,b,c.axes,true,f,c.threads); };
    } else if constexpr (std::is_same_v<I,C> && std::is_same_v<O,T>) {
        return [=]{ ducc0::c2r(a,b,c.axes,false,f,c.threads); };
    } else {
        if (c.kind == "dct") return [=]{ ducc0::dct(a,b,c.axes,c.type,f,false,c.threads); };
        if (c.kind == "dst") return [=]{ ducc0::dst(a,b,c.axes,c.type,f,false,c.threads); };
        if (c.kind == "r2r_fftw") return [=]{ ducc0::r2r_fftw(a,b,c.axes,true,f,c.threads); };
        if (c.kind == "separable_fht") return [=]{ ducc0::r2r_separable_fht(a,b,c.axes,f,c.threads); };
        if (c.kind == "genuine_fht") return [=]{ ducc0::r2r_genuine_fht(a,b,c.axes,f,c.threads); };
    }
#elif defined(BENCH_FFTW)
    using F = Fftw<T>;
    using Dim = typename F::Dim;
    const auto& shape = c.kind == "irfft" ? c.out_shape : c.in_shape;
    const auto os = contiguous(c.out_shape);
    std::vector<Dim> dims, batch;
    for (auto a : c.axes)
        dims.push_back({ptrdiff_t(shape[a]),c.in_stride[a],os[a]});
    for (size_t a = 0; a < shape.size(); ++a)
        if (std::find(c.axes.begin(),c.axes.end(),a) == c.axes.end())
            batch.push_back({ptrdiff_t(shape[a]),c.in_stride[a],os[a]});
    int r = dims.size(), nb = batch.size();
    auto d = dims.data(), h = batch.data();
    unsigned flags = c.planning == "measure" ? FFTW_MEASURE : FFTW_ESTIMATE;
    F::threads(c.threads);
    typename F::Plan p;
    if constexpr (std::is_same_v<I,C> && std::is_same_v<O,C>)
        p = F::c2c(r,d,nb,h,in,out,forward ? FFTW_FORWARD : FFTW_BACKWARD,flags);
    else if constexpr (std::is_same_v<I,T> && std::is_same_v<O,C>)
        p = F::r2c(r,d,nb,h,in,out,flags);
    else if constexpr (std::is_same_v<I,C> && std::is_same_v<O,T>)
        p = F::c2r(r,d,nb,h,in,out,flags);
    else {
        fftw_r2r_kind k;
        if (c.kind == "dct") {
            const fftw_r2r_kind ks[] = {FFTW_REDFT00, FFTW_REDFT10, FFTW_REDFT01, FFTW_REDFT11};
            k = ks[c.type-1];
        } else if (c.kind == "dst") {
            const fftw_r2r_kind ks[] = {FFTW_RODFT00, FFTW_RODFT10, FFTW_RODFT01, FFTW_RODFT11};
            k = ks[c.type-1];
        } else if (c.kind == "r2r_fftw") k = FFTW_R2HC;
        else if (c.kind == "separable_fht" || (c.kind == "genuine_fht" && r == 1)) k = FFTW_DHT;
        else throw Unsupported("FFTW has no genuine multidimensional Hartley transform");
        std::vector<fftw_r2r_kind> kinds(r,k);
        p = F::r2r(r,d,nb,h,in,out,kinds.data(),flags);
    }
    if (!p) throw std::runtime_error("FFTW rejected plan");
    std::shared_ptr<std::remove_pointer_t<typename F::Plan>> plan(p,F::destroy);
    return [=]{
        F::execute(plan.get());
        if (f != T(1))
            for (size_t i=0; i<size(c.out_shape); ++i) out[i] *= f;
    };
#elif defined(BENCH_ADMIRAL)
    admiral::options opts;
    opts.nthreads = c.threads;
    opts.eff = c.planning == "measure" ? admiral::effort::measure : admiral::effort::estimate;
    const auto& shape = c.kind == "irfft" ? c.out_shape : c.in_shape;
    Shape all(shape.size());
    std::iota(all.begin(),all.end(),0);
    bool dense = c.in_stride == contiguous(c.in_shape);
    bool all_axes = all == c.axes;
    if constexpr (std::is_same_v<I,C> && std::is_same_v<O,C>) {
        if (dense && all_axes) {
            auto p = std::make_shared<admiral::plan<T>>(shape,opts);
            if (c.mode=="inplace") return [=]{ if (forward) p->forward(out,f); else p->inverse(out,f); };
            return [=]{ if (forward) p->forward(in,out,f); else p->inverse(in,out,f); };
        }
        if constexpr (!std::is_same_v<T,long double>) {
            if (shape.size()==2 && c.axes.size()==1 &&
                c.in_stride[0]>0 && c.in_stride[1]>0 && c.mode!="inplace") {
                auto a = c.axes[0], b = 1-a;
                auto os = contiguous(c.out_shape);
                auto p = std::make_shared<admiral::strides_plan<T>>(
                    shape[a],shape[b],c.in_stride[a],c.in_stride[b],os[a],os[b],opts);
                return [=]{ if (forward) p->forward(in,out,f); else p->inverse(in,out,f); };
            }
        }
    } else if constexpr (!std::is_same_v<I,O>) {
        if (dense && all_axes) {
            auto p = std::make_shared<admiral::plan_r2c<T>>(shape,opts);
            if constexpr (std::is_same_v<I,T>)
                return [=]{ p->forward(in,out,f); };
            else
                return [=]{ p->inverse(in,out,f); };
        }
    } else if constexpr (!std::is_same_v<T,long double>) {
        if (dense && shape.size()==1 && (c.kind=="dct" || c.kind=="dst") &&
            (c.type==2 || c.type==3)) {
            auto k = c.kind=="dct" ?
                (c.type==2 ? admiral::r2r_kind::dct2 : admiral::r2r_kind::dct3) :
                (c.type==2 ? admiral::r2r_kind::dst2 : admiral::r2r_kind::dst3);
            auto p = std::make_shared<admiral::plan_r2r<T>>(shape[0],k,1,opts);
            // Admiral's explicit type-III scale is relative to its normalized inverse.
            const T r2r_scale = c.type == 3 ? f * T(2) * T(shape[0]) : f;
            return [=]{ p->forward(in,out,r2r_scale); };
        }
    }
#endif
    throw Unsupported("native API does not support this layout, axes or transform");
}

template<class T, class I, class O>
void run(const Config& c) {
    static_assert(sizeof(std::complex<T>) == 2*sizeof(T));
    std::ifstream file(c.infile, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open input");
    auto bytes = file.tellg();
#if !defined(BENCH_DUCC)
    if (size(c.in_shape) == 0) throw Unsupported("empty batch not supported by native adapter");
#endif
    if (bytes < 0 || bytes % sizeof(I)) throw std::runtime_error("invalid input size");
    std::vector<I> storage(std::max<size_t>(1,size_t(bytes)/sizeof(I)));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(storage.data()),bytes))
        throw std::runtime_error("cannot read input");
    auto original = storage;
    std::vector<O> output(std::max<size_t>(1,size(c.out_shape)));
    I* in = storage.data()+c.offset;
    O* out = output.data();
    if (c.mode == "inplace") {
        if constexpr (std::is_same_v<I,O>) {
            if (c.in_stride != contiguous(c.in_shape))
                throw Unsupported("in-place native adapter requires contiguous input");
            out = in;
        } else throw Unsupported("in-place real FFT needs padded storage");
    }
    const bool destructive = c.mode=="inplace" || c.kind=="irfft";
    auto reset = [&]{ std::copy(original.begin(),original.end(),storage.begin()); };
    auto start = Clock::now();
    auto call = make_call<T>(c,in,out);
    double plan_ns = elapsed(start);
    reset();
    call();
    if (!destructive && storage != original)
        throw std::runtime_error("input modified");
    std::ofstream result(c.outfile,std::ios::binary);
    result.write(reinterpret_cast<const char*>(out),size(c.out_shape)*sizeof(O));
    if (!result) throw std::runtime_error("cannot write output");
    result.close();

    std::vector<double> samples;
    if (c.samples) {
        for (int j=0;j<2;++j) {
            if (destructive) reset();
            call();
        }
        int iterations = 1;
        if (!destructive) {
            start = Clock::now();
            call();
            iterations = std::clamp(int(c.target_ms*1e6/std::max(1.0,elapsed(start))),1,512);
        }
        for (int j=0;j<c.samples;++j) {
            if (destructive) reset();
            start = Clock::now();
            for (int k=0;k<iterations;++k) call();
            samples.push_back(elapsed(start)/iterations);
        }
    }
#if defined(BENCH_DUCC)
    std::cout << "{\"timing_mode\":\"one-shot\",\"planning\":null,\"plan_ns\":null";
#else
    std::cout << "{\"timing_mode\":\"plan-execute\",\"planning\":\""
              << c.planning << "\",\"plan_ns\":" << plan_ns;
#endif
    std::cout << ",\"input_restored\":" << (destructive?"true":"false") << ",\"samples_ns\":[";
    for (size_t j=0;j<samples.size();++j) {
        if (j) std::cout << ",";
        std::cout << std::setprecision(12) << samples[j];
    }
    std::cout << "]}\n";
}

template<class T>
void dispatch(const Config& c) {
    using C = std::complex<T>;
    if (c.kind=="fft" || c.kind=="ifft") run<T,C,C>(c);
    else if (c.kind=="rfft") run<T,T,C>(c);
    else if (c.kind=="irfft") run<T,C,T>(c);
    else run<T,T,T>(c);
}

int main(int argc, char** argv) {
    try {
        if (argc!=17) throw std::runtime_error("expected 16 arguments");
        Config c;
        c.kind=argv[1];
        c.in_shape=parse<Shape>(argv[3]); c.out_shape=parse<Shape>(argv[4]);
        c.axes=parse<Shape>(argv[5]); c.in_stride=parse<Strides>(argv[6]);
        c.offset=std::stoull(argv[7]); c.infile=argv[8]; c.outfile=argv[9];
        c.threads=std::stoull(argv[10]); c.mode=argv[11]; c.scale=argv[12];
        c.type=std::stoi(argv[13]); c.samples=std::stoi(argv[14]);
        c.target_ms=std::stod(argv[15]); c.planning=argv[16];
        std::string precision=argv[2];
        if (precision=="f32") dispatch<float>(c);
        else if (precision=="f64") dispatch<double>(c);
        else if (precision=="ld") dispatch<long double>(c);
        else throw std::runtime_error("invalid precision");
    } catch (const Unsupported& e) {
        std::cerr << e.what() << "\n";
        return 77;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
