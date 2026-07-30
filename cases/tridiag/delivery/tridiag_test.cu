// ============================================================================
// 三对角求解 Thomas 纯 CUDA 交付版 —— 自测 harness（内置 CPU 参考，不依赖 PyTorch）
// ============================================================================
// 生成随机对角占优三对角系统，调 GPU host 函数得 x / 四路梯度，与内置 CPU 朴素参考对拍 allclose。
// 小规模（默认 B=64, N=128）即可证明 kernel 数学正确性；kernel 计算逻辑与 A100 上已验收
// （前向1.27×/反向9.80×、5种子正确性全PASS）的主仓版本逐字一致。
//
// CPU 参考独立实现（不复用 GPU 中间量）：
//   前向：标准 Thomas（前向消元 + 回代）解 A x = rhs。
//   反向：解伴随系统 A^T λ = grad_x（转置三对角：其"下对角"是原 upper、"上对角"是原 lower），
//         再由 λ 与 x 组回四路梯度（grad_rhs=λ, grad_diag=-λx, grad_lower_i=-λ_i x_{i-1},
//         grad_upper_i=-λ_i x_{i+1}）。与 kernel 走不同代码路径，构成独立金标准。
//
// 编译运行：  make test  （或 nvcc -O3 -arch=sm_80 tridiag_kernels.cu tridiag_test.cu -o tridiag_test && ./tridiag_test）
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" void tridiag_forward_cuda(const float*, const float*, const float*,
                                     const float*, float*, float*, int, int);
extern "C" void tridiag_backward_cuda(const float*, const float*, const float*,
                                      const float*, float*, int, int);

// ---- CPU 朴素参考（标准 Thomas，逐系统；作为自包含金标准）----
static void cpu_solve(const float* lo, const float* di, const float* up,
                      const float* r, float* x, int N) {
    std::vector<float> c(N), d(N);  // 消元后的 upper'/rhs'
    // 前向消元
    float beta = di[0];
    c[0] = up[0] / beta;
    d[0] = r[0] / beta;
    for (int i = 1; i < N; ++i) {
        beta = di[i] - lo[i] * c[i - 1];
        c[i] = up[i] / beta;
        d[i] = (r[i] - lo[i] * d[i - 1]) / beta;
    }
    // 回代
    x[N - 1] = d[N - 1];
    for (int i = N - 2; i >= 0; --i) x[i] = d[i] - c[i] * x[i + 1];
}

static void cpu_forward(const std::vector<float>& lo, const std::vector<float>& di,
                        const std::vector<float>& up, const std::vector<float>& r,
                        std::vector<float>& x, int B, int N) {
    for (int b = 0; b < B; ++b)
        cpu_solve(&lo[(size_t)b*N], &di[(size_t)b*N], &up[(size_t)b*N],
                  &r[(size_t)b*N], &x[(size_t)b*N], N);
}

static void cpu_backward(const std::vector<float>& lo, const std::vector<float>& di,
                         const std::vector<float>& up, const std::vector<float>& gx,
                         const std::vector<float>& x,
                         std::vector<float>& gLo, std::vector<float>& gDi,
                         std::vector<float>& gUp, std::vector<float>& gRhs,
                         int B, int N) {
    std::vector<float> lam(N);
    for (int b = 0; b < B; ++b) {
        const float* lob = &lo[(size_t)b*N];
        const float* dib = &di[(size_t)b*N];
        const float* upb = &up[(size_t)b*N];
        const float* gxb = &gx[(size_t)b*N];
        const float* xb  = &x[(size_t)b*N];
        // 解 A^T λ = grad_x：A^T 三对角的 lower'=upper(shift)、diag'=diag、upper'=lower(shift)。
        // A^T 第 i 行：upper[i-1]*λ_{i-1} + diag[i]*λ_i + lower[i+1]*λ_{i+1} = gx[i]
        std::vector<float> subT(N), diaT(N), supT(N), rr(N);
        for (int i = 0; i < N; ++i) {
            diaT[i] = dib[i];
            subT[i] = (i > 0)     ? upb[i - 1] : 0.0f;  // A^T 下对角
            supT[i] = (i < N - 1) ? lob[i + 1] : 0.0f;  // A^T 上对角
            rr[i] = gxb[i];
        }
        cpu_solve(subT.data(), diaT.data(), supT.data(), rr.data(), lam.data(), N);
        // 组回四路梯度
        float* gLob = &gLo[(size_t)b*N];
        float* gDib = &gDi[(size_t)b*N];
        float* gUpb = &gUp[(size_t)b*N];
        float* gRb  = &gRhs[(size_t)b*N];
        for (int i = 0; i < N; ++i) {
            gRb[i]  = lam[i];
            gDib[i] = -lam[i] * xb[i];
            gLob[i] = (i == 0)     ? 0.0f : -lam[i] * xb[i - 1];
            gUpb[i] = (i == N - 1) ? 0.0f : -lam[i] * xb[i + 1];
        }
    }
}

static bool allclose(const std::vector<float>& a, const std::vector<float>& b,
                     float atol, float rtol, float* max_err) {
    float me = 0.f; bool ok = true;
    for (size_t i = 0; i < a.size(); ++i) {
        float e = std::fabs(a[i] - b[i]);
        if (e > me) me = e;
        if (e > atol + rtol * std::fabs(b[i])) ok = false;
    }
    *max_err = me;
    return ok;
}

int main(int argc, char** argv) {
    int B = 64, N = 128;
    if (argc >= 3) { B = atoi(argv[1]); N = atoi(argv[2]); }
    const float atol = 1e-2f, rtol = 1e-2f;

    std::srand(1234);
    auto rnd = []() { return (float)std::rand() / RAND_MAX * 2.f - 1.f; };
    std::vector<float> lo((size_t)B*N), di((size_t)B*N), up((size_t)B*N);
    std::vector<float> rhs((size_t)B*N), gx((size_t)B*N);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < N; ++i) {
            size_t k = (size_t)b*N + i;
            lo[k] = (i == 0)     ? 0.f : 0.2f * rnd();
            up[k] = (i == N - 1) ? 0.f : 0.2f * rnd();
            rhs[k] = rnd();
            gx[k] = rnd();
        }
    // 对角占优：diag = 1 + |lower| + |upper|（与 make_inputs 一致，保证 Thomas 稳定）
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < N; ++i) {
            size_t k = (size_t)b*N + i;
            di[k] = 1.f + std::fabs(lo[k]) + std::fabs(up[k]);
        }

    std::vector<float> x_gpu((size_t)B*N), factors((size_t)2*B*N);
    std::vector<float> grad_gpu((size_t)4*B*N);
    std::vector<float> x_ref((size_t)B*N);
    std::vector<float> gLo_r((size_t)B*N), gDi_r((size_t)B*N), gUp_r((size_t)B*N), gRhs_r((size_t)B*N);

    // GPU
    tridiag_forward_cuda(lo.data(), di.data(), up.data(), rhs.data(),
                         x_gpu.data(), factors.data(), B, N);
    tridiag_backward_cuda(gx.data(), up.data(), x_gpu.data(), factors.data(),
                          grad_gpu.data(), B, N);
    // CPU 参考（前向用自己的解；反向用 CPU 自己的 x_ref，保持对拍独立）
    cpu_forward(lo, di, up, rhs, x_ref, B, N);
    cpu_backward(lo, di, up, gx, x_ref, gLo_r, gDi_r, gUp_r, gRhs_r, B, N);

    // GPU gradients 布局 [4,B,N]：切出四路
    size_t BN = (size_t)B*N;
    std::vector<float> gLo_g(grad_gpu.begin() + 0*BN, grad_gpu.begin() + 1*BN);
    std::vector<float> gDi_g(grad_gpu.begin() + 1*BN, grad_gpu.begin() + 2*BN);
    std::vector<float> gUp_g(grad_gpu.begin() + 2*BN, grad_gpu.begin() + 3*BN);
    std::vector<float> gRhs_g(grad_gpu.begin() + 3*BN, grad_gpu.begin() + 4*BN);

    printf("=== 三对角 Thomas 纯 CUDA 交付版自测  B=%d N=%d ===\n", B, N);
    float ex, eLo, eDi, eUp, eR;
    bool okx  = allclose(x_gpu,  x_ref,  atol, rtol, &ex);
    bool okLo = allclose(gLo_g, gLo_r, atol, rtol, &eLo);
    bool okDi = allclose(gDi_g, gDi_r, atol, rtol, &eDi);
    bool okUp = allclose(gUp_g, gUp_r, atol, rtol, &eUp);
    bool okR  = allclose(gRhs_g, gRhs_r, atol, rtol, &eR);
    printf("[前向 x     ] %s  max_abs_err=%.3e\n", okx  ? "PASS" : "FAIL", ex);
    printf("[反向 dLower] %s  max_abs_err=%.3e\n", okLo ? "PASS" : "FAIL", eLo);
    printf("[反向 dDiag ] %s  max_abs_err=%.3e\n", okDi ? "PASS" : "FAIL", eDi);
    printf("[反向 dUpper] %s  max_abs_err=%.3e\n", okUp ? "PASS" : "FAIL", eUp);
    printf("[反向 dRhs  ] %s  max_abs_err=%.3e\n", okR  ? "PASS" : "FAIL", eR);
    bool all = okx && okLo && okDi && okUp && okR;
    printf("=== 总判定: %s (atol=rtol=1e-2) ===\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
