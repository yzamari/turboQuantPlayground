// P2 Stage B: end-to-end plumbing test for prepare_keys / release_keys /
// mse_score_pooled.
//
// This test does NOT exercise the OpenCL backend (the host build has
// TQ_WITH_OPENCL=OFF). Instead it builds a mock IBackend that:
//
//   * delegates every kernel to the embedded CPU-scalar backend, so the
//     numerical contract of TurboQuantKVCache::attention_scores() /
//     attend() is identical to the existing kv_cache_test path; and
//   * counts every prepare_keys / release_keys / mse_score_pooled call
//     in atomics, so we can assert the SessionEntry-driven plumbing
//     wires the handle through end-to-end.
//
// We exercise TurboQuantKVCache directly (option (b) in the plan) rather
// than going through attention_turboquant() — that gives us the cleanest
// way to attach a mock backend without adding a test-only injection API
// to the production attention bridge.

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"
#include "tq_test.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace turboquant;

namespace {

// MockBackend wraps a CPU-scalar IBackend and instruments the new
// session-pool plumbing with atomic counters. Every kernel just
// forwards. prepare_keys returns a unique opaque pointer — we use a
// per-instance bump-allocator so the handle is something the test can
// recognize, not an actual GPU buffer.
class MockBackend final : public IBackend {
public:
    MockBackend() : inner_(create_backend(BackendKind::CpuScalar)) {}

    bool init() override {
        // The wrapped CPU-scalar backend is already init()'d by
        // create_backend(); we don't need to do anything else.
        return inner_ != nullptr;
    }

    const char* name() const override { return "mock_pooled"; }

    void rotate(const float* in, const float* Pi, int n, int D, float* out) override {
        inner_->rotate(in, Pi, n, D, out);
    }

    void mse_encode(const float* rotated, const float* boundaries,
                    int N, int D, int bits, void* packed_out) override {
        inner_->mse_encode(rotated, boundaries, N, D, bits, packed_out);
    }

    void mse_score(const float* q_rot, const void* mse_packed,
                   const float* norms, const float* centroids,
                   int BH, int N, int D, int bits, float* out) override {
        ++mse_score_calls;
        inner_->mse_score(q_rot, mse_packed, norms, centroids,
                          BH, N, D, bits, out);
    }

    // Override mse_score_pooled so we can count + record the handle the
    // cache forwarded to us. Then delegate to the unpooled scalar kernel
    // (the contents of mse_packed are identical — pooling is a transport
    // optimization, not a numerical change).
    void mse_score_pooled(const float* q_rot, const void* mse_packed,
                          const float* norms, const float* centroids,
                          int BH, int N, int D, int bits, float* out,
                          void* gpu_key_handle) override {
        ++mse_score_pooled_calls;
        last_handle_seen.store(gpu_key_handle);
        inner_->mse_score(q_rot, mse_packed, norms, centroids,
                          BH, N, D, bits, out);
    }

    void qjl_score(const float* q_sketch, const uint8_t* signs,
                   const float* res_norms, const float* mse_in,
                   int BH, int N, int D, float qjl_scale, float* out) override {
        inner_->qjl_score(q_sketch, signs, res_norms, mse_in,
                          BH, N, D, qjl_scale, out);
    }

    void value_dequant(const uint8_t* packed, const float* scales, const float* zeros,
                       int N, int D, int bits, int group_size, float* out) override {
        inner_->value_dequant(packed, scales, zeros, N, D, bits, group_size, out);
    }

    void* prepare_keys(int BH, int D, int key_bits) override {
        ++prepare_keys_calls;
        last_prepare_shape[0] = BH;
        last_prepare_shape[1] = D;
        last_prepare_shape[2] = key_bits;
        // Hand back a unique non-null sentinel per call. The cache /
        // session layer treats this as an opaque pointer; we just need to
        // be able to recognize it on the way back through
        // mse_score_pooled() and release_keys().
        ++handle_counter;
        void* h = reinterpret_cast<void*>(
            static_cast<uintptr_t>(0x1000ULL) +
            static_cast<uintptr_t>(handle_counter.load()));
        last_issued_handle.store(h);
        return h;
    }

    void release_keys(void* handle) override {
        ++release_keys_calls;
        last_released_handle.store(handle);
    }

    // Public counters — atomic because the IBackend contract allows
    // multi-threaded callers, even though this test is single-threaded.
    std::atomic<int>   prepare_keys_calls    {0};
    std::atomic<int>   release_keys_calls    {0};
    std::atomic<int>   mse_score_calls       {0};
    std::atomic<int>   mse_score_pooled_calls{0};
    std::atomic<int>   handle_counter        {0};
    std::atomic<void*> last_handle_seen      {nullptr};
    std::atomic<void*> last_issued_handle    {nullptr};
    std::atomic<void*> last_released_handle  {nullptr};
    int                last_prepare_shape[3] {0, 0, 0};

private:
    std::unique_ptr<IBackend> inner_;
};

void fill_uniform(std::vector<float>& v, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& x : v) x = u(rng);
}

}  // namespace

int main() {
    // ----- Setup -----
    constexpr int BH       = 2;
    constexpr int D        = 64;
    constexpr int key_bits = 3;
    constexpr int prefill  = 70;     // > buffer_size(64) → forces a flush
    constexpr int decode_n = 4;

    MockBackend mock;
    TQ_CHECK(mock.init());

    auto Pi = generate_pi_qr(D, 42);
    auto S  = generate_qjl_S(D, 1042);

    TurboQuantKVCache::Config cfg;
    cfg.head_dim         = D;
    cfg.key_bits         = key_bits;
    cfg.value_bits       = 2;
    cfg.value_group_size = 32;
    cfg.buffer_size      = 64;
    cfg.layer_idx        = 0;

    TurboQuantKVCache cache(cfg, &mock, std::move(Pi), std::move(S));

    // ----- Drive the SessionEntry-equivalent dance ourselves -----
    //
    // attention_turboquant.cpp's session-cached path does:
    //
    //   handle = backend->prepare_keys(BH, D, key_bits);
    //   cache.set_gpu_key_handle(handle);
    //   ...
    //   eventually: backend->release_keys(handle);
    //
    // We replicate that here so we can verify (a) the cache forwards the
    // handle to mse_score_pooled() instead of mse_score(), (b) the
    // forwarded handle equals the one prepare_keys() handed out, and
    // (c) release_keys() releases the same handle.
    void* handle = mock.prepare_keys(BH, D, key_bits);
    TQ_CHECK(handle != nullptr);
    TQ_CHECK_EQ(mock.prepare_keys_calls.load(), 1);
    TQ_CHECK_EQ(mock.last_prepare_shape[0], BH);
    TQ_CHECK_EQ(mock.last_prepare_shape[1], D);
    TQ_CHECK_EQ(mock.last_prepare_shape[2], key_bits);
    cache.set_gpu_key_handle(handle);
    TQ_CHECK(cache.gpu_key_handle() == handle);

    // ----- Prefill (n_quant_ becomes 6 because prefill > buffer_size) -----
    std::vector<float> keys  (static_cast<size_t>(BH) * prefill * D);
    std::vector<float> values(static_cast<size_t>(BH) * prefill * D);
    fill_uniform(keys,   0xa17e7170ULL);
    fill_uniform(values, 0xb1ce5e7dULL);
    cache.prefill(keys.data(), values.data(), BH, prefill);
    TQ_CHECK_EQ(cache.seq_len(), prefill);

    // ----- One attention call: the cache must call mse_score_pooled,
    //       NOT mse_score, and the handle it forwards must match.
    constexpr int n_q = 1;
    std::vector<float> q(static_cast<size_t>(BH) * n_q * D);
    fill_uniform(q, 0xc09d7011ULL);
    std::vector<float> scores(static_cast<size_t>(BH) * n_q * prefill);

    const int pooled_before = mock.mse_score_pooled_calls.load();
    const int unpooled_before = mock.mse_score_calls.load();
    cache.attention_scores(q.data(), BH, n_q, scores.data());

    TQ_CHECK(mock.mse_score_pooled_calls.load() > pooled_before);
    // No raw mse_score calls — those would mean the handle plumbing was
    // bypassed (Stage A behavior, before this change).
    TQ_CHECK_EQ(mock.mse_score_calls.load(), unpooled_before);
    TQ_CHECK(mock.last_handle_seen.load() == handle);

    // ----- Decode loop: every step's attention call should still flow
    //       through mse_score_pooled, never mse_score.
    for (int step = 0; step < decode_n; ++step) {
        std::vector<float> kt(static_cast<size_t>(BH) * D);
        std::vector<float> vt(static_cast<size_t>(BH) * D);
        fill_uniform(kt, 0xdec0de00ULL + step);
        fill_uniform(vt, 0xdec0de80ULL + step);
        cache.append(kt.data(), vt.data(), BH);

        const int p_before = mock.mse_score_pooled_calls.load();
        const int u_before = mock.mse_score_calls.load();
        cache.attention_scores(q.data(), BH, n_q, scores.data());
        TQ_CHECK(mock.mse_score_pooled_calls.load() > p_before);
        TQ_CHECK_EQ(mock.mse_score_calls.load(), u_before);
        TQ_CHECK(mock.last_handle_seen.load() == handle);
    }

    // ----- prepare_keys is idempotent w.r.t. count for the same shape on
    //       the test side — we never re-prepare in this test. The session
    //       layer also only prepares once per (session_id, layer_il) until
    //       invalidation. Verify we never double-counted.
    TQ_CHECK_EQ(mock.prepare_keys_calls.load(), 1);

    // ----- Release: simulate what release_session_gpu_state does, then
    //       verify the same handle came back to the backend.
    mock.release_keys(handle);
    TQ_CHECK_EQ(mock.release_keys_calls.load(), 1);
    TQ_CHECK(mock.last_released_handle.load() == handle);

    // ----- nullptr handle: the cache without a handle must still work
    //       (the plain TurboQuantKVCache user — host bench, parity test —
    //       leaves the handle nullptr and the default mse_score_pooled
    //       impl falls back to mse_score). Build a fresh cache and
    //       confirm.
    {
        MockBackend mock2;
        TQ_CHECK(mock2.init());
        auto Pi2 = generate_pi_qr(D, 42);
        auto S2  = generate_qjl_S(D, 1042);
        TurboQuantKVCache cache2(cfg, &mock2, std::move(Pi2), std::move(S2));
        cache2.prefill(keys.data(), values.data(), BH, prefill);
        TQ_CHECK(cache2.gpu_key_handle() == nullptr);
        cache2.attention_scores(q.data(), BH, n_q, scores.data());
        // mse_score_pooled is still called (it's the new dispatch
        // surface), but the handle threaded through is null — proving
        // the no-pool fallback path works end-to-end.
        TQ_CHECK(mock2.mse_score_pooled_calls.load() > 0);
        TQ_CHECK(mock2.last_handle_seen.load() == nullptr);
        TQ_CHECK_EQ(mock2.prepare_keys_calls.load(), 0);
        TQ_CHECK_EQ(mock2.release_keys_calls.load(), 0);
    }

    return tq_test::report_and_exit();
}
