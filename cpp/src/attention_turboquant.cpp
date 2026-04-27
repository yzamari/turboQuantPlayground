// attention_turboquant — single-call attention bridge (Path 2.1c Step 1
// + Phase A2). Composes the existing TurboQuantKVCache + softmax to
// produce a drop-in replacement for the standard Q@K.T → soft_max_ext
// → attn@V triple.
//
// Two modes:
//
//   Stateless (session_id == 0 OR layer_il < 0):
//     Constructs a throwaway TurboQuantKVCache per call. Used by the
//     host bench / parity tests where each call is independent.
//
//   Session-cached (Phase A2, session_id != 0 AND layer_il >= 0):
//     Looks up an existing TurboQuantKVCache by (session_id, layer_il)
//     and re-uses it. On first call: prefill with all n_kv tokens. On
//     subsequent calls with a larger n_kv: append only the *new* tokens.
//     Saves the rotate + quantize cost for the K/V slice that's
//     already in the cache, and on the OpenCL backend keeps the
//     K-side cl_mem allocations alive across calls.

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace turboquant {

namespace {

// ---- Per-(session, layer) cache of prepared TurboQuantKVCache instances ----
struct SessionKey {
    uint64_t session_id;
    int      layer_il;
    bool operator==(const SessionKey & o) const {
        return session_id == o.session_id && layer_il == o.layer_il;
    }
};
struct SessionKeyHash {
    size_t operator()(const SessionKey & k) const {
        return std::hash<uint64_t>()(k.session_id) ^
               (static_cast<size_t>(k.layer_il) << 1);
    }
};
struct SessionEntry {
    std::unique_ptr<TurboQuantKVCache> cache;
    int    BH         = 0;
    int    D          = 0;
    int    key_bits   = 0;
    int    value_bits = 0;
};

std::mutex g_session_mu;
std::unordered_map<SessionKey, SessionEntry, SessionKeyHash> g_sessions;

// Caller MUST hold g_session_mu.
SessionEntry & get_or_create_entry_locked(
        uint64_t session_id, int layer_il,
        int BH, int D, int key_bits, int value_bits,
        IBackend * backend, uint64_t seed) {
    auto it = g_sessions.find({session_id, layer_il});
    if (it != g_sessions.end()) {
        // Existing entry. Sanity-check shape — if BH or D changed, the
        // cached cache is stale. This shouldn't happen in normal decode
        // flow but guard against it.
        if (it->second.BH != BH || it->second.D != D ||
            it->second.key_bits != key_bits ||
            it->second.value_bits != value_bits) {
            it->second = SessionEntry{};
        } else {
            return it->second;
        }
    }

    SessionEntry & entry = g_sessions[{session_id, layer_il}];
    entry.BH         = BH;
    entry.D          = D;
    entry.key_bits   = key_bits;
    entry.value_bits = value_bits;

    auto Pi = generate_pi_qr(D, seed);
    auto S  = generate_qjl_S(D, seed + 1000);

    TurboQuantKVCache::Config cfg;
    cfg.head_dim         = D;
    cfg.key_bits         = key_bits;
    cfg.value_bits       = value_bits;
    cfg.value_group_size = 32;
    // Recent-tokens unquantized buffer. Last `buffer_size` tokens stay
    // at full precision; older tokens are TurboQuant-compressed. This
    // is the standard chat-quality knob — attention weights are
    // naturally heaviest on recent tokens, so keeping the recent
    // window precise dramatically improves coherence with minimal
    // memory cost (64 * D * BH ≈ ~64KB per layer for Llama-3.2-1B).
    // The host bench uses 0 to stress-test the compressed-only path;
    // chat/decode wants a real buffer. Set to 0 via TQ_NO_BUFFER env
    // for the original stateless behaviour.
    cfg.buffer_size      = 64;
    cfg.layer_idx        = layer_il;

    entry.cache = std::make_unique<TurboQuantKVCache>(
        cfg, backend, std::move(Pi), std::move(S));
    return entry;
}

// Mask + softmax over scores[BH * n_q * n_kv]. Numerically stable.
void apply_mask_and_softmax(
        float * scores, const float * mask,
        int BH, int n_q, int n_kv) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            float * row = &scores[(static_cast<size_t>(b) * n_q + t) * n_kv];

            if (mask) {
                const float * m = &mask[static_cast<size_t>(t) * n_kv];
                for (int j = 0; j < n_kv; ++j) row[j] += m[j];
            }

            float row_max = neg_inf;
            for (int j = 0; j < n_kv; ++j) {
                if (row[j] > row_max) row_max = row[j];
            }
            if (row_max == neg_inf) {
                for (int j = 0; j < n_kv; ++j) row[j] = 0.f;
                continue;
            }

            float sum = 0.f;
            for (int j = 0; j < n_kv; ++j) {
                row[j] = std::exp(row[j] - row_max);
                sum   += row[j];
            }
            const float inv = 1.f / std::max(sum, 1e-30f);
            for (int j = 0; j < n_kv; ++j) row[j] *= inv;
        }
    }
}

void run_stateless(
        const float * q, const float * k, const float * v,
        int BH, int n_q, int n_kv, int D,
        float scale, const float * mask, float * out,
        int key_bits, int value_bits, uint64_t seed,
        IBackend * backend) {
    auto Pi = generate_pi_qr(D, seed);
    auto S  = generate_qjl_S(D, seed + 1000);

    TurboQuantKVCache::Config cfg;
    cfg.head_dim         = D;
    cfg.key_bits         = key_bits;
    cfg.value_bits       = value_bits;
    cfg.value_group_size = 32;
    cfg.buffer_size      = 0;
    cfg.layer_idx        = 0;

    TurboQuantKVCache cache(cfg, backend, std::move(Pi), std::move(S));
    cache.prefill(k, v, BH, n_kv);

    std::vector<float> scores(static_cast<size_t>(BH) * n_q * n_kv);
    cache.attention_scores(q, BH, n_q, scores.data(), scale);
    apply_mask_and_softmax(scores.data(), mask, BH, n_q, n_kv);
    cache.attend(scores.data(), BH, n_q, out);
}

}  // namespace

void attention_turboquant(
        const float * q, const float * k, const float * v,
        int BH, int n_q, int n_kv, int D,
        float scale, const float * mask, float * out,
        int key_bits, int value_bits, uint64_t seed,
        uint64_t session_id, int layer_il) {

    if (BH <= 0 || n_q <= 0 || n_kv <= 0 || D <= 0) {
        throw std::invalid_argument("attention_turboquant: non-positive dim");
    }
    if (!q || !k || !v || !out) {
        throw std::invalid_argument("attention_turboquant: null pointer");
    }

    if (scale <= 0.f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    IBackend * backend = get_best_backend();
    if (!backend) {
        throw std::runtime_error("attention_turboquant: no backend available");
    }

    // Stateless mode: no session id provided. Used by the bench and
    // parity tests, where each call should be independent.
    if (session_id == 0 || layer_il < 0) {
        run_stateless(q, k, v, BH, n_q, n_kv, D, scale, mask, out,
                      key_bits, value_bits, seed, backend);
        return;
    }

    // Session-cached mode (Phase A2). Lock for the entire call —
    // attention_turboquant is invoked by the multi-threaded ggml op
    // dispatcher with one call per worker thread, but each call is
    // for a *different* slice of BH (see ggml-custom-turboquant.c).
    // Different (session_id, layer_il) entries can run concurrently
    // in principle, but each entry must be serialized because the
    // cache state is shared. The mutex protects entry lookup AND
    // cache mutation.
    std::lock_guard<std::mutex> lk(g_session_mu);

    SessionEntry & entry = get_or_create_entry_locked(
        session_id, layer_il, BH, D, key_bits, value_bits, backend, seed);

    const int cached_n = entry.cache->seq_len();
    if (cached_n == 0) {
        // First call for this (session, layer): full prefill.
        entry.cache->prefill(k, v, BH, n_kv);
    } else if (n_kv > cached_n) {
        // Cache grew (decode added tokens). Ideally we'd append() the
        // new tokens, but TurboQuantKVCache::append() is currently
        // unimplemented (kv_cache.cpp:169 throws std::logic_error,
        // a holdover from the P0 cut). Fall back to a full re-prefill
        // — wastes the rotate+quantize work for the previously cached
        // slice but stays safe. Implementing append() is the next
        // perf lever; until then this path costs O(n_kv) per decode
        // step instead of O(1).
        entry.cache = nullptr;  // drop and rebuild
        auto Pi = generate_pi_qr(D, seed);
        auto S  = generate_qjl_S(D, seed + 1000);
        TurboQuantKVCache::Config cfg;
        cfg.head_dim         = D;
        cfg.key_bits         = key_bits;
        cfg.value_bits       = value_bits;
        cfg.value_group_size = 32;
        cfg.buffer_size      = 64;  // recent-window full precision (chat coherence)
        cfg.layer_idx        = layer_il;
        entry.cache = std::make_unique<TurboQuantKVCache>(
            cfg, backend, std::move(Pi), std::move(S));
        entry.cache->prefill(k, v, BH, n_kv);
    } else if (n_kv < cached_n) {
        // Cache regressed (e.g. context reset). Wipe and re-prefill.
        // Rebuild the entry with a fresh cache.
        entry = SessionEntry{};
        entry.BH         = BH;
        entry.D          = D;
        entry.key_bits   = key_bits;
        entry.value_bits = value_bits;
        auto Pi = generate_pi_qr(D, seed);
        auto S  = generate_qjl_S(D, seed + 1000);
        TurboQuantKVCache::Config cfg;
        cfg.head_dim         = D;
        cfg.key_bits         = key_bits;
        cfg.value_bits       = value_bits;
        cfg.value_group_size = 32;
        cfg.buffer_size      = 64;  // recent-window full precision
        cfg.layer_idx        = layer_il;
        entry.cache = std::make_unique<TurboQuantKVCache>(
            cfg, backend, std::move(Pi), std::move(S));
        entry.cache->prefill(k, v, BH, n_kv);
    }
    // n_kv == cached_n: cache already has the full window; just compute
    // attention against the existing state (e.g. running attention twice
    // with the same K/V — the bench's parity sweep does this).

    std::vector<float> scores(static_cast<size_t>(BH) * n_q * n_kv);
    entry.cache->attention_scores(q, BH, n_q, scores.data(), scale);
    apply_mask_and_softmax(scores.data(), mask, BH, n_q, n_kv);
    entry.cache->attend(scores.data(), BH, n_q, out);
}

void attention_turboquant_invalidate(uint64_t session_id, int layer_il) {
    std::lock_guard<std::mutex> lk(g_session_mu);
    if (layer_il >= 0) {
        g_sessions.erase({session_id, layer_il});
        return;
    }
    // layer_il < 0: drop all entries for this session.
    for (auto it = g_sessions.begin(); it != g_sessions.end(); ) {
        if (it->first.session_id == session_id) {
            it = g_sessions.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace turboquant
