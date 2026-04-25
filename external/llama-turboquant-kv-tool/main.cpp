// llama-turboquant-kv
// ---------------------------------------------------------------------------
// Proof-of-concept tool that wires the TurboQuant KV-cache compression
// library (https://github.com/.../turboQuantPlayground) into a real
// llama.cpp run.
//
// What it does
// ------------
//   1. Loads a GGUF model + tokenizes a prompt and runs a real prefill
//      through llama.cpp.
//   2. Queries the model for its KV-cache geometry
//      (n_layer, n_head_kv, head_dim) and reads the *actual* serialized
//      KV-cache size from llama_state_seq_get_size — the ground-truth
//      "fp16 baseline memory" number for this model + this prompt.
//   3. Constructs shape-matched random K/V tensors with the same layout
//      Llama actually uses ([B*H_kv, S, D] per layer) and feeds them
//      through TurboQuantKVCache (key_bits=3, value_bits=2). This is the
//      "TurboQuant cache" — it produces real compressed bytes via the
//      same code path the production library exposes.
//   4. Computes attention against both caches with a synthetic query and
//      compares: cosine similarity + attention-output relative L2.
//   5. Prints a CSV row per layer plus a summary; designed to be `adb
//      push`ed and run on-device.
//
// Why "shape-matched synthetic" K/V instead of the live tensors
// -------------------------------------------------------------
// llama.cpp's public C API does not expose raw K/V tensor pointers.  The
// state buffer returned by llama_state_seq_get_data is opaque and
// version-dependent. Replacing the cache wholesale would require
// subclassing the internal llama_kv_cache_unified type — out of scope
// for a lowest-risk, single-session integration.  This tool therefore
// proves the *integration shape* (correct dimensions, on-device
// performance, ratio + quality numbers) which is exactly what the user
// asked for as the v0 deliverable.  See docs/llamacpp-integration.md for
// the full-replacement path.
// ---------------------------------------------------------------------------

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

double cosine_sim(const float * a, const float * b, std::size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na * nb) + 1e-12);
}

double rel_l2(const float * tq, const float * ref, std::size_t n) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(tq[i]) - ref[i];
        num += d * d;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / (den + 1e-12));
}

void softmax_inplace(float * x, std::size_t n) {
    float mx = x[0];
    for (std::size_t i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::expf(x[i] - mx);
        sum += x[i];
    }
    float inv = static_cast<float>(1.0 / (sum + 1e-12));
    for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
}

// Reference fp16-equivalent attention scores against fp32 K (we treat the
// "baseline" as fp32 cache; the fp16 memory number is reported separately
// by simply halving the byte count, which is what llama.cpp uses).
void baseline_attention_scores(const float * keys, const float * query,
                               int BH, int n_q, int seq_len, int D,
                               float scale, float * out) {
    for (int bh = 0; bh < BH; ++bh) {
        for (int q = 0; q < n_q; ++q) {
            const float * qv = query + (bh * n_q + q) * D;
            float       * o  = out   + (bh * n_q + q) * seq_len;
            for (int s = 0; s < seq_len; ++s) {
                const float * kv = keys + (bh * seq_len + s) * D;
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += qv[d] * kv[d];
                o[s] = static_cast<float>(dot) * scale;
            }
        }
    }
}

void baseline_attend(const float * weights, const float * values,
                     int BH, int n_q, int seq_len, int D, float * out) {
    for (int bh = 0; bh < BH; ++bh) {
        for (int q = 0; q < n_q; ++q) {
            const float * w = weights + (bh * n_q + q) * seq_len;
            float       * o = out     + (bh * n_q + q) * D;
            std::memset(o, 0, D * sizeof(float));
            for (int s = 0; s < seq_len; ++s) {
                const float * vv = values + (bh * seq_len + s) * D;
                float ww = w[s];
                for (int d = 0; d < D; ++d) o[d] += ww * vv[d];
            }
        }
    }
}

struct LayerRow {
    int layer;
    int seq_len;
    int BH;
    int D;
    std::size_t baseline_fp16_bytes;
    std::size_t baseline_fp32_bytes;
    std::size_t tq_bytes;
    double      compression_ratio_fp16;
    double      compression_ratio_fp32;
    double      cos_scores;
    double      cos_weights;
    double      rel_l2_output;
    double      encode_ms;
    double      attn_ms_baseline;
    double      attn_ms_tq;
};

void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nllama-turboquant-kv: integrate TurboQuant KV-cache compression with llama.cpp.\n");
    LOG("\n  basic:        %s -m model.gguf -p \"hello world\" -c 512\n", argv[0]);
    LOG("\n  csv output:   %s -m model.gguf -p \"...\"  --tq-csv /sdcard/turboquant_kv.csv\n", argv[0]);
    LOG("\noptions:\n");
    LOG("  --tq-csv <path>           write per-layer CSV here\n");
    LOG("  --tq-key-bits N           keys quant bits (default 3)\n");
    LOG("  --tq-value-bits N         values quant bits (default 2)\n");
    LOG("  --tq-only-shape           only print KV geometry then exit\n");
    LOG("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "The quick brown fox jumps over the lazy dog. " \
                    "TurboQuant compresses the KV cache to ~4x while " \
                    "preserving attention quality.";
    params.n_predict = 0;        // prefill only
    params.kv_unified = true;

    // Custom flags we strip before delegating to common_params_parse.
    std::string csv_path;
    int  key_bits   = 3;
    int  value_bits = 2;
    bool only_shape = false;

    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--tq-csv" && i + 1 < argc) { csv_path = argv[++i]; continue; }
        if (k == "--tq-key-bits" && i + 1 < argc) { key_bits = std::atoi(argv[++i]); continue; }
        if (k == "--tq-value-bits" && i + 1 < argc) { value_bits = std::atoi(argv[++i]); continue; }
        if (k == "--tq-only-shape") { only_shape = true; continue; }
        filtered_argv.push_back(argv[i]);
    }
    int    new_argc = static_cast<int>(filtered_argv.size());
    char **new_argv = filtered_argv.data();

    common_init();

    if (!common_params_parse(new_argc, new_argv, params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    // ---- Load model + run a real prefill ---------------------------------
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        std::fprintf(stderr, "failed to init llama model\n");
        return 1;
    }

    const int n_layer  = llama_model_n_layer(model);
    const int n_head_kv= llama_model_n_head_kv(model);
    const int n_embd   = llama_model_n_embd(model);
    const int n_head   = llama_model_n_head(model);
    const int head_dim = n_head > 0 ? (n_embd / n_head) : 0;

    std::printf("\n=== llama-turboquant-kv ===\n");
    std::printf("model:      %s\n", params.model.path.c_str());
    std::printf("n_layer:    %d\n", n_layer);
    std::printf("n_head:     %d\n", n_head);
    std::printf("n_head_kv:  %d\n", n_head_kv);
    std::printf("n_embd:     %d\n", n_embd);
    std::printf("head_dim:   %d\n", head_dim);

    if (only_shape) {
        return 0;
    }

    // Tokenize and prefill.
    auto tokens = common_tokenize(ctx, params.prompt, true);
    if (tokens.empty()) {
        std::fprintf(stderr, "tokenization produced 0 tokens\n");
        return 1;
    }
    const int seq_len = static_cast<int>(tokens.size());
    std::printf("prompt:     \"%s\"\n", params.prompt.c_str());
    std::printf("seq_len:    %d tokens\n", seq_len);

    // Decode the prompt.
    {
        llama_batch b = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, b) != 0) {
            std::fprintf(stderr, "llama_decode failed\n");
            return 1;
        }
    }

    // Real KV cache size as llama.cpp sees it for sequence 0.
    const std::size_t llama_seq_bytes = llama_state_seq_get_size(ctx, 0);
    std::printf("llama state_seq_size: %.2f MB (seq 0, after prefill)\n",
                llama_seq_bytes / (1024.0 * 1024.0));

    // Per-layer fp16 KV bytes the kernel actually allocates at this seq_len.
    // 2 = K + V; n_head_kv * head_dim values per token; 2 bytes each (fp16).
    const std::size_t fp16_per_layer =
        static_cast<std::size_t>(2) * n_head_kv * head_dim * seq_len * sizeof(uint16_t);
    const std::size_t fp16_total = static_cast<std::size_t>(n_layer) * fp16_per_layer;
    std::printf("fp16 KV (this seq_len): %.2f MB total, %.2f MB per layer\n",
                fp16_total / (1024.0 * 1024.0),
                fp16_per_layer / (1024.0 * 1024.0));

    // ---- Run TurboQuant on shape-matched K/V per layer -------------------
    auto backend = turboquant::create_best_backend();
    if (!backend) {
        std::fprintf(stderr, "no turboquant backend available\n");
        return 1;
    }
    std::printf("turboquant backend: %s   %s\n",
                backend->name(), turboquant::version_string());

    if (!turboquant::get_codebook(head_dim, key_bits)) {
        std::fprintf(stderr,
                     "no embedded codebook for d=%d, bits=%d — re-build "
                     "turboquant with the matching codebook JSON.\n",
                     head_dim, key_bits);
        return 1;
    }

    // Synthesize K/V with a layer-dependent RNG. Each layer gets its own
    // Pi/S, just like the production library does.
    const int BH  = n_head_kv;            // single batch, GQA group count
    const int n_q = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<LayerRow> rows;
    rows.reserve(n_layer);

    std::printf("\nlayer  seq_len  fp16/L MB  tq/L MB  ratio_fp16  cos_scores  cos_weights  rel_l2_out  enc_ms  base_ms  tq_ms\n");
    std::printf("-----  -------  ---------  -------  ----------  ----------  -----------  ----------  ------  -------  ------\n");

    for (int layer = 0; layer < n_layer; ++layer) {
        const std::size_t n_kv = static_cast<std::size_t>(BH) * seq_len * head_dim;
        std::vector<float> keys(n_kv), values(n_kv);
        std::vector<float> query(static_cast<std::size_t>(BH) * n_q * head_dim);
        std::mt19937 rng(0xA5A5u + layer * 7919u);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : keys  ) v = nd(rng);
        for (auto & v : values) v = nd(rng);
        for (auto & v : query ) v = nd(rng);

        // Baseline timing + scores/output.
        std::vector<float> base_scores (static_cast<std::size_t>(BH) * n_q * seq_len);
        std::vector<float> base_weights(base_scores.size());
        std::vector<float> base_out    (static_cast<std::size_t>(BH) * n_q * head_dim);

        auto t0 = std::chrono::steady_clock::now();
        baseline_attention_scores(keys.data(), query.data(), BH, n_q, seq_len, head_dim,
                                  scale, base_scores.data());
        auto t1 = std::chrono::steady_clock::now();
        base_weights = base_scores;
        for (int b = 0; b < BH; ++b) {
            for (int q = 0; q < n_q; ++q) {
                softmax_inplace(base_weights.data() + (b * n_q + q) * seq_len, seq_len);
            }
        }
        baseline_attend(base_weights.data(), values.data(), BH, n_q, seq_len, head_dim,
                        base_out.data());

        // TurboQuant cache.
        auto Pi = turboquant::generate_pi_qr(head_dim, /*seed=*/42 + layer);
        auto S  = turboquant::generate_qjl_S (head_dim, /*seed=*/1042 + layer);
        turboquant::TurboQuantKVCache::Config cfg;
        cfg.head_dim         = head_dim;
        cfg.key_bits         = key_bits;
        cfg.value_bits       = value_bits;
        cfg.value_group_size = 32;
        cfg.buffer_size      = 0;
        cfg.layer_idx        = layer;
        turboquant::TurboQuantKVCache tq(cfg, backend.get(), Pi, S);

        auto e0 = std::chrono::steady_clock::now();
        tq.prefill(keys.data(), values.data(), BH, seq_len);
        auto e1 = std::chrono::steady_clock::now();

        std::vector<float> tq_scores (base_scores.size());
        std::vector<float> tq_weights(base_scores.size());
        std::vector<float> tq_out    (base_out.size());
        auto u0 = std::chrono::steady_clock::now();
        tq.attention_scores(query.data(), BH, n_q, tq_scores.data(), scale);
        auto u1 = std::chrono::steady_clock::now();
        tq_weights = tq_scores;
        for (int b = 0; b < BH; ++b) {
            for (int q = 0; q < n_q; ++q) {
                softmax_inplace(tq_weights.data() + (b * n_q + q) * seq_len, seq_len);
            }
        }
        tq.attend(tq_weights.data(), BH, n_q, tq_out.data());

        LayerRow r;
        r.layer  = layer;
        r.seq_len = seq_len;
        r.BH     = BH;
        r.D      = head_dim;
        r.baseline_fp32_bytes = n_kv * 2 * sizeof(float);
        r.baseline_fp16_bytes = n_kv * 2 * sizeof(uint16_t);
        r.tq_bytes            = tq.memory_bytes();
        r.compression_ratio_fp16 = (r.tq_bytes > 0)
            ? static_cast<double>(r.baseline_fp16_bytes) / r.tq_bytes : 0.0;
        r.compression_ratio_fp32 = (r.tq_bytes > 0)
            ? static_cast<double>(r.baseline_fp32_bytes) / r.tq_bytes : 0.0;
        r.cos_scores = cosine_sim(tq_scores.data(),  base_scores.data(),  base_scores.size());
        r.cos_weights= cosine_sim(tq_weights.data(), base_weights.data(), base_weights.size());
        r.rel_l2_output = rel_l2(tq_out.data(), base_out.data(), base_out.size());
        r.encode_ms      = std::chrono::duration<double, std::milli>(e1 - e0).count();
        r.attn_ms_baseline = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.attn_ms_tq     = std::chrono::duration<double, std::milli>(u1 - u0).count();
        rows.push_back(r);

        std::printf("%5d  %7d  %9.3f  %7.3f  %10.2fx %10.4f  %11.4f  %10.4f  %6.2f  %7.2f  %6.2f\n",
                    r.layer, r.seq_len,
                    r.baseline_fp16_bytes / (1024.0 * 1024.0),
                    r.tq_bytes / (1024.0 * 1024.0),
                    r.compression_ratio_fp16,
                    r.cos_scores, r.cos_weights, r.rel_l2_output,
                    r.encode_ms, r.attn_ms_baseline, r.attn_ms_tq);
    }

    // Aggregate.
    std::size_t fp16_sum = 0, tq_sum = 0;
    double cos_scores_sum = 0, cos_weights_sum = 0, rel_l2_sum = 0;
    double enc_sum = 0, base_attn_sum = 0, tq_attn_sum = 0;
    for (const auto & r : rows) {
        fp16_sum        += r.baseline_fp16_bytes;
        tq_sum          += r.tq_bytes;
        cos_scores_sum  += r.cos_scores;
        cos_weights_sum += r.cos_weights;
        rel_l2_sum      += r.rel_l2_output;
        enc_sum         += r.encode_ms;
        base_attn_sum   += r.attn_ms_baseline;
        tq_attn_sum     += r.attn_ms_tq;
    }
    const double n = static_cast<double>(rows.size());
    std::printf("\n=== summary (%d layers @ seq_len=%d, head_dim=%d, BH=%d, key_bits=%d, value_bits=%d) ===\n",
                (int) rows.size(), seq_len, head_dim, BH, key_bits, value_bits);
    std::printf("total fp16 KV bytes  : %.2f MB\n", fp16_sum / (1024.0 * 1024.0));
    std::printf("total TurboQuant KV  : %.2f MB\n", tq_sum  / (1024.0 * 1024.0));
    std::printf("compression ratio    : %.2fx (vs fp16)\n",
                tq_sum > 0 ? static_cast<double>(fp16_sum) / tq_sum : 0.0);
    std::printf("avg cosine(scores)   : %.4f\n", cos_scores_sum  / n);
    std::printf("avg cosine(weights)  : %.4f\n", cos_weights_sum / n);
    std::printf("avg rel_l2(output)   : %.4f\n", rel_l2_sum / n);
    std::printf("avg encode time      : %.2f ms / layer\n", enc_sum / n);
    std::printf("avg baseline attn    : %.2f ms / layer\n", base_attn_sum / n);
    std::printf("avg turboquant attn  : %.2f ms / layer\n", tq_attn_sum / n);
    std::printf("llama serialized seq : %.2f MB (reference, contains K+V+positions)\n",
                llama_seq_bytes / (1024.0 * 1024.0));

    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (out) {
            out << "layer,seq_len,BH,D,fp16_bytes,fp32_bytes,tq_bytes,"
                   "ratio_fp16,ratio_fp32,cos_scores,cos_weights,rel_l2_output,"
                   "encode_ms,attn_ms_baseline,attn_ms_tq\n";
            for (const auto & r : rows) {
                out << r.layer << ',' << r.seq_len << ',' << r.BH << ',' << r.D << ','
                    << r.baseline_fp16_bytes << ',' << r.baseline_fp32_bytes << ','
                    << r.tq_bytes << ','
                    << r.compression_ratio_fp16 << ',' << r.compression_ratio_fp32 << ','
                    << r.cos_scores << ',' << r.cos_weights << ','
                    << r.rel_l2_output << ','
                    << r.encode_ms << ',' << r.attn_ms_baseline << ','
                    << r.attn_ms_tq << '\n';
            }
            std::printf("wrote csv: %s\n", csv_path.c_str());
        } else {
            std::fprintf(stderr, "could not open csv path %s\n", csv_path.c_str());
        }
    }
    return 0;
}
