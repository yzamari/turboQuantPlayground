#include "turboquant/api.hpp"

#include "packing.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace turboquant {

// -----------------------------------------------------------------------------
// TurboQuantMSE
// -----------------------------------------------------------------------------
TurboQuantMSE::TurboQuantMSE(int dim, int bits, IBackend* backend,
                             std::vector<float> Pi)
    : dim_(dim), bits_(bits), backend_(backend), Pi_(std::move(Pi)) {
    if (!backend_) throw std::invalid_argument("TurboQuantMSE: backend is null");
    if (Pi_.size() != static_cast<size_t>(dim) * dim) {
        throw std::invalid_argument("TurboQuantMSE: Pi must be d*d row-major");
    }
    cb_ = get_codebook(dim, bits);
    if (!cb_) {
        throw std::runtime_error(
            "TurboQuantMSE: no embedded codebook for the requested (d, bits) pair");
    }
}

MSEQuantized TurboQuantMSE::quantize(const float* x, int n_vec) const {
    MSEQuantized q;
    q.n_vec      = n_vec;
    q.bits       = bits_;
    q.d          = dim_;
    q.packed_len = packed_len_for(dim_, bits_);
    q.indices.assign(static_cast<size_t>(n_vec) * q.packed_len, 0);
    q.norms.resize(n_vec);

    // 1) per-vector norm
    // 2) normalize
    // 3) rotate (backend matmul: y = x_unit @ Pi^T)
    // 4) mse_encode (fused searchsorted + bit-pack)
    std::vector<float> x_unit(static_cast<size_t>(n_vec) * dim_);
    for (int v = 0; v < n_vec; ++v) {
        const float* xv = x + static_cast<size_t>(v) * dim_;
        double s = 0.0;
        for (int j = 0; j < dim_; ++j) s += static_cast<double>(xv[j]) * xv[j];
        float n = static_cast<float>(std::sqrt(s));
        q.norms[v] = n;
        float inv  = 1.0f / (n + 1e-10f);
        float* yu  = x_unit.data() + static_cast<size_t>(v) * dim_;
        for (int j = 0; j < dim_; ++j) yu[j] = xv[j] * inv;
    }

    std::vector<float> y(static_cast<size_t>(n_vec) * dim_);
    backend_->rotate(x_unit.data(), Pi_.data(), n_vec, dim_, y.data());

    backend_->mse_encode(y.data(), cb_->decision_boundaries.data(),
                         n_vec, dim_, bits_, q.indices.data());
    return q;
}

void TurboQuantMSE::dequantize(const MSEQuantized& q, float* out) const {
    assert(q.d == dim_ && q.bits == bits_);

    // Unpack -> centroids -> inverse rotate -> rescale.
    std::vector<int32_t> idx(static_cast<size_t>(q.n_vec) * dim_);
    unpack_indices(q.indices.data(), q.n_vec, dim_, bits_, idx.data());

    std::vector<float> y_hat(static_cast<size_t>(q.n_vec) * dim_);
    const float* C = cb_->centroids.data();
    for (int v = 0; v < q.n_vec; ++v) {
        const int32_t* iv = idx.data() + static_cast<size_t>(v) * dim_;
        float* yv         = y_hat.data() + static_cast<size_t>(v) * dim_;
        for (int j = 0; j < dim_; ++j) yv[j] = C[iv[j]];
    }

    // Inverse rotation: x_hat = y_hat @ Pi  (note: NOT Pi^T)
    // We do it as a row-major matmul: out[n, j] = sum_k y_hat[n, k] * Pi[k, j].
    for (int v = 0; v < q.n_vec; ++v) {
        const float* yv = y_hat.data() + static_cast<size_t>(v) * dim_;
        float*       ov = out + static_cast<size_t>(v) * dim_;
        for (int j = 0; j < dim_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < dim_; ++k) {
                sum += static_cast<double>(yv[k]) * Pi_[k * dim_ + j];
            }
            ov[j] = static_cast<float>(sum) * q.norms[v];
        }
    }
}

// -----------------------------------------------------------------------------
// TurboQuantProd
// -----------------------------------------------------------------------------
TurboQuantProd::TurboQuantProd(int dim, int bits, IBackend* backend,
                               std::vector<float> Pi, std::vector<float> S)
    : dim_(dim), bits_(bits), backend_(backend),
      mse_(dim, bits - 1, backend, std::move(Pi)),
      S_(std::move(S)) {
    if (bits < 2) {
        throw std::invalid_argument("TurboQuantProd requires bits >= 2");
    }
    if (S_.size() != static_cast<size_t>(dim) * dim) {
        throw std::invalid_argument("TurboQuantProd: S must be d*d row-major");
    }
    qjl_scale_ = std::sqrt(3.14159265358979323846f / 2.0f) / static_cast<float>(dim);
}

ProdQuantized TurboQuantProd::quantize(const float* x, int n_vec) const {
    ProdQuantized q;
    q.n_vec            = n_vec;
    q.d                = dim_;
    q.mse_bits         = bits_ - 1;
    q.packed_len_mse   = packed_len_for(dim_, q.mse_bits);
    q.packed_len_signs = packed_signs_len(dim_);

    // Stage 1: MSE quantize at (b - 1) bits
    MSEQuantized mq = mse_.quantize(x, n_vec);
    q.mse_indices = std::move(mq.indices);
    q.norms       = std::move(mq.norms);

    // Reconstruct x_hat for residual computation
    std::vector<float> x_hat(static_cast<size_t>(n_vec) * dim_);
    MSEQuantized mq_view;  // reborrow data so dequantize sees the moved buffer
    mq_view.indices    = q.mse_indices;
    mq_view.norms      = q.norms;
    mq_view.n_vec      = n_vec;
    mq_view.bits       = q.mse_bits;
    mq_view.d          = dim_;
    mq_view.packed_len = q.packed_len_mse;
    mse_.dequantize(mq_view, x_hat.data());

    // Residual + its norm
    std::vector<float> residual(static_cast<size_t>(n_vec) * dim_);
    q.residual_norms.resize(n_vec);
    for (int v = 0; v < n_vec; ++v) {
        const float* xv  = x + static_cast<size_t>(v) * dim_;
        const float* xhv = x_hat.data() + static_cast<size_t>(v) * dim_;
        float* rv        = residual.data() + static_cast<size_t>(v) * dim_;
        double s = 0.0;
        for (int j = 0; j < dim_; ++j) {
            float r = xv[j] - xhv[j];
            rv[j] = r;
            s += static_cast<double>(r) * r;
        }
        q.residual_norms[v] = static_cast<float>(std::sqrt(s));
    }

    // Stage 2: project residual through S^T, then sign-pack
    // projected[v, j] = sum_k residual[v, k] * S[j, k]   (since out = r @ S^T)
    std::vector<float> projected(static_cast<size_t>(n_vec) * dim_);
    for (int v = 0; v < n_vec; ++v) {
        const float* rv = residual.data() + static_cast<size_t>(v) * dim_;
        float* pv       = projected.data() + static_cast<size_t>(v) * dim_;
        for (int j = 0; j < dim_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < dim_; ++k) {
                sum += static_cast<double>(rv[k]) * S_[j * dim_ + k];
            }
            pv[j] = static_cast<float>(sum);
        }
    }
    q.qjl_signs.assign(static_cast<size_t>(n_vec) * q.packed_len_signs, 0);
    pack_qjl_signs(projected.data(), n_vec, dim_, q.qjl_signs.data());
    return q;
}

void TurboQuantProd::dequantize(const ProdQuantized& q, float* out) const {
    // Stage 1
    MSEQuantized mq;
    mq.indices    = q.mse_indices;
    mq.norms      = q.norms;
    mq.n_vec      = q.n_vec;
    mq.bits       = q.mse_bits;
    mq.d          = dim_;
    mq.packed_len = q.packed_len_mse;
    mse_.dequantize(mq, out);  // out holds x_mse

    // Stage 2: x += qjl_scale * res_norm * (signs @ S)
    // signs[v, k] in {-1, +1}, S row-major [d, d]
    // contribution[v, j] = sum_k signs[v, k] * S[k, j]
    std::vector<float> signs_f(static_cast<size_t>(q.n_vec) * dim_);
    unpack_qjl_signs_to_float(q.qjl_signs.data(), q.n_vec, dim_, signs_f.data());

    for (int v = 0; v < q.n_vec; ++v) {
        const float* sv = signs_f.data() + static_cast<size_t>(v) * dim_;
        float*       ov = out + static_cast<size_t>(v) * dim_;
        float scale     = qjl_scale_ * q.residual_norms[v];
        for (int j = 0; j < dim_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < dim_; ++k) {
                sum += static_cast<double>(sv[k]) * S_[k * dim_ + j];
            }
            ov[j] += static_cast<float>(sum) * scale;
        }
    }
}

void TurboQuantProd::attention_score(const float* query, int BH, int n_q,
                                     const ProdQuantized& key, int N,
                                     float* out) const {
    // Pre-rotate the query once: q_rot[bh*n_q, j] = sum_k query[bh*n_q, k] * Pi[j, k]
    // (since q_rot = query @ Pi^T)
    const int BHQ = BH * n_q;
    std::vector<float> q_rot   (static_cast<size_t>(BHQ) * dim_);
    std::vector<float> q_sketch(static_cast<size_t>(BHQ) * dim_);
    const auto& Pi = mse_.Pi();
    for (int b = 0; b < BHQ; ++b) {
        const float* qv = query + static_cast<size_t>(b) * dim_;
        float* qr       = q_rot.data()    + static_cast<size_t>(b) * dim_;
        float* qs       = q_sketch.data() + static_cast<size_t>(b) * dim_;
        for (int j = 0; j < dim_; ++j) {
            double sr = 0.0;
            double ss = 0.0;
            for (int k = 0; k < dim_; ++k) {
                sr += static_cast<double>(qv[k]) * Pi[j * dim_ + k];
                ss += static_cast<double>(qv[k]) * S_[j * dim_ + k];
            }
            qr[j] = static_cast<float>(sr);
            qs[j] = static_cast<float>(ss);
        }
    }

    // Per-(BH, n_q) we need keys at index [bh * N .. (bh+1) * N).
    // The key buffers are per-BH (B*H groups, N keys each). The kernel walks
    // the Cartesian product of n_q queries × N keys for each BH.
    //
    // For this scalar wrapper we just call the backend kernels per-BH-per-query.

    const int packed_len_mse = key.packed_len_mse;
    const int packed_len_sig = key.packed_len_signs;

    for (int bh = 0; bh < BH; ++bh) {
        const uint8_t* mse_packed_bh = key.mse_indices.data()    + static_cast<size_t>(bh) * N * packed_len_mse;
        const uint8_t* signs_bh      = key.qjl_signs.data()      + static_cast<size_t>(bh) * N * packed_len_sig;
        const float*   norms_bh      = key.norms.data()          + static_cast<size_t>(bh) * N;
        const float*   res_bh        = key.residual_norms.data() + static_cast<size_t>(bh) * N;

        for (int t = 0; t < n_q; ++t) {
            const float* qr_t = q_rot.data()    + (static_cast<size_t>(bh) * n_q + t) * dim_;
            const float* qs_t = q_sketch.data() + (static_cast<size_t>(bh) * n_q + t) * dim_;
            float*       ot   = out             + (static_cast<size_t>(bh) * n_q + t) * N;

            // Treat BH=1 for the kernel (single bh, single query) so packed
            // strides match.
            backend_->mse_score(qr_t, mse_packed_bh, norms_bh,
                                mse_.centroids().data(),
                                /*BH=*/1, N, dim_, key.mse_bits, ot);
            backend_->qjl_score(qs_t, signs_bh, res_bh, ot,
                                /*BH=*/1, N, dim_, qjl_scale_, ot);
        }
    }
}

}  // namespace turboquant
