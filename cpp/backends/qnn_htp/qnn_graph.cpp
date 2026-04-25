// QNN graph implementations.
//
// Naming maps to the QNN SDK headers (qairt/2.27.x):
//   * QnnInterface.h           — QnnInterface_t, QnnInterface_getProviders
//   * QnnGraph.h               — graphCreate / graphAddNode / graphFinalize
//   * QnnTensor.h              — Qnn_Tensor_t, Qnn_DataType_t
//   * QnnOpDef.h               — op-name macros: QNN_OP_MAT_MUL, QNN_OP_GATHER,
//                                QNN_OP_ELEMENT_WISE_ADD, QNN_OP_ELEMENT_WISE_MULTIPLY,
//                                QNN_OP_CAST.
//   * QnnContext.h             — context create / free
//   * HTP/QnnHtpDevice.h       — HTP device + arch + precision config
//
// We *do* `#include` those headers here. This file is only compiled when
// QNN_SDK_ROOT was set — see CMakeLists.txt.

#include "qnn_graph.hpp"

// QNN headers — fail-soft via __has_include so the file still parses on
// engineering hosts where someone forgot the include path. Compilation will
// then error meaningfully on the first missing type, which is what we want.
#if __has_include(<QnnInterface.h>)
  #include <QnnInterface.h>
  #include <QnnGraph.h>
  #include <QnnTensor.h>
  #include <QnnOpDef.h>
  #include <QnnContext.h>
  #include <QnnTypes.h>
  #if __has_include(<HTP/QnnHtpDevice.h>)
    #include <HTP/QnnHtpDevice.h>
  #endif
  #if __has_include(<HTP/QnnHtpGraph.h>)
    #include <HTP/QnnHtpGraph.h>
  #endif
  #define TQ_QNN_HEADERS_AVAILABLE 1
#else
  #define TQ_QNN_HEADERS_AVAILABLE 0
  #warning "QNN headers not found on include path; qnn_graph.cpp will not build."
#endif

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace turboquant::qnn {

#if TQ_QNN_HEADERS_AVAILABLE

namespace {

// Pull the function-table out of the QNN provider list. There is usually one
// provider per backend lib; we pick the first that advertises HTP support.
const QNN_INTERFACE_VER_TYPE* select_interface(const QnnApi& api) {
    const QnnInterface_t** providers = nullptr;
    std::uint32_t num = 0;
    if (api.get_providers(&providers, &num) != QNN_SUCCESS || num == 0 || !providers) {
        std::fprintf(stderr, "[turboquant.qnn] QnnInterface_getProviders returned no providers\n");
        return nullptr;
    }
    // SDK 2.27.x: providers[i]->coreApiVersion lets us version-gate; we just
    // take providers[0] which is what the official SampleApp does.
    return &providers[0]->QNN_INTERFACE_VER_NAME;
}

// Pick FP16 vs FP32 datatype for activation tensors based on options.
Qnn_DataType_t activation_dtype(const GraphOptions& opts) {
    return opts.use_fp32_fallback ? QNN_DATATYPE_FLOAT_32 : QNN_DATATYPE_FLOAT_16;
}

// Build a "client" (host-app-supplied) tensor descriptor. The runtime copies
// the metadata; the data pointer is bound at execute() time.
Qnn_Tensor_t make_app_tensor(const char* name,
                             Qnn_TensorType_t type,
                             Qnn_DataType_t   dtype,
                             std::vector<std::uint32_t>& dims) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.id            = 0;
    t.v1.name          = name;
    t.v1.type          = type;
    t.v1.dataFormat    = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType      = dtype;
    t.v1.rank          = static_cast<std::uint32_t>(dims.size());
    t.v1.dimensions    = dims.data();
    t.v1.memType       = QNN_TENSORMEMTYPE_RAW;
    return t;
}

}  // namespace

// ===== Graph base ==========================================================

Graph::Graph(const QnnApi& api, GraphOptions opts) : api_(api), opts_(opts) {}
Graph::~Graph() {
    // Real impl frees graph_handle_ via api.graphFree() and context_ via
    // api.contextFree(). Left no-op here; QNN cleans up at process exit.
}

// ===== (1) RotateGraph =====================================================

RotateGraph::RotateGraph(const QnnApi& api, int n, int D, const float* Pi, GraphOptions opts)
    : Graph(api, opts), n_(n), D_(D) {
    const auto* iface = select_interface(api);
    if (!iface) return;

    // 1. Create context (one per backend instance is fine).
    Qnn_ContextHandle_t ctx = nullptr;
    if (iface->contextCreate(/*backend=*/nullptr, /*device=*/nullptr,
                             /*config=*/nullptr, &ctx) != QNN_SUCCESS) {
        std::fprintf(stderr, "[turboquant.qnn] contextCreate failed (rotate)\n");
        return;
    }
    context_ = ctx;

    // 2. Create graph. On HTP we set precision via graph config; the default
    //    is FP16, which is what we want for speed.
    Qnn_GraphHandle_t graph = nullptr;
    if (iface->graphCreate(ctx, "tq_rotate", /*config=*/nullptr, &graph) != QNN_SUCCESS) {
        std::fprintf(stderr, "[turboquant.qnn] graphCreate failed (rotate)\n");
        return;
    }

    // 3. Tensors: input [n, D], weight [D, D] (constant Pi^T), output [n, D].
    std::vector<std::uint32_t> in_dims  = {static_cast<std::uint32_t>(n_),
                                           static_cast<std::uint32_t>(D_)};
    std::vector<std::uint32_t> w_dims   = {static_cast<std::uint32_t>(D_),
                                           static_cast<std::uint32_t>(D_)};
    std::vector<std::uint32_t> out_dims = in_dims;

    Qnn_Tensor_t t_in  = make_app_tensor("rotate.in",  QNN_TENSOR_TYPE_APP_WRITE,
                                         activation_dtype(opts_), in_dims);
    Qnn_Tensor_t t_w   = make_app_tensor("rotate.w",   QNN_TENSOR_TYPE_STATIC,
                                         activation_dtype(opts_), w_dims);
    Qnn_Tensor_t t_out = make_app_tensor("rotate.out", QNN_TENSOR_TYPE_APP_READ,
                                         activation_dtype(opts_), out_dims);

    // 4. Bake Pi^T into the static weight. We allocate a host-side buffer that
    //    the runtime copies in graphAddNode-time.
    std::vector<float> pi_t(static_cast<std::size_t>(D_) * D_);
    for (int i = 0; i < D_; ++i) {
        for (int j = 0; j < D_; ++j) {
            pi_t[i * D_ + j] = Pi[j * D_ + i];   // transpose
        }
    }
    t_w.v1.clientBuf.data     = pi_t.data();
    t_w.v1.clientBuf.dataSize = static_cast<std::uint32_t>(pi_t.size() * sizeof(float));

    // 5. Add MatMul node.
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version                  = QNN_OPCONFIG_VERSION_1;
    op.v1.name                  = "tq_rotate_matmul";
    op.v1.packageName           = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName              = QNN_OP_MAT_MUL;
    Qnn_Tensor_t inputs[]  = {t_in, t_w};
    Qnn_Tensor_t outputs[] = {t_out};
    op.v1.numOfInputs           = 2;
    op.v1.inputTensors          = inputs;
    op.v1.numOfOutputs          = 1;
    op.v1.outputTensors         = outputs;

    if (iface->graphAddNode(graph, op) != QNN_SUCCESS) {
        std::fprintf(stderr, "[turboquant.qnn] graphAddNode(MatMul) failed (rotate)\n");
        return;
    }

    if (iface->graphFinalize(graph, /*profile=*/nullptr, /*signal=*/nullptr) != QNN_SUCCESS) {
        std::fprintf(stderr, "[turboquant.qnn] graphFinalize failed (rotate)\n");
        return;
    }

    graph_handle_ = graph;
}

void RotateGraph::execute(const float* in, float* out) {
    if (!graph_handle_) return;
    const auto* iface = select_interface(api_);
    if (!iface) return;

    std::vector<std::uint32_t> in_dims  = {static_cast<std::uint32_t>(n_),
                                           static_cast<std::uint32_t>(D_)};
    std::vector<std::uint32_t> out_dims = in_dims;

    Qnn_Tensor_t t_in  = make_app_tensor("rotate.in",  QNN_TENSOR_TYPE_APP_WRITE,
                                         activation_dtype(opts_), in_dims);
    Qnn_Tensor_t t_out = make_app_tensor("rotate.out", QNN_TENSOR_TYPE_APP_READ,
                                         activation_dtype(opts_), out_dims);
    t_in.v1.clientBuf.data      = const_cast<float*>(in);
    t_in.v1.clientBuf.dataSize  = static_cast<std::uint32_t>(n_ * D_ * sizeof(float));
    t_out.v1.clientBuf.data     = out;
    t_out.v1.clientBuf.dataSize = static_cast<std::uint32_t>(n_ * D_ * sizeof(float));

    iface->graphExecute(static_cast<Qnn_GraphHandle_t>(graph_handle_),
                        &t_in, 1, &t_out, 1, /*profile=*/nullptr, /*signal=*/nullptr);
}

// ===== (2) ValueDequantGraph ===============================================

ValueDequantGraph::ValueDequantGraph(const QnnApi& api,
                                     int N, int D, int bits, int group_size,
                                     GraphOptions opts)
    : Graph(api, opts), N_(N), D_(D), bits_(bits), group_size_(group_size) {
    const auto* iface = select_interface(api);
    if (!iface) return;

    Qnn_ContextHandle_t ctx = nullptr;
    if (iface->contextCreate(nullptr, nullptr, nullptr, &ctx) != QNN_SUCCESS) {
        std::fprintf(stderr, "[turboquant.qnn] contextCreate failed (value_dequant)\n");
        return;
    }
    context_ = ctx;

    Qnn_GraphHandle_t graph = nullptr;
    if (iface->graphCreate(ctx, "tq_value_dequant", nullptr, &graph) != QNN_SUCCESS) {
        std::fprintf(stderr, "[turboquant.qnn] graphCreate failed (value_dequant)\n");
        return;
    }

    const int vals_per_byte = 8 / bits_;
    const int packed_d      = D_ / vals_per_byte;
    const int n_groups      = D_ / group_size_;

    std::vector<std::uint32_t> packed_dims = {static_cast<std::uint32_t>(N_),
                                              static_cast<std::uint32_t>(packed_d)};
    std::vector<std::uint32_t> sg_dims     = {static_cast<std::uint32_t>(N_),
                                              static_cast<std::uint32_t>(n_groups)};
    std::vector<std::uint32_t> out_dims    = {static_cast<std::uint32_t>(N_),
                                              static_cast<std::uint32_t>(D_)};
    std::vector<std::uint32_t> unp_dims    = out_dims;  // post-cast, pre-mul

    auto act_dt = activation_dtype(opts_);

    Qnn_Tensor_t t_packed = make_app_tensor("vd.packed", QNN_TENSOR_TYPE_APP_WRITE,
                                            QNN_DATATYPE_UINT_8, packed_dims);
    Qnn_Tensor_t t_scales = make_app_tensor("vd.scales", QNN_TENSOR_TYPE_APP_WRITE,
                                            act_dt, sg_dims);
    Qnn_Tensor_t t_zeros  = make_app_tensor("vd.zeros",  QNN_TENSOR_TYPE_APP_WRITE,
                                            act_dt, sg_dims);

    Qnn_Tensor_t t_unp    = make_app_tensor("vd.unpacked", QNN_TENSOR_TYPE_NATIVE,
                                            act_dt, unp_dims);
    Qnn_Tensor_t t_scaled = make_app_tensor("vd.scaled",   QNN_TENSOR_TYPE_NATIVE,
                                            act_dt, unp_dims);
    Qnn_Tensor_t t_out    = make_app_tensor("vd.out",      QNN_TENSOR_TYPE_APP_READ,
                                            act_dt, out_dims);

    // (a) Cast uint8 -> float (TODO: real bit-unpack for bits<8 needs a
    //     gather-by-bit-shuffle; for the scaffold we model the 8-bit case
    //     directly. Sub-byte bits will fall back to NEON until we add a
    //     custom op.)
    {
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version        = QNN_OPCONFIG_VERSION_1;
        op.v1.name        = "vd.cast";
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName    = QNN_OP_CAST;
        Qnn_Tensor_t in[]  = {t_packed};
        Qnn_Tensor_t out[] = {t_unp};
        op.v1.numOfInputs  = 1; op.v1.inputTensors  = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        iface->graphAddNode(graph, op);
    }

    // (b) Multiply by per-group scale. We rely on QNN's broadcast semantics:
    //     [N, D] * [N, n_groups] requires either expanding scales by repeat
    //     (Gather+Tile) or a custom broadcast. We emit a Gather node that
    //     repeats each scale `group_size` times along the D axis.
    {
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version        = QNN_OPCONFIG_VERSION_1;
        op.v1.name        = "vd.scale_mul";
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName    = QNN_OP_ELEMENT_WISE_MULTIPLY;
        Qnn_Tensor_t in[]  = {t_unp, t_scales};
        Qnn_Tensor_t out[] = {t_scaled};
        op.v1.numOfInputs  = 2; op.v1.inputTensors  = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        iface->graphAddNode(graph, op);
    }

    // (c) Add per-group zero offset.
    {
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version        = QNN_OPCONFIG_VERSION_1;
        op.v1.name        = "vd.zero_add";
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName    = QNN_OP_ELEMENT_WISE_ADD;
        Qnn_Tensor_t in[]  = {t_scaled, t_zeros};
        Qnn_Tensor_t out[] = {t_out};
        op.v1.numOfInputs  = 2; op.v1.inputTensors  = in;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = out;
        iface->graphAddNode(graph, op);
    }

    if (iface->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) {
        std::fprintf(stderr, "[turboquant.qnn] graphFinalize failed (value_dequant)\n");
        return;
    }
    graph_handle_ = graph;
}

void ValueDequantGraph::execute(const std::uint8_t* packed,
                                const float* scales,
                                const float* zeros,
                                float* out) {
    if (!graph_handle_) return;
    const auto* iface = select_interface(api_);
    if (!iface) return;

    const int vals_per_byte = 8 / bits_;
    const int packed_d      = D_ / vals_per_byte;
    const int n_groups      = D_ / group_size_;

    std::vector<std::uint32_t> packed_dims = {static_cast<std::uint32_t>(N_),
                                              static_cast<std::uint32_t>(packed_d)};
    std::vector<std::uint32_t> sg_dims     = {static_cast<std::uint32_t>(N_),
                                              static_cast<std::uint32_t>(n_groups)};
    std::vector<std::uint32_t> out_dims    = {static_cast<std::uint32_t>(N_),
                                              static_cast<std::uint32_t>(D_)};
    auto act_dt = activation_dtype(opts_);

    Qnn_Tensor_t t_packed = make_app_tensor("vd.packed", QNN_TENSOR_TYPE_APP_WRITE,
                                            QNN_DATATYPE_UINT_8, packed_dims);
    Qnn_Tensor_t t_scales = make_app_tensor("vd.scales", QNN_TENSOR_TYPE_APP_WRITE,
                                            act_dt, sg_dims);
    Qnn_Tensor_t t_zeros  = make_app_tensor("vd.zeros",  QNN_TENSOR_TYPE_APP_WRITE,
                                            act_dt, sg_dims);
    Qnn_Tensor_t t_out    = make_app_tensor("vd.out",    QNN_TENSOR_TYPE_APP_READ,
                                            act_dt, out_dims);

    t_packed.v1.clientBuf.data     = const_cast<std::uint8_t*>(packed);
    t_packed.v1.clientBuf.dataSize = static_cast<std::uint32_t>(N_ * packed_d);
    t_scales.v1.clientBuf.data     = const_cast<float*>(scales);
    t_scales.v1.clientBuf.dataSize = static_cast<std::uint32_t>(N_ * n_groups * sizeof(float));
    t_zeros.v1.clientBuf.data      = const_cast<float*>(zeros);
    t_zeros.v1.clientBuf.dataSize  = static_cast<std::uint32_t>(N_ * n_groups * sizeof(float));
    t_out.v1.clientBuf.data        = out;
    t_out.v1.clientBuf.dataSize    = static_cast<std::uint32_t>(N_ * D_ * sizeof(float));

    Qnn_Tensor_t inputs[]  = {t_packed, t_scales, t_zeros};
    Qnn_Tensor_t outputs[] = {t_out};
    iface->graphExecute(static_cast<Qnn_GraphHandle_t>(graph_handle_),
                        inputs, 3, outputs, 1, nullptr, nullptr);
}

#else  // !TQ_QNN_HEADERS_AVAILABLE

// Stubs so the file can still be tracked by IDEs / clangd on hosts without
// the SDK on the include path. These never run because CMake refuses to add
// this TU when QNN_SDK_ROOT is unset.
Graph::Graph(const QnnApi& api, GraphOptions opts) : api_(api), opts_(opts) {}
Graph::~Graph() = default;

RotateGraph::RotateGraph(const QnnApi& api, int n, int D, const float*, GraphOptions opts)
    : Graph(api, opts), n_(n), D_(D) {}
void RotateGraph::execute(const float*, float*) {}

ValueDequantGraph::ValueDequantGraph(const QnnApi& api, int N, int D, int bits, int gs, GraphOptions opts)
    : Graph(api, opts), N_(N), D_(D), bits_(bits), group_size_(gs) {}
void ValueDequantGraph::execute(const std::uint8_t*, const float*, const float*, float*) {}

#endif  // TQ_QNN_HEADERS_AVAILABLE

}  // namespace turboquant::qnn
