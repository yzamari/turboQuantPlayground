// Adreno Vulkan compute backend implementing IBackend.
//
// All five kernels (rotate, mse_encode, mse_score, qjl_score, value_dequant)
// run as Vulkan 1.1 compute shaders. The SPIR-V for each shader is compiled
// at build time by glslc and embedded into the static library as a const byte
// array via the shared embed_resource CMake helper.
//
// Buffer strategy (correctness first; pooling is a TODO):
//   For each call we create per-call host-visible upload buffers and a
//   device-local output buffer. Copy host -> upload, dispatch the pipeline,
//   copy device-local out -> readback (host-visible) -> caller.
//
// The 8-bit storage extension (VK_KHR_8bit_storage) is intentionally NOT
// required: shaders read/write packed uchar data through uint SSBOs and shift.
// This keeps us compatible with conservative Vulkan 1.1 implementations.

#include "turboquant/backend.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" {
extern const unsigned char tq_vulkan_mse_score_uchar[];   extern const size_t tq_vulkan_mse_score_uchar_size;
extern const unsigned char tq_vulkan_mse_score_u32_b3[];  extern const size_t tq_vulkan_mse_score_u32_b3_size;
extern const unsigned char tq_vulkan_qjl_score[];         extern const size_t tq_vulkan_qjl_score_size;
extern const unsigned char tq_vulkan_mse_encode_uchar[];  extern const size_t tq_vulkan_mse_encode_uchar_size;
extern const unsigned char tq_vulkan_mse_encode_u32_b3[]; extern const size_t tq_vulkan_mse_encode_u32_b3_size;
extern const unsigned char tq_vulkan_value_dequant[];     extern const size_t tq_vulkan_value_dequant_size;
extern const unsigned char tq_vulkan_rotate[];            extern const size_t tq_vulkan_rotate_size;
}

namespace turboquant {
namespace {

// --------------------------------------------------------------------------
// Error helpers
// --------------------------------------------------------------------------
#define VK_OK(call)                                                               \
    do {                                                                          \
        VkResult err__ = (call);                                                  \
        if (err__ != VK_SUCCESS) {                                                \
            std::fprintf(stderr, "[vulkan] %s -> %d at %s:%d\n",                  \
                         #call, (int)err__, __FILE__, __LINE__);                  \
            return;                                                               \
        }                                                                         \
    } while (0)

#define VK_OK_RET(call, ret)                                                      \
    do {                                                                          \
        VkResult err__ = (call);                                                  \
        if (err__ != VK_SUCCESS) {                                                \
            std::fprintf(stderr, "[vulkan] %s -> %d at %s:%d\n",                  \
                         #call, (int)err__, __FILE__, __LINE__);                  \
            return ret;                                                           \
        }                                                                         \
    } while (0)

// One pipeline + descriptor-set layout per shader.
struct Pipeline {
    VkShaderModule        module       = VK_NULL_HANDLE;
    VkDescriptorSetLayout dset_layout  = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout  = VK_NULL_HANDLE;
    VkPipeline            pipeline     = VK_NULL_HANDLE;
    int                   n_bindings   = 0;
    uint32_t              push_size    = 0;
};

// A resident allocation: VkBuffer + DeviceMemory. Created/destroyed per call.
struct Buf {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
    void*          mapped = nullptr;  // non-null only for host-visible
};

class VulkanBackend final : public IBackend {
public:
    const char* name() const override { return "vulkan"; }

    bool init() override {
        // ---- VkInstance ------------------------------------------------------
        VkApplicationInfo app{};
        app.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "turboquant";
        app.apiVersion       = VK_API_VERSION_1_1;

        VkInstanceCreateInfo ici{};
        ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VK_OK_RET(vkCreateInstance(&ici, nullptr, &instance_), false);

        // ---- VkPhysicalDevice (prefer discrete then integrated) -------------
        uint32_t n_phys = 0;
        VK_OK_RET(vkEnumeratePhysicalDevices(instance_, &n_phys, nullptr), false);
        if (n_phys == 0) {
            std::fprintf(stderr, "[vulkan] no physical devices\n");
            return false;
        }
        std::vector<VkPhysicalDevice> phys(n_phys);
        VK_OK_RET(vkEnumeratePhysicalDevices(instance_, &n_phys, phys.data()), false);

        VkPhysicalDevice chosen = VK_NULL_HANDLE;
        int chosen_score = -1;
        for (VkPhysicalDevice p : phys) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(p, &props);
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   score = 4;
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 3;
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)    score = 2;
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)           score = 1;
            if (score > chosen_score) {
                chosen_score = score;
                chosen = p;
            }
        }
        if (chosen == VK_NULL_HANDLE) {
            std::fprintf(stderr, "[vulkan] no suitable physical device\n");
            return false;
        }
        phys_ = chosen;

        // Diagnostics + memory props.
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_, &props);
        std::fprintf(stderr, "[vulkan] device: %s (api %u.%u.%u)\n",
                     props.deviceName,
                     VK_VERSION_MAJOR(props.apiVersion),
                     VK_VERSION_MINOR(props.apiVersion),
                     VK_VERSION_PATCH(props.apiVersion));
        vkGetPhysicalDeviceMemoryProperties(phys_, &mem_props_);

        // ---- Compute queue family -------------------------------------------
        uint32_t n_qf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n_qf, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(n_qf);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n_qf, qfs.data());
        queue_family_ = UINT32_MAX;
        for (uint32_t i = 0; i < n_qf; ++i) {
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family_ = i;
                break;
            }
        }
        if (queue_family_ == UINT32_MAX) {
            std::fprintf(stderr, "[vulkan] no compute queue family\n");
            return false;
        }

        // ---- VkDevice + queue -----------------------------------------------
        const float qprio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &qprio;

        VkDeviceCreateInfo dci{};
        dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &qci;
        VK_OK_RET(vkCreateDevice(phys_, &dci, nullptr, &device_), false);
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        // ---- Command pool ---------------------------------------------------
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = queue_family_;
        VK_OK_RET(vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_), false);

        // ---- Descriptor pool: generous, since we recycle dsets per call ----
        // Each dispatch uses up to 5 storage-buffer bindings; allow many sets.
        const uint32_t kMaxSets = 1024;
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = kMaxSets * 5;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = kMaxSets;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        VK_OK_RET(vkCreateDescriptorPool(device_, &dpci, nullptr, &dpool_), false);

        // ---- Build all pipelines --------------------------------------------
        if (!build_pipeline(pl_score_uchar_,    tq_vulkan_mse_score_uchar,    tq_vulkan_mse_score_uchar_size,    5, sizeof(int) * 7))                  return false;
        if (!build_pipeline(pl_score_u32_b3_,   tq_vulkan_mse_score_u32_b3,   tq_vulkan_mse_score_u32_b3_size,   5, sizeof(int) * 4))                  return false;
        if (!build_pipeline(pl_qjl_,            tq_vulkan_qjl_score,          tq_vulkan_qjl_score_size,          5, sizeof(int) * 4 + sizeof(float)))  return false;
        if (!build_pipeline(pl_encode_uchar_,   tq_vulkan_mse_encode_uchar,   tq_vulkan_mse_encode_uchar_size,   3, sizeof(int) * 9))                  return false;
        if (!build_pipeline(pl_encode_u32_b3_,  tq_vulkan_mse_encode_u32_b3,  tq_vulkan_mse_encode_u32_b3_size,  3, sizeof(int) * 3))                  return false;
        if (!build_pipeline(pl_vdq_,            tq_vulkan_value_dequant,      tq_vulkan_value_dequant_size,      4, sizeof(int) * 8))                  return false;
        if (!build_pipeline(pl_rotate_,         tq_vulkan_rotate,             tq_vulkan_rotate_size,             3, sizeof(int) * 2))                  return false;

        return true;
    }

    ~VulkanBackend() override {
        if (device_) {
            vkDeviceWaitIdle(device_);
            destroy_pipeline(pl_score_uchar_);
            destroy_pipeline(pl_score_u32_b3_);
            destroy_pipeline(pl_qjl_);
            destroy_pipeline(pl_encode_uchar_);
            destroy_pipeline(pl_encode_u32_b3_);
            destroy_pipeline(pl_vdq_);
            destroy_pipeline(pl_rotate_);
            if (dpool_)    vkDestroyDescriptorPool(device_, dpool_,    nullptr);
            if (cmd_pool_) vkDestroyCommandPool   (device_, cmd_pool_, nullptr);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    // ----------------------------------------------------------------------
    // IBackend overrides
    // ----------------------------------------------------------------------
    void rotate(const float* in, const float* Pi,
                int n, int D, float* out) override {
        const VkDeviceSize in_bytes  = (VkDeviceSize)n * D * sizeof(float);
        const VkDeviceSize pi_bytes  = (VkDeviceSize)D * D * sizeof(float);
        const VkDeviceSize out_bytes = (VkDeviceSize)n * D * sizeof(float);

        Buf b_in  = make_storage_with_data(in,  in_bytes);
        Buf b_pi  = make_storage_with_data(Pi,  pi_bytes);
        Buf b_out = make_storage_buffer(out_bytes, /*host_visible=*/true);

        VkBuffer bufs[] = { b_in.buffer, b_pi.buffer, b_out.buffer };
        int pc[2] = { n, D };

        // Local size (8,8); dispatch over (D, n).
        uint32_t gx = (D + 7) / 8;
        uint32_t gy = (n + 7) / 8;
        run_dispatch(pl_rotate_, bufs, 3, &pc, sizeof(pc), gx, gy, 1);

        std::memcpy(out, b_out.mapped, out_bytes);

        free_buf(b_in);
        free_buf(b_pi);
        free_buf(b_out);
    }

    void mse_encode(const float* rotated, const float* boundaries,
                    int N, int D, int bits, void* packed_out) override {
        const int n_boundaries = (1 << bits) - 1;
        const VkDeviceSize rot_bytes = (VkDeviceSize)N * D * sizeof(float);
        const VkDeviceSize bnd_bytes = (VkDeviceSize)n_boundaries * sizeof(float);

        Buf b_rot = make_storage_with_data(rotated,    rot_bytes);
        Buf b_bnd = make_storage_with_data(boundaries, bnd_bytes);

        if (bits == 3) {
            int n_words = (D + 9) / 10;
            VkDeviceSize out_bytes = (VkDeviceSize)N * n_words * sizeof(uint32_t);
            Buf b_out = make_storage_buffer(out_bytes, /*host_visible=*/true);

            int pc[3] = { N, D, n_words };
            VkBuffer bufs[] = { b_rot.buffer, b_bnd.buffer, b_out.buffer };
            uint32_t gx = (n_words + 63) / 64;
            uint32_t gy = N;
            run_dispatch(pl_encode_u32_b3_, bufs, 3, &pc, sizeof(pc), gx, gy, 1);
            std::memcpy(packed_out, b_out.mapped, out_bytes);
            free_buf(b_out);
        } else {
            int vals_per_byte, eff_bits, mask;
            switch (bits) {
                case 1: vals_per_byte = 8; eff_bits = 1; mask = 0x1;  break;
                case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3;  break;
                case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF;  break;
                case 8: default:
                        vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
            }
            int packed_d        = (D + vals_per_byte - 1) / vals_per_byte;
            int n_words_per_row = (packed_d + 3) / 4;
            int n_words_total   = N * n_words_per_row;
            VkDeviceSize out_bytes = (VkDeviceSize)n_words_total * sizeof(uint32_t);
            Buf b_out = make_storage_buffer(out_bytes, /*host_visible=*/true);

            int pc[9] = { N, D, packed_d, n_boundaries, eff_bits,
                          vals_per_byte, mask, n_words_per_row, n_words_total };
            VkBuffer bufs[] = { b_rot.buffer, b_bnd.buffer, b_out.buffer };
            uint32_t gx = (n_words_per_row + 63) / 64;
            uint32_t gy = N;
            run_dispatch(pl_encode_uchar_, bufs, 3, &pc, sizeof(pc), gx, gy, 1);

            // Compact: source has stride (n_words_per_row*4) bytes per row,
            // user wants packed_d bytes per row.
            const uint8_t* src = static_cast<const uint8_t*>(b_out.mapped);
            uint8_t*       dst = static_cast<uint8_t*>(packed_out);
            const size_t   src_stride = (size_t)n_words_per_row * 4;
            for (int i = 0; i < N; ++i) {
                std::memcpy(dst + (size_t)i * packed_d,
                            src + (size_t)i * src_stride,
                            packed_d);
            }
            free_buf(b_out);
        }

        free_buf(b_rot);
        free_buf(b_bnd);
    }

    void mse_score(const float* q_rot, const void* mse_packed,
                   const float* norms, const float* centroids,
                   int BH, int N, int D, int bits,
                   float* out) override {
        const VkDeviceSize q_bytes   = (VkDeviceSize)BH * D * sizeof(float);
        const VkDeviceSize nm_bytes  = (VkDeviceSize)BH * N * sizeof(float);
        const VkDeviceSize cn_bytes  = (VkDeviceSize)(1 << bits) * sizeof(float);
        const VkDeviceSize out_bytes = (VkDeviceSize)BH * N * sizeof(float);

        Buf b_q   = make_storage_with_data(q_rot,     q_bytes);
        Buf b_nm  = make_storage_with_data(norms,     nm_bytes);
        Buf b_cen = make_storage_with_data(centroids, cn_bytes);
        Buf b_out = make_storage_buffer  (out_bytes, /*host_visible=*/true);

        if (bits == 3) {
            int n_words = (D + 9) / 10;
            VkDeviceSize pk_bytes = (VkDeviceSize)BH * N * n_words * sizeof(uint32_t);
            Buf b_pk = make_storage_with_data(mse_packed, pk_bytes);

            int pc[4] = { BH, N, D, n_words };
            VkBuffer bufs[] = { b_q.buffer, b_pk.buffer, b_nm.buffer,
                                b_cen.buffer, b_out.buffer };
            uint32_t gx = (N + 63) / 64;
            uint32_t gy = BH;
            run_dispatch(pl_score_u32_b3_, bufs, 5, &pc, sizeof(pc), gx, gy, 1);
            std::memcpy(out, b_out.mapped, out_bytes);
            free_buf(b_pk);
        } else {
            int vals_per_byte, eff_bits, mask;
            switch (bits) {
                case 1: vals_per_byte = 8; eff_bits = 1; mask = 0x1;  break;
                case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3;  break;
                case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF;  break;
                case 8: default:
                        vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
            }
            int packed_d = (D + vals_per_byte - 1) / vals_per_byte;
            VkDeviceSize pk_bytes_logical = (VkDeviceSize)BH * N * packed_d;
            // Allocation must be a multiple of 4 bytes since the SSBO is uint.
            VkDeviceSize pk_bytes_padded = (pk_bytes_logical + 3) & ~VkDeviceSize(3);
            Buf b_pk = make_storage_buffer(pk_bytes_padded, /*host_visible=*/true);
            std::memcpy(b_pk.mapped, mse_packed, pk_bytes_logical);
            // Tail bytes in the last uint are unused (read but masked out by D check).

            int pc[7] = { BH, N, D, packed_d, eff_bits, vals_per_byte, mask };
            VkBuffer bufs[] = { b_q.buffer, b_pk.buffer, b_nm.buffer,
                                b_cen.buffer, b_out.buffer };
            uint32_t gx = (N + 63) / 64;
            uint32_t gy = BH;
            run_dispatch(pl_score_uchar_, bufs, 5, &pc, sizeof(pc), gx, gy, 1);
            std::memcpy(out, b_out.mapped, out_bytes);
            free_buf(b_pk);
        }

        free_buf(b_q);
        free_buf(b_nm);
        free_buf(b_cen);
        free_buf(b_out);
    }

    void qjl_score(const float* q_sketch, const uint8_t* signs,
                   const float* res_norms, const float* mse_in,
                   int BH, int N, int D, float qjl_scale,
                   float* out) override {
        const int packed_d_signs = (D + 7) / 8;
        const VkDeviceSize q_bytes   = (VkDeviceSize)BH * D * sizeof(float);
        const VkDeviceSize sg_bytes  = (VkDeviceSize)BH * N * packed_d_signs;
        const VkDeviceSize sg_padded = (sg_bytes + 3) & ~VkDeviceSize(3);
        const VkDeviceSize rn_bytes  = (VkDeviceSize)BH * N * sizeof(float);
        const VkDeviceSize ms_bytes  = (VkDeviceSize)BH * N * sizeof(float);
        const VkDeviceSize out_bytes = (VkDeviceSize)BH * N * sizeof(float);

        Buf b_q  = make_storage_with_data(q_sketch,  q_bytes);
        Buf b_sg = make_storage_buffer(sg_padded, /*host_visible=*/true);
        std::memcpy(b_sg.mapped, signs, sg_bytes);
        Buf b_rn = make_storage_with_data(res_norms, rn_bytes);
        Buf b_ms = make_storage_with_data(mse_in,    ms_bytes);
        Buf b_out= make_storage_buffer(out_bytes, /*host_visible=*/true);

        struct PC { int BH; int N; int D; int PACKED_D_SIGNS; float QJL_SCALE; } pc;
        pc.BH = BH; pc.N = N; pc.D = D; pc.PACKED_D_SIGNS = packed_d_signs;
        pc.QJL_SCALE = qjl_scale;

        VkBuffer bufs[] = { b_q.buffer, b_sg.buffer, b_rn.buffer,
                            b_ms.buffer, b_out.buffer };
        uint32_t gx = (N + 63) / 64;
        uint32_t gy = BH;
        run_dispatch(pl_qjl_, bufs, 5, &pc, sizeof(pc), gx, gy, 1);

        std::memcpy(out, b_out.mapped, out_bytes);

        free_buf(b_q); free_buf(b_sg); free_buf(b_rn);
        free_buf(b_ms); free_buf(b_out);
    }

    void value_dequant(const uint8_t* packed, const float* scales,
                       const float* zeros, int N, int D, int bits,
                       int group_size, float* out) override {
        int vals_per_byte, eff_bits, mask;
        switch (bits) {
            case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3;  break;
            case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF;  break;
            case 8: default:
                    vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
        }
        int n_groups = D / group_size;
        int packed_d = D / vals_per_byte;

        VkDeviceSize pk_bytes_logical = (VkDeviceSize)N * packed_d;
        VkDeviceSize pk_bytes_padded  = (pk_bytes_logical + 3) & ~VkDeviceSize(3);
        VkDeviceSize sc_bytes  = (VkDeviceSize)N * n_groups * sizeof(float);
        VkDeviceSize ze_bytes  = (VkDeviceSize)N * n_groups * sizeof(float);
        VkDeviceSize out_bytes = (VkDeviceSize)N * D * sizeof(float);

        Buf b_pk = make_storage_buffer(pk_bytes_padded, /*host_visible=*/true);
        std::memcpy(b_pk.mapped, packed, pk_bytes_logical);
        Buf b_sc = make_storage_with_data(scales, sc_bytes);
        Buf b_ze = make_storage_with_data(zeros,  ze_bytes);
        Buf b_out= make_storage_buffer(out_bytes, /*host_visible=*/true);

        int pc[8] = { N, D, eff_bits, vals_per_byte, mask,
                      group_size, n_groups, packed_d };
        VkBuffer bufs[] = { b_pk.buffer, b_sc.buffer, b_ze.buffer, b_out.buffer };
        uint32_t gx = (D + 63) / 64;
        uint32_t gy = N;
        run_dispatch(pl_vdq_, bufs, 4, &pc, sizeof(pc), gx, gy, 1);

        std::memcpy(out, b_out.mapped, out_bytes);

        free_buf(b_pk); free_buf(b_sc); free_buf(b_ze); free_buf(b_out);
    }

private:
    // --------------------------------------------------------------------
    // Pipeline construction
    // --------------------------------------------------------------------
    bool build_pipeline(Pipeline& p,
                        const unsigned char* spv, size_t spv_size,
                        int n_bindings, uint32_t push_size) {
        // Shader module.
        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv_size;
        smci.pCode    = reinterpret_cast<const uint32_t*>(spv);
        VK_OK_RET(vkCreateShaderModule(device_, &smci, nullptr, &p.module), false);

        // Descriptor set layout (all storage buffers).
        std::vector<VkDescriptorSetLayoutBinding> bindings(n_bindings);
        for (int i = 0; i < n_bindings; ++i) {
            bindings[i].binding         = (uint32_t)i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = (uint32_t)bindings.size();
        dlci.pBindings    = bindings.data();
        VK_OK_RET(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &p.dset_layout), false);

        // Pipeline layout: 1 set + push constants.
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = push_size;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &p.dset_layout;
        plci.pushConstantRangeCount = push_size > 0 ? 1u : 0u;
        plci.pPushConstantRanges    = push_size > 0 ? &pcr : nullptr;
        VK_OK_RET(vkCreatePipelineLayout(device_, &plci, nullptr, &p.pipe_layout), false);

        // Compute pipeline.
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.layout = p.pipe_layout;
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = p.module;
        cpci.stage.pName  = "main";
        VK_OK_RET(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline), false);

        p.n_bindings = n_bindings;
        p.push_size  = push_size;
        return true;
    }

    void destroy_pipeline(Pipeline& p) {
        if (p.pipeline)    vkDestroyPipeline           (device_, p.pipeline,    nullptr);
        if (p.pipe_layout) vkDestroyPipelineLayout     (device_, p.pipe_layout, nullptr);
        if (p.dset_layout) vkDestroyDescriptorSetLayout(device_, p.dset_layout, nullptr);
        if (p.module)      vkDestroyShaderModule       (device_, p.module,      nullptr);
        p = {};
    }

    // --------------------------------------------------------------------
    // Memory + buffers
    // --------------------------------------------------------------------
    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mem_props_.memoryTypes[i].propertyFlags & want) == want) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    Buf make_storage_buffer(VkDeviceSize bytes, bool host_visible) {
        Buf b{};
        if (bytes == 0) bytes = 4;  // Vulkan disallows zero-size buffers.
        b.size = bytes;

        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = bytes;
        bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_OK_RET(vkCreateBuffer(device_, &bci, nullptr, &b.buffer), b);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device_, b.buffer, &mr);

        VkMemoryPropertyFlags want;
        if (host_visible) {
            want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        } else {
            want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        uint32_t mt = find_memory_type(mr.memoryTypeBits, want);
        if (mt == UINT32_MAX && host_visible) {
            // Fallback: just host-visible without coherent.
            want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            mt   = find_memory_type(mr.memoryTypeBits, want);
        }
        if (mt == UINT32_MAX) {
            std::fprintf(stderr, "[vulkan] no memory type for buffer (host_visible=%d)\n", (int)host_visible);
            return b;
        }

        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = mt;
        VK_OK_RET(vkAllocateMemory(device_, &mai, nullptr, &b.memory), b);
        VK_OK_RET(vkBindBufferMemory(device_, b.buffer, b.memory, 0),  b);

        if (host_visible) {
            VK_OK_RET(vkMapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), b);
        }
        return b;
    }

    Buf make_storage_with_data(const void* data, VkDeviceSize bytes) {
        // Host-visible storage buffer that we copy into directly. Avoids the
        // staging-buffer round trip; correctness first.
        Buf b = make_storage_buffer(bytes, /*host_visible=*/true);
        if (b.mapped && data && bytes > 0) {
            std::memcpy(b.mapped, data, bytes);
        }
        return b;
    }

    void free_buf(Buf& b) {
        if (b.mapped) {
            vkUnmapMemory(device_, b.memory);
            b.mapped = nullptr;
        }
        if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
        if (b.memory) vkFreeMemory   (device_, b.memory, nullptr);
        b = {};
    }

    // --------------------------------------------------------------------
    // Dispatch
    // --------------------------------------------------------------------
    void run_dispatch(Pipeline& p,
                      VkBuffer* bufs, int n_bufs,
                      const void* push, uint32_t push_size,
                      uint32_t gx, uint32_t gy, uint32_t gz) {
        // Allocate descriptor set.
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = dpool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &p.dset_layout;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        VK_OK(vkAllocateDescriptorSets(device_, &dsai, &dset));

        std::vector<VkDescriptorBufferInfo> bi(n_bufs);
        std::vector<VkWriteDescriptorSet>   ws(n_bufs);
        for (int i = 0; i < n_bufs; ++i) {
            bi[i].buffer = bufs[i];
            bi[i].offset = 0;
            bi[i].range  = VK_WHOLE_SIZE;
            ws[i] = {};
            ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet          = dset;
            ws[i].dstBinding      = (uint32_t)i;
            ws[i].descriptorCount = 1;
            ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[i].pBufferInfo     = &bi[i];
        }
        vkUpdateDescriptorSets(device_, (uint32_t)ws.size(), ws.data(), 0, nullptr);

        // Allocate command buffer.
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cmd_pool_;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_OK(vkAllocateCommandBuffers(device_, &cbai, &cmd));

        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_OK(vkBeginCommandBuffer(cmd, &cbi));

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                p.pipe_layout, 0, 1, &dset, 0, nullptr);
        if (push_size > 0) {
            vkCmdPushConstants(cmd, p.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, push_size, push);
        }
        vkCmdDispatch(cmd, gx, gy, gz);

        VK_OK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        VK_OK(vkCreateFence(device_, &fci, nullptr, &fence));
        VK_OK(vkQueueSubmit(queue_, 1, &si, fence));
        VK_OK(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device_, fence, nullptr);

        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
        vkFreeDescriptorSets(device_, dpool_, 1, &dset);
    }

    // --------------------------------------------------------------------
    // State
    // --------------------------------------------------------------------
    VkInstance       instance_     = VK_NULL_HANDLE;
    VkPhysicalDevice phys_         = VK_NULL_HANDLE;
    VkDevice         device_       = VK_NULL_HANDLE;
    VkQueue          queue_        = VK_NULL_HANDLE;
    uint32_t         queue_family_ = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties mem_props_{};
    VkCommandPool    cmd_pool_     = VK_NULL_HANDLE;
    VkDescriptorPool dpool_        = VK_NULL_HANDLE;

    Pipeline pl_score_uchar_{}, pl_score_u32_b3_{}, pl_qjl_{};
    Pipeline pl_encode_uchar_{}, pl_encode_u32_b3_{};
    Pipeline pl_vdq_{}, pl_rotate_{};
};

}  // namespace

std::unique_ptr<IBackend> create_vulkan_backend() {
    return std::make_unique<VulkanBackend>();
}

}  // namespace turboquant
