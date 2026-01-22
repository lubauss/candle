// Quantized gather_mm kernel for MoE (Mixture of Experts) acceleration
// This kernel solves the threadgroup memory overflow issue in kernel_mul_mm_id
// by storing indices in device memory instead of threadgroup memory.
//
// The original kernel_mul_mm_id uses threadgroup memory for rowids:
//   threadgroup ushort2 rowids[max_experts][max_tokens]
// which is limited to 32KB on Metal, causing hangs for large images (>5000 tokens).
//
// This kernel accepts pre-filtered indices from device memory (no limit).
//
// Copyright 2024-2025 Based on candle-metal-kernels quantized.metal

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// Block size constants (match kernel_mul_mm_id for compatibility)
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4
#define THREAD_MAT_N 2
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2
#define THREAD_PER_COL 4
#define SG_MAT_SIZE 64
#define SG_MAT_ROW 8

// Quantization constants
#define QK_K 256
#define K_SCALE_SIZE 12

///////////////////////////////////////////////////////////////////////////////
// Quantized block structures (must match quantized.metal)
///////////////////////////////////////////////////////////////////////////////

typedef struct {
    half    d;           // delta
    half    m;           // min
    uint8_t qs[QK_K/2];  // quants
    uint8_t scales[QK_K/32]; // scales
} block_q4_0_k;

// Q4_K: 4-bit quantization with K-quants (144 bytes per 256 values)
typedef struct {
    union {
        struct {
            half d;    // super-block scale for quantized scales
            half dmin; // super-block scale for quantized mins
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants (128 bytes for 256 values)
} block_q4_K;

// Q8_0: 8-bit quantization (simple)
typedef struct {
    half  d;         // delta
    int8_t qs[32];   // quants
} block_q8_0;

// Q4_0: 4-bit quantization (simple, 18 bytes per 32 values)
typedef struct {
    half d;           // delta
    uint8_t qs[16];   // quants (4-bit packed)
} block_q4_0;

///////////////////////////////////////////////////////////////////////////////
// Parameters structure for gather_qmm kernel
///////////////////////////////////////////////////////////////////////////////

struct GatherQmmParams {
    int M;                    // Number of rows (tokens)
    int N;                    // Output features per expert
    int K;                    // Input features (hidden dim)
    int num_experts;          // Total number of experts
    int blocks_per_row;       // Number of quantized blocks per weight row
};

///////////////////////////////////////////////////////////////////////////////
// Helper functions for Q4_K dequantization
///////////////////////////////////////////////////////////////////////////////

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Dequantize 16 values from a Q4_K block at position il (0-15)
template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Simple gather_qmm kernel for Q4_K
// One threadgroup per output row, processes all N output columns
// Uses device memory for expert indices (no threadgroup limit!)
///////////////////////////////////////////////////////////////////////////////

[[kernel, host_name("gather_qmm_q4_K_simple")]]
void gather_qmm_q4_K_simple(
    device const block_q4_K * weights [[buffer(0)]],  // [num_experts, N, K/256] blocks
    device const float      * input   [[buffer(1)]],  // [M, K] float activations
    device       float      * output  [[buffer(2)]],  // [M, N] float output
    device const uint32_t   * indices [[buffer(3)]],  // [M] expert index per row
    constant GatherQmmParams& params  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]]) {

    const int row = tgid.y;
    const int col = tid.x;

    if (row >= params.M || col >= params.N) {
        return;
    }

    // Get expert index from device memory (no threadgroup limit!)
    const uint32_t expert = indices[row];
    if (expert >= uint32_t(params.num_experts)) {
        output[row * params.N + col] = 0.0f;
        return;
    }

    // Calculate weight offset for this expert and output column
    // Weights layout: [num_experts, N, K/256] where each block is 144 bytes
    const int blocks_per_expert_row = params.K / QK_K;  // blocks per weight row
    const int blocks_per_expert = params.N * blocks_per_expert_row;

    device const block_q4_K * expert_weights = weights + expert * blocks_per_expert;
    device const block_q4_K * col_weights = expert_weights + col * blocks_per_expert_row;

    // Input row pointer
    device const float * input_row = input + row * params.K;

    // Accumulate dot product
    float acc = 0.0f;

    // Process each Q4_K block (256 values each)
    for (int block_idx = 0; block_idx < blocks_per_expert_row; block_idx++) {
        device const block_q4_K * b = col_weights + block_idx;
        const int k_base = block_idx * QK_K;

        // Extract scale and min from the block
        const float d = float(b->dm.x);    // scale
        const float dmin = float(b->dm.y); // min

        // Process all 256 values in this block
        // Q4_K has 8 groups of 32 values, each with its own scale/min
        for (int group = 0; group < 8; group++) {
            // Get scale and min for this group (6-bit quantized in scales array)
            int is = group;
            uchar2 sc;
            if (is < 4) {
                sc = uchar2{uchar(b->scales[is] & 63), uchar(b->scales[is + 4] & 63)};
            } else {
                sc = uchar2{
                    uchar((b->scales[is + 4] & 0xF) | ((b->scales[is - 4] & 0xc0) >> 2)),
                    uchar((b->scales[is + 4] >> 4) | ((b->scales[is] & 0xc0) >> 2))
                };
            }

            const float scale = d * float(sc[0]);
            const float min_val = dmin * float(sc[1]);

            // Process 32 values in this group
            const int q_offset = group * 16;  // 32 values packed into 16 bytes
            for (int i = 0; i < 32; i++) {
                const int k = k_base + group * 32 + i;
                if (k >= params.K) break;

                // Extract 4-bit quant
                const int byte_idx = q_offset + i / 2;
                uint8_t q = b->qs[byte_idx];
                q = (i % 2 == 0) ? (q & 0x0F) : (q >> 4);

                // Dequantize: weight = scale * q - min
                const float weight = scale * float(q) - min_val;
                acc += weight * input_row[k];
            }
        }
    }

    output[row * params.N + col] = acc;
}

///////////////////////////////////////////////////////////////////////////////
// Tiled gather_qmm kernel for Q4_K
// Uses threadgroup memory for input tiling, device memory for indices
///////////////////////////////////////////////////////////////////////////////

template <int BK = 64, int TN = 4>
[[kernel]] void gather_qmm_q4_K_tiled(
    device const block_q4_K * weights [[buffer(0)]],
    device const float      * input   [[buffer(1)]],
    device       float      * output  [[buffer(2)]],
    device const uint32_t   * indices [[buffer(3)]],
    constant GatherQmmParams& params  [[buffer(4)]],
    uint tgid_y [[threadgroup_position_in_grid]],
    uint tid_x [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {

    const int row = tgid_y;
    if (row >= params.M) {
        return;
    }

    // Get expert index from device memory
    const uint32_t expert = indices[row];
    if (expert >= uint32_t(params.num_experts)) {
        // Zero out this row
        for (int c = tid_x; c < params.N; c += threads_per_group) {
            output[row * params.N + c] = 0.0f;
        }
        return;
    }

    const int K = params.K;
    const int N = params.N;
    const int blocks_per_expert_row = K / QK_K;
    const int blocks_per_expert = N * blocks_per_expert_row;

    device const block_q4_K * expert_weights = weights + expert * blocks_per_expert;
    device const float * input_row = input + row * K;
    device float * output_row = output + row * N;

    // Each thread handles TN output columns
    const int col_start = tid_x * TN;
    if (col_start >= N) {
        return;
    }

    // Accumulators for TN outputs
    float sums[TN] = {0.0f};

    // Threadgroup memory for input tile
    threadgroup float input_shared[BK];

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_base = kt * BK;

        // Cooperatively load input tile into shared memory
        for (int i = tid_x; i < BK && (k_base + i) < K; i += threads_per_group) {
            input_shared[i] = input_row[k_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int k_end = min(BK, K - k_base);

        // Process TN output columns
        for (int j = 0; j < TN && (col_start + j) < N; j++) {
            device const block_q4_K * col_weights = expert_weights + (col_start + j) * blocks_per_expert_row;

            // Determine which blocks overlap with this k_tile
            const int block_start = k_base / QK_K;
            const int block_end = (k_base + k_end + QK_K - 1) / QK_K;

            for (int block_idx = block_start; block_idx < block_end && block_idx < blocks_per_expert_row; block_idx++) {
                device const block_q4_K * b = col_weights + block_idx;
                const int block_k_start = block_idx * QK_K;

                const float d = float(b->dm.x);
                const float dmin = float(b->dm.y);

                // Process groups that overlap with current tile
                for (int group = 0; group < 8; group++) {
                    const int group_k_start = block_k_start + group * 32;
                    const int group_k_end = group_k_start + 32;

                    // Skip if group doesn't overlap with tile
                    if (group_k_end <= k_base || group_k_start >= k_base + k_end) {
                        continue;
                    }

                    // Get scale and min for this group
                    int is = group;
                    uchar2 sc;
                    if (is < 4) {
                        sc = uchar2{uchar(b->scales[is] & 63), uchar(b->scales[is + 4] & 63)};
                    } else {
                        sc = uchar2{
                            uchar((b->scales[is + 4] & 0xF) | ((b->scales[is - 4] & 0xc0) >> 2)),
                            uchar((b->scales[is + 4] >> 4) | ((b->scales[is] & 0xc0) >> 2))
                        };
                    }

                    const float scale = d * float(sc[0]);
                    const float min_val = dmin * float(sc[1]);

                    const int q_offset = group * 16;
                    for (int i = 0; i < 32; i++) {
                        const int k = group_k_start + i;
                        if (k < k_base || k >= k_base + k_end || k >= K) continue;

                        const int byte_idx = q_offset + i / 2;
                        uint8_t q = b->qs[byte_idx];
                        q = (i % 2 == 0) ? (q & 0x0F) : (q >> 4);

                        const float weight = scale * float(q) - min_val;
                        sums[j] += weight * input_shared[k - k_base];
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results
    for (int j = 0; j < TN && (col_start + j) < N; j++) {
        output_row[col_start + j] = sums[j];
    }
}

///////////////////////////////////////////////////////////////////////////////
// SIMD-optimized gather_qmm kernel for Q4_K
// Uses simdgroup operations for better performance on larger matrices
///////////////////////////////////////////////////////////////////////////////

[[kernel, host_name("gather_qmm_q4_K_simd")]]
void gather_qmm_q4_K_simd(
    device const block_q4_K * weights [[buffer(0)]],
    device const float      * input   [[buffer(1)]],
    device       float      * output  [[buffer(2)]],
    device const uint32_t   * indices [[buffer(3)]],
    constant GatherQmmParams& params  [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

    const int row = tgid.y;
    const int col_block = tgid.x * 128;  // Each threadgroup handles 128 columns

    if (row >= params.M) {
        return;
    }

    // Get expert index from device memory
    const uint32_t expert = indices[row];

    const int K = params.K;
    const int N = params.N;

    // Each simdgroup handles 32 columns
    const int col = col_block + simd_group_id * 32 + simd_lane_id;

    if (col >= N || expert >= uint32_t(params.num_experts)) {
        if (col < N) {
            output[row * N + col] = 0.0f;
        }
        return;
    }

    const int blocks_per_expert_row = K / QK_K;
    const int blocks_per_expert = N * blocks_per_expert_row;

    device const block_q4_K * expert_weights = weights + expert * blocks_per_expert;
    device const block_q4_K * col_weights = expert_weights + col * blocks_per_expert_row;
    device const float * input_row = input + row * K;

    float acc = 0.0f;

    // Process each Q4_K block
    for (int block_idx = 0; block_idx < blocks_per_expert_row; block_idx++) {
        device const block_q4_K * b = col_weights + block_idx;
        const int k_base = block_idx * QK_K;

        const float d = float(b->dm.x);
        const float dmin = float(b->dm.y);

        // Process 8 groups of 32 values
        for (int group = 0; group < 8; group++) {
            int is = group;
            uchar2 sc;
            if (is < 4) {
                sc = uchar2{uchar(b->scales[is] & 63), uchar(b->scales[is + 4] & 63)};
            } else {
                sc = uchar2{
                    uchar((b->scales[is + 4] & 0xF) | ((b->scales[is - 4] & 0xc0) >> 2)),
                    uchar((b->scales[is + 4] >> 4) | ((b->scales[is] & 0xc0) >> 2))
                };
            }

            const float scale = d * float(sc[0]);
            const float min_val = dmin * float(sc[1]);

            // Vectorized processing of 32 values
            const int q_offset = group * 16;
            const int k_group = k_base + group * 32;

            // Process 8 values at a time using float4
            for (int i = 0; i < 32; i += 4) {
                if (k_group + i + 3 >= K) {
                    // Handle remainder
                    for (int r = i; r < 32 && k_group + r < K; r++) {
                        const int byte_idx = q_offset + r / 2;
                        uint8_t q = b->qs[byte_idx];
                        q = (r % 2 == 0) ? (q & 0x0F) : (q >> 4);
                        acc += (scale * float(q) - min_val) * input_row[k_group + r];
                    }
                    break;
                }

                // Extract 4 quantized values
                float4 weights_vec;
                for (int v = 0; v < 4; v++) {
                    const int byte_idx = q_offset + (i + v) / 2;
                    uint8_t q = b->qs[byte_idx];
                    q = ((i + v) % 2 == 0) ? (q & 0x0F) : (q >> 4);
                    weights_vec[v] = scale * float(q) - min_val;
                }

                // Load input values
                float4 input_vec = float4(
                    input_row[k_group + i],
                    input_row[k_group + i + 1],
                    input_row[k_group + i + 2],
                    input_row[k_group + i + 3]
                );

                acc += dot(weights_vec, input_vec);
            }
        }
    }

    output[row * N + col] = acc;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel instantiations (only for templated kernels)
///////////////////////////////////////////////////////////////////////////////

// Tiled kernel - instantiate template with BK=64, TN=4
template [[host_name("gather_qmm_q4_K_tiled")]]
[[kernel]] void gather_qmm_q4_K_tiled<64, 4>(
    device const block_q4_K * weights [[buffer(0)]],
    device const float      * input   [[buffer(1)]],
    device       float      * output  [[buffer(2)]],
    device const uint32_t   * indices [[buffer(3)]],
    constant GatherQmmParams& params  [[buffer(4)]],
    uint tgid_y [[threadgroup_position_in_grid]],
    uint tid_x [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);
