//! Quantized gather_mm for MoE - stores indices in device memory (no threadgroup limit)
//!
//! This kernel solves the threadgroup memory overflow issue in kernel_mul_mm_id
//! for large image inputs (>1MB) that generate 5000+ tokens.
//!
//! The original kernel_mul_mm_id uses threadgroup memory:
//!   threadgroup ushort2 rowids[max_experts][max_tokens]
//! which is limited to 32KB on Metal, causing hangs for large images.
//!
//! This kernel accepts pre-filtered indices from device memory (no limit).

use crate::metal::{Buffer, ComputeCommandEncoder, Device};
use crate::utils::EncoderProvider;
use crate::{set_params, EncoderParam, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

pub use super::quantized::GgmlDType;

/// Kernel variant for gather_qmm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatherQmmKernel {
    /// Simple kernel: one thread per output element
    /// Best for small outputs or debugging
    Simple,
    /// Tiled kernel: uses threadgroup memory for input
    /// Better for medium-sized outputs
    Tiled,
    /// SIMD kernel: uses vectorized loads
    /// Best for large outputs
    Simd,
}

impl Default for GatherQmmKernel {
    fn default() -> Self {
        // SIMD kernel is generally the best performer
        Self::Simd
    }
}

/// Parameters for gather_qmm kernel (must match Metal struct GatherQmmParams)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GatherQmmParams {
    m: i32,              // Number of rows (tokens)
    n: i32,              // Output features per expert
    k: i32,              // Input features (hidden dim)
    num_experts: i32,    // Total number of experts
    blocks_per_row: i32, // Number of quantized blocks per weight row
}

impl EncoderParam for GatherQmmParams {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Call the quantized gather_qmm kernel for MoE acceleration.
///
/// This kernel fuses the index_select (gather) and quantized matmul operations,
/// using device memory for indices to avoid the 32KB threadgroup limit.
///
/// # Arguments
/// * `device` - Metal device
/// * `ep` - Encoder provider
/// * `kernels` - Kernel cache
/// * `dtype` - Quantization type (Q4_K, Q4_0, Q8_0)
/// * `(m, n, k, num_experts)` - Matrix dimensions:
///   - m: number of rows (tokens)
///   - n: output features per expert
///   - k: input features (hidden dimension)
///   - num_experts: number of expert weight matrices
/// * `weights` - Quantized expert weights [num_experts, n, k/block_size] blocks
/// * `weights_offset` - Byte offset into weights buffer
/// * `input` - Input activations [m, k] float
/// * `input_offset` - Byte offset into input buffer
/// * `indices` - Expert indices [m], one per row (u32)
/// * `indices_offset` - Byte offset into indices buffer
/// * `output` - Output buffer [m, n] float
#[allow(clippy::too_many_arguments)]
pub fn call_gather_qmm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (m, n, k, num_experts): (usize, usize, usize, usize),
    weights: &Buffer,
    weights_offset: usize,
    input: &Buffer,
    input_offset: usize,
    indices: &Buffer,
    indices_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_gather_qmm_with_kernel(
        device,
        ep,
        kernels,
        dtype,
        GatherQmmKernel::default(),
        (m, n, k, num_experts),
        weights,
        weights_offset,
        input,
        input_offset,
        indices,
        indices_offset,
        output,
    )
}

/// Call gather_qmm with a specific kernel variant
#[allow(clippy::too_many_arguments)]
pub fn call_gather_qmm_with_kernel(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    kernel_variant: GatherQmmKernel,
    (m, n, k, num_experts): (usize, usize, usize, usize),
    weights: &Buffer,
    weights_offset: usize,
    input: &Buffer,
    input_offset: usize,
    indices: &Buffer,
    indices_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // Only Q4_K is currently supported
    let block_size = match dtype {
        GgmlDType::Q4K => 256, // QK_K = 256 values per block
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp("non-Q4K", "gather_qmm"));
        }
    };

    let blocks_per_row = k / block_size;

    let params = GatherQmmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        num_experts: num_experts as i32,
        blocks_per_row: blocks_per_row as i32,
    };

    let (kernel_name, grid_size, group_size) = match kernel_variant {
        GatherQmmKernel::Simple => {
            // Simple kernel: one thread per output element
            let name = match dtype {
                GgmlDType::Q4K => "gather_qmm_q4_K_simple",
                _ => unreachable!(),
            };
            let grid = MTLSize {
                width: n,
                height: m,
                depth: 1,
            };
            let group = MTLSize {
                width: 32.min(n),
                height: 1,
                depth: 1,
            };
            (name, grid, group)
        }
        GatherQmmKernel::Tiled => {
            // Tiled kernel: one threadgroup per row
            // Each thread handles TN=4 columns
            let name = match dtype {
                GgmlDType::Q4K => "gather_qmm_q4_K_tiled",
                _ => unreachable!(),
            };
            let threads_per_row = (n + 3) / 4; // Each thread handles 4 columns
            let threads_per_group = 256.min(threads_per_row);
            let grid = MTLSize {
                width: 1,
                height: m,
                depth: 1,
            };
            let group = MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            };
            (name, grid, group)
        }
        GatherQmmKernel::Simd => {
            // SIMD kernel: 4 simdgroups per threadgroup, 32 threads per simdgroup
            // Each threadgroup handles 128 columns
            let name = match dtype {
                GgmlDType::Q4K => "gather_qmm_q4_K_simd",
                _ => unreachable!(),
            };
            let col_tiles = (n + 127) / 128;
            let grid = MTLSize {
                width: col_tiles,
                height: m,
                depth: 1,
            };
            let group = MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            }; // 32 threads per simdgroup, 4 simdgroups
            (name, grid, group)
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::GatherQmm, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (weights, weights_offset),
            (input, input_offset),
            output,
            (indices, indices_offset),
            params
        )
    );

    encoder.use_resource(weights, MTLResourceUsage::Read);
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(indices, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);

    Ok(())
}

/// Parameters for parallel gather_qmm kernel (must match Metal struct GatherQmmParallelParams)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GatherQmmParallelParams {
    m: i32,              // Number of tokens
    n: i32,              // Output features per expert
    k: i32,              // Input features (hidden dim)
    num_experts: i32,    // Total number of experts
    top_k: i32,          // Number of experts per token
    blocks_per_row: i32, // Number of quantized blocks per weight row
}

impl EncoderParam for GatherQmmParallelParams {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Call the parallel expert dispatch gather_qmm kernel for MoE acceleration.
///
/// This kernel processes ALL top-k experts for each token in parallel using 3D dispatch.
/// This is the key optimization for MoE models - instead of processing experts sequentially,
/// all 8 experts (for Qwen3-30B) are computed simultaneously.
///
/// # Arguments
/// * `device` - Metal device
/// * `ep` - Encoder provider
/// * `kernels` - Kernel cache
/// * `dtype` - Quantization type (Q4_K)
/// * `(m, n, k, num_experts, top_k)` - Matrix dimensions:
///   - m: number of tokens
///   - n: output features per expert
///   - k: input features (hidden dimension)
///   - num_experts: total number of expert weight matrices
///   - top_k: number of experts selected per token
/// * `weights` - Quantized expert weights [num_experts, n, k/block_size] blocks
/// * `weights_offset` - Byte offset into weights buffer
/// * `input` - Input activations [m, k] float
/// * `input_offset` - Byte offset into input buffer
/// * `indices` - Expert indices [m, top_k] (u32), one per (token, expert_slot)
/// * `indices_offset` - Byte offset into indices buffer
/// * `output` - Output buffer [m, top_k, n] float
#[allow(clippy::too_many_arguments)]
pub fn call_gather_qmm_parallel(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (m, n, k, num_experts, top_k): (usize, usize, usize, usize, usize),
    weights: &Buffer,
    weights_offset: usize,
    input: &Buffer,
    input_offset: usize,
    indices: &Buffer,
    indices_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // Only Q4_K is currently supported
    let block_size = match dtype {
        GgmlDType::Q4K => 256, // QK_K = 256 values per block
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "non-Q4K",
                "gather_qmm_parallel",
            ));
        }
    };

    let blocks_per_row = k / block_size;

    let params = GatherQmmParallelParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        num_experts: num_experts as i32,
        top_k: top_k as i32,
        blocks_per_row: blocks_per_row as i32,
    };

    // 3D dispatch: (col_tiles, num_tokens, top_k)
    // Each threadgroup handles 128 columns for one (token, expert_slot) pair
    let col_tiles = (n + 127) / 128;
    let grid = MTLSize {
        width: col_tiles,
        height: m,
        depth: top_k,
    };
    let group = MTLSize {
        width: 32,
        height: 4,
        depth: 1,
    }; // 32 threads per simdgroup, 4 simdgroups per threadgroup

    let kernel_name = match dtype {
        GgmlDType::Q4K => "gather_qmm_q4_K_parallel",
        _ => unreachable!(),
    };

    let pipeline = kernels.load_pipeline(device, Source::GatherQmm, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (weights, weights_offset),
            (input, input_offset),
            output,
            (indices, indices_offset),
            params
        )
    );

    encoder.use_resource(weights, MTLResourceUsage::Read);
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(indices, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid, group);

    Ok(())
}
