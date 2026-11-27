// Block-sparse attention over a Q8_0 K/V cache (QSA): every query attends to its own list of
// ratio-cell blocks, read straight from the cache. No gathered K/V copy and no cache-sized mask,
// so the cost follows the selected budget instead of the context length; the same kernel serves
// decode (a few queries, split over many blocks) and prompt batches.
//
// Expressed as a GGML_OP_FLASH_ATTN_EXT node with src[5] = block ids (see ggml_cuda_qsa_attn_is_sparse):
//   src[0] q     F32  [D, n_head, n_tok]
//   src[1] k     Q8_0 [D, n_head_kv, n_cells]   (a cache view; cell = block*ratio + c)
//   src[2] v     Q8_0 [D, n_head_kv, n_cells]
//   src[3] mask  F16  [n_wblk*ratio, 1, 1, n_tok]  0 or -inf per selected cell
//   src[5] ids   I32  [n_wblk, n_tok]
//   dst          F32  [D, n_head, n_tok]
//   op_params: [0] scale, [6] ratio
//
// [kv-stream] a streamed cache (LLAMA_KV_STREAM_CELLS, see llama-kv-cache.h) adds
//   src[6] map   I32  the layer's residency map: ctl[16] | page_table[n_cold] | slot_block[S] | slot_stamp[S] |
//                     slot_ref[S] | miss_block[S] | miss_slot[S], page_table[cold block] = slot or -1
//   src[7] prm   I32  [4]: [0] first cell that is intact in the ring
//   op_params: [7] 1 = every cell is still at its own index, 2 = resolve first; [8..9] / [10..11] the host K / V
//              pointers (this stream's cells from P on, in pinned memory); [12] P, [13] T, [14] S, [15] n_cold
// The device K/V tensors then hold, in blocks of 4 cells: [0, P/4) the first cells at their own index,
// [P/4, P/4 + T/4) a ring with the most recent cells, and S slots for older ("cold") blocks after it.
// Mode 2 attends in chunks of queries that name at most S blocks; per chunk, qsa_stream_resolve makes every cold
// block resident and rewrites the ids to physical blocks, and qsa_stream_copy reads the missing blocks from host
// memory into their slots. All of it on the device, no readback. This is the CUDA port of
// ggml-vulkan/vulkan-shaders/qsa_stream.comp and ggml_vk_qsa_block_attn: keep the two in step.

#include "qsa-attn.cuh"
#include "common.cuh"

#include "qsa-attn-kernels.cuh" // [qsa-dp4a] the device code lives there
#include <atomic> // [qsa-hpw6] one bounded selection diagnostic

bool ggml_cuda_qsa_attn_is_sparse(const ggml_tensor * dst) {
    return dst->op == GGML_OP_FLASH_ATTN_EXT && dst->src[5] != nullptr && dst->src[5]->type == GGML_TYPE_I32;
}

bool ggml_cuda_qsa_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    const ggml_tensor * ids = dst->src[5];
    if (q->type != GGML_TYPE_F32 || (k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_F16) || v->type != k->type || !m || m->type != GGML_TYPE_F16) {
        return false;
    }
    if (q->ne[0] != QSA_ATTN_D || k->ne[0] != QSA_ATTN_D || v->ne[0] != QSA_ATTN_D || ggml_get_op_params_i32(dst, 6) != QSA_ATTN_R) {
        return false;
    }
    const int gqa = q->ne[1] / k->ne[1];
    // the head groupings instantiated below: 24/12/8/6/4/3/2/1 query heads per KV head (6: 4 GPUs on the KV-duplicated model)
    if (q->ne[1] % k->ne[1] != 0 || !(gqa == 24 || gqa == 16 || gqa == 12 || gqa == 8 || gqa == 6 || gqa == 4 || gqa == 3 ||
            gqa == 2 || gqa == 1)) {
        return false;
    }
    // [kv-stream] a streamed cache: src[6] the layer's residency map, src[7] the per-ubatch parameters. Without the
    // resolve stage below, the kernel would read cold blocks at their logical index, past the rows the device holds.
    if ((dst->src[6] != nullptr) != (dst->src[7] != nullptr)) {
        return false;
    }
    if (dst->src[6]) {
        const int32_t mode = ggml_get_op_params_i32(dst, 7);
        if (dst->src[6]->type != GGML_TYPE_I32 || dst->src[7]->type != GGML_TYPE_I32 || (mode != 1 && mode != 2) ||
            !ggml_is_contiguous(dst->src[6]) || !ggml_is_contiguous(dst->src[7]) ||
            k->nb[2] % sizeof(uint32_t) != 0 || v->nb[2] % sizeof(uint32_t) != 0 ||
            ggml_get_op_params_i32(dst, 14) < ids->ne[0]) {
            return false;
        }
    }
    return q->nb[0] == sizeof(float) && ggml_is_contiguous(m) && ggml_is_contiguous(ids) &&
           ids->ne[1] == q->ne[2] && m->ne[0] == ids->ne[0]*QSA_ATTN_R && m->ne[3] == q->ne[2] &&
           dst->ne[0] == QSA_ATTN_D && dst->ne[1] == q->ne[1] && dst->ne[2] == q->ne[2] && ggml_is_contiguous(dst);
}

// [kv-stream] the map's control words (map[0..15])
#define QSA_CTL_EPOCH    0
#define QSA_CTL_HAND     1
#define QSA_CTL_MISSES   2
#define QSA_CTL_OVERFLOW 3
#define QSA_CTL_TOT_MISS 4
#define QSA_CTL_TOT_CALL 5
#define QSA_CTL_TOT_IDS  6

#define QSA_STREAM_RESOLVE_THREADS 1024
#define QSA_STREAM_COPY_THREADS    256

// [kv-stream] one block. Every cold block the ids name is made resident and the ids are rewritten to physical blocks:
// hits take this call's epoch and their reference bit, misses are claimed once, each miss gets a victim slot from a
// clock sweep that never takes a slot this call uses. Same steps as qsa_stream.comp (MODE 0).
static __global__ void qsa_stream_resolve(
        int32_t * __restrict__ ids, int32_t * __restrict__ map, const int32_t * __restrict__ prm,
        const int n_ids, const int p_blocks, const int t_blocks, const int S, const int n_cold, const int n_blocks) {
    const int tid = threadIdx.x;
    const int PT  = 16;
    const int SB  = PT + n_cold;
    const int SS  = SB + S;
    const int SR  = SS + S;
    const int MB  = SR + S;
    const int MS  = MB + S;

    __shared__ int s_nmiss;
    __shared__ int s_got;
    __shared__ int s_ovf;

    // read by every thread before thread 0 updates them at the end, after a barrier
    const int epoch = map[QSA_CTL_EPOCH] + 1;
    const int hand  = map[QSA_CTL_HAND];
    const int r_blk = prm[0] / 4;

    if (tid == 0) { s_nmiss = 0; s_got = 0; s_ovf = 0; }
    __syncthreads();

    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    for (int i = tid; i < n_ids; i += QSA_STREAM_RESOLVE_THREADS) {
        const int b = ids[i];
        if (b < p_blocks || b >= r_blk || b >= n_blocks) {
            continue;
        }
        const int cb = b - p_blocks;
        const int sl = map[PT + cb];
        if (sl >= 0) {
            map[SS + sl] = epoch;
            map[SR + sl] = 1;
        } else if (sl == -1) {
            if (atomicCAS(&map[PT + cb], -1, -2) == -1) {
                const int k = atomicAdd(&s_nmiss, 1);
                if (k < S) {
                    map[MB + k] = cb;
                }
            }
        }
    }
    __syncthreads();

    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses is never taken; a referenced one
    //    loses its bit as the hand passes and can be taken on a later pass. The exit test is read by every thread
    //    between two barriers, so no thread's atomicAdd of this pass can make the threads disagree on it.
    const int need = min(s_nmiss, S);
    const int step = min(QSA_STREAM_RESOLVE_THREADS, S);
    int scanned = 0;
    while (true) {
        __syncthreads();
        const bool done = s_got >= need || scanned >= 3*S;
        __syncthreads();
        if (done) {
            break;
        }
        if (tid < S) {
            const int j = (int) (((int64_t) hand + scanned + tid) % S);
            if (map[SS + j] != epoch) {
                if (map[SB + j] < 0 || map[SR + j] == 0) {
                    const int k = atomicAdd(&s_got, 1);
                    if (k < need) {
                        const int cb  = map[MB + k];
                        const int old = map[SB + j];
                        if (old >= 0) {
                            map[PT + old] = -1;
                        }
                        map[SB + j]  = cb;
                        map[PT + cb] = j;
                        map[SS + j]  = epoch;
                        map[SR + j]  = 1;
                        map[MS + k]  = j;
                    }
                } else {
                    map[SR + j] = 0;
                }
            }
        }
        scanned += step;
    }
    const int got = min(s_got, need);

    // 3. the ids become physical blocks. A block that found no slot (more cold blocks in one call than slots) is
    //    unclaimed again and counted: the host aborts on that count (llama_kv_cache::stream_check).
    const int base = p_blocks + t_blocks;
    for (int i = tid; i < n_ids; i += QSA_STREAM_RESOLVE_THREADS) {
        const int b = ids[i];
        if (b < 0 || b >= n_blocks) {
            ids[i] = 0; // padding, never a visible block
        } else if (b < p_blocks) {
            // resident at its own index
        } else if (b >= r_blk) {
            ids[i] = p_blocks + (b - p_blocks) % t_blocks;
        } else {
            const int cb = b - p_blocks;
            const int sl = map[PT + cb];
            if (sl >= 0) {
                ids[i] = base + sl;
            } else {
                map[PT + cb] = -1;
                ids[i] = 0;
                atomicAdd(&s_ovf, 1);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        map[QSA_CTL_EPOCH]     = epoch;
        map[QSA_CTL_HAND]      = (int) (((int64_t) hand + scanned) % S);
        map[QSA_CTL_MISSES]    = got;
        map[QSA_CTL_OVERFLOW] += s_ovf;
        map[QSA_CTL_TOT_MISS] += got;
        map[QSA_CTL_TOT_CALL] += 1;
        map[QSA_CTL_TOT_IDS]  += n_ids;
    }
}

// [kv-stream] one block per possible miss: block k copies miss k (if there is one) from the host copy, read over
// PCIe from pinned memory, into its slot. Same as qsa_stream.comp (MODE 1). host: this stream's cells from P on;
// dev: block 0 of this stream's device tensor; words: u32 per block of 4 cells.
static __global__ void qsa_stream_copy(
        const int32_t * __restrict__ map, const uint32_t * __restrict__ host, uint32_t * __restrict__ dev,
        const int S, const int n_cold, const int base, const int words) {
    const int k = blockIdx.x;
    if (k >= map[QSA_CTL_MISSES]) {
        return;
    }
    const int MB = 16 + n_cold + 3*S;
    const int MS = MB + S;
    const uint32_t * src = host + (int64_t) map[MB + k] * words;
    uint32_t       * out = dev  + (int64_t) (base + map[MS + k]) * words;
    for (int i = threadIdx.x; i < words; i += QSA_STREAM_COPY_THREADS) {
        out[i] = src[i];
    }
}

// [kv-stream] the address the device reads a pinned host pointer through
static const void * qsa_stream_host_dev_ptr(const ggml_tensor * dst, int i_param) {
    uint64_t p;
    memcpy(&p, (const char *) dst->op_params + i_param*sizeof(int32_t), sizeof(p));
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    // the vendor headers do not map the pointer attribute query; pinned host memory is mapped at the same address
    // under unified addressing
    const void * dev = (const void *) (uintptr_t) p;
#else
    // cudaMallocHost memory (the CUDA host buffer type) is mapped into the device's address space under unified
    // addressing; ask for the address rather than assume it
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, (const void *) (uintptr_t) p));
    GGML_ASSERT(attr.type == cudaMemoryTypeHost && attr.devicePointer != nullptr &&
                "[kv-stream] the host K/V store is not device-mapped pinned memory");
    const void * dev = attr.devicePointer;
#endif
    GGML_ASSERT((uintptr_t) dev % sizeof(uint32_t) == 0);
    return dev;
}

void ggml_cuda_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * ids = dst->src[5];
    GGML_ASSERT(ggml_cuda_qsa_attn_supported(dst));

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_head = q->ne[1];
    const int n_tok  = q->ne[2];
    const int gqa    = n_head / k->ne[1];
    const int n_wblk = ids->ne[0];

    // 3 heads per warp for gqa 12 (4 warps per block); other group sizes pick a divisor
    const int hpw   = gqa % 3 == 0 && gqa/3 <= 8 ? 3 : gqa % 4 == 0 && gqa/4 <= 8 ? 4 : gqa % 2 == 0 && gqa/2 <= 8 ? 2 : 1;
    const int n_grp = gqa / hpw;
    GGML_ASSERT(n_grp <= 8);

    // [kv-stream] mode 2: the queries are attended in chunks that cannot name more blocks than the layer has slots,
    // so every cold block of a chunk finds one (mode 1: every cell is still at its own index, nothing to resolve)
    const bool streamed  = dst->src[6] != nullptr && ggml_get_op_params_i32(dst, 7) == 2;
    int        tok_chunk = n_tok;
    if (streamed) {
        const int S = ggml_get_op_params_i32(dst, 14);
        GGML_ASSERT(n_wblk <= S && "[kv-stream] fewer cache slots than one query selects");
        tok_chunk = std::max(1, std::min(n_tok, S / n_wblk));
    }

    // enough blocks to fill the GPU: small batches split each query's block list
    const int n_sm    = ggml_cuda_info().devices[ctx.device].nsm;
    const int blocks0 = tok_chunk * (int) k->ne[1];
    int n_split = std::max(1, std::min(64, (n_sm*8 + blocks0 - 1) / blocks0));
    // An exhaustive list is used by dense MTP attention. Bound the work and
    // shared memory per CTA there; the usual ~512-block QSA lists are unchanged.
    if (k->type == GGML_TYPE_F16 && n_wblk > 1024) {
        n_split = std::max(n_split, std::min(512, (n_wblk + 127)/128));
    }
    n_split = std::min(n_split, n_wblk);

    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool());
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool());
    if (n_split > 1) {
        part_acc.alloc((size_t) tok_chunk*n_head*n_split*QSA_ATTN_D);
        part_ml.alloc((size_t) tok_chunk*n_head*n_split);
    }

    const dim3 block(WARP_SIZE, n_grp);
    cudaStream_t stream = ctx.stream();

    const int    per  = (n_wblk + n_split - 1) / n_split;
    const size_t smem = (size_t) per*(sizeof(int32_t) + QSA_ATTN_R*sizeof(float));
    // [qsa-dp4a] env GGML_CUDA_QSA_DP4A=1: q.k with int8 dot products (qsa_attn_kernel_dp4a), same launch shape
    static const bool use_dp4a = [] {
        const char * e = getenv("GGML_CUDA_QSA_DP4A");
        const bool on = e != nullptr && atoi(e) != 0;
        if (on) {
            GGML_LOG_WARN("[qsa-dp4a] qsa attention: q.k with int8 dot products (GGML_CUDA_QSA_DP4A=1)\n");
        }
        return on;
    }();
    // [qsa-hpw6] opt-in; never affects non-dp4a or other GQA shapes.
    static const bool use_hpw6 = [] {
        const char * e = getenv("GGML_CUDA_QSA_HPW6");
        return e != nullptr && atoi(e) != 0;
    }();

    // [kv-stream] the resolve and copy inputs (mode 2 only)
    int32_t        * map_d  = nullptr;
    const int32_t  * prm_d  = nullptr;
    const uint32_t * host_k = nullptr;
    const uint32_t * host_v = nullptr;
    int p_blocks = 0, t_blocks = 0, n_slots = 0, n_cold = 0;
    if (streamed) {
        map_d    = (int32_t *) dst->src[6]->data;
        prm_d    = (const int32_t *) dst->src[7]->data;
        host_k   = (const uint32_t *) qsa_stream_host_dev_ptr(dst,  8);
        host_v   = (const uint32_t *) qsa_stream_host_dev_ptr(dst, 10);
        p_blocks = ggml_get_op_params_i32(dst, 12) / 4;
        t_blocks = ggml_get_op_params_i32(dst, 13) / 4;
        n_slots  = ggml_get_op_params_i32(dst, 14);
        n_cold   = ggml_get_op_params_i32(dst, 15);
        GGML_ASSERT(t_blocks > 0 && (uintptr_t) k->data % sizeof(uint32_t) == 0 && (uintptr_t) v->data % sizeof(uint32_t) == 0);
    }

    for (int t0 = 0; t0 < n_tok; t0 += tok_chunk) {
        const int nt = std::min(tok_chunk, n_tok - t0);
        // this chunk's queries: the kernels index every per-token input from these bases
        const char    * q_c   = (const char *) q->data + (size_t) t0*q->nb[2];
        const half    * m_c   = (const half *) ((const char *) m->data + (size_t) t0*m->nb[3]);
        int32_t       * ids_c = (int32_t *) ((char *) ids->data + (size_t) t0*ids->nb[1]);
        float         * dst_c = (float *) dst->data + (size_t) t0*n_head*QSA_ATTN_D;

        if (streamed) {
            // the ids are rewritten in place, as ggml-vulkan does: they are this node's own input
            const int n_ids = nt*n_wblk;
            qsa_stream_resolve<<<1, QSA_STREAM_RESOLVE_THREADS, 0, stream>>>(
                ids_c, map_d, prm_d, n_ids, p_blocks, t_blocks, n_slots, n_cold, p_blocks + n_cold);
            CUDA_CHECK(cudaGetLastError());
            const int n_groups = std::min(n_slots, n_ids);
            const int base     = p_blocks + t_blocks;
            qsa_stream_copy<<<n_groups, QSA_STREAM_COPY_THREADS, 0, stream>>>(
                map_d, host_k, (uint32_t *) k->data, n_slots, n_cold, base, (int) (k->nb[2]));
            qsa_stream_copy<<<n_groups, QSA_STREAM_COPY_THREADS, 0, stream>>>(
                map_d, host_v, (uint32_t *) v->data, n_slots, n_cold, base, (int) (v->nb[2]));
            CUDA_CHECK(cudaGetLastError());
        }

        const dim3 grid(n_split, k->ne[1], nt);
#define QSA_ARGS q_c, (const char *) k->data, (const char *) v->data, m_c, \
            ids_c, dst_c, part_acc.ptr, part_ml.ptr, \
            n_head, gqa, n_wblk, n_split, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], \
            m->nb[3], ids->nb[1], scale
#define QSA_LAUNCH(HPW, NGRP) \
        if (k->type == GGML_TYPE_F16) { qsa_attn_kernel<HPW, NGRP, true><<<grid, block, smem, stream>>>(QSA_ARGS); } \
        else if (use_dp4a) { qsa_attn_kernel_dp4a<HPW, NGRP><<<grid, block, smem, stream>>>(QSA_ARGS); } \
        else          { qsa_attn_kernel<HPW, NGRP><<<grid, block, smem, stream>>>(QSA_ARGS); }
        if (k->type == GGML_TYPE_Q8_0 && use_hpw6 && use_dp4a && gqa == 12) {
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                GGML_LOG_WARN("[qsa-hpw6] selected dp4a<6,2>: tokens=%d heads=%d kv=%lld splits=%d threads=64 smem=%zu\n",
                    n_tok, n_head, (long long) k->ne[1], n_split, smem);
            }
            qsa_attn_kernel_dp4a<6, 2><<<grid, dim3(WARP_SIZE, 2), smem, stream>>>(QSA_ARGS);
        } else if (hpw == 3 && n_grp == 8) { QSA_LAUNCH(3, 8); }
        else if (hpw == 3 && n_grp == 4) { QSA_LAUNCH(3, 4); }
        else if (hpw == 3 && n_grp == 2) { QSA_LAUNCH(3, 2); }
        else if (hpw == 3 && n_grp == 1) { QSA_LAUNCH(3, 1); }
        else if (hpw == 4 && n_grp == 1) { QSA_LAUNCH(4, 1); }
        else if (hpw == 4 && n_grp == 2) { QSA_LAUNCH(4, 2); }
        else if (hpw == 4 && n_grp == 4) { QSA_LAUNCH(4, 4); }
        else if (hpw == 2 && n_grp == 4) { QSA_LAUNCH(2, 4); }
        else if (hpw == 2 && n_grp == 1) { QSA_LAUNCH(2, 1); }
        else if (hpw == 1 && n_grp == 1) { QSA_LAUNCH(1, 1); }
        else { GGML_ABORT("qsa attention: unsupported head grouping %d x %d", hpw, n_grp); }
#undef QSA_LAUNCH
#undef QSA_ARGS
        CUDA_CHECK(cudaGetLastError());

        if (n_split > 1) {
            qsa_attn_combine<<<dim3(n_head, nt), QSA_ATTN_D, 0, stream>>>(part_acc.ptr, part_ml.ptr, dst_c, n_split);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}
