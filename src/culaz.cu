// CuLAZ: GPU-accelerated LAZ decoding.
//
// Copyright 2026 Elias Kristmann, Markus Schütz
// Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
//
// The arithmetic decoder, the symbol and bit models, the integer compressor
// and the point format 2 decompression in this file are a CUDA port of
// LASzip 3.5.0 (https://github.com/LASzip/LASzip):
//
//   (c) 2007-2022, rapidlasso GmbH - fast tools to catch reality
//   Licensed under the Apache License, Version 2.0
//
// Changes: rewritten as CUDA device code that decodes one chunk per warp,
// with warp-cooperative symbol search and model updates
//
// LASzip's arithmetic coder is an adapted version of Amir Said's FastAC:
//
//   Copyright (c) 2019 by Amir Said (said@ieee.org) &
//                         William A. Pearlman (pearlw@ecse.rpi.edu)
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are
//   met:
//
//   1. Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
//   2. Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
//   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>
#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>
#include "culaz.cuh"

namespace cg = cooperative_groups;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define AC__MinLength 0x01000000U

// ---------------------------------------------------------------
// 1. Arithmetic Decoder
// ---------------------------------------------------------------

struct ArithmeticDecoder {
    uint32_t length;
    uint32_t value;
    const uint8_t* buffer;
    uint32_t pos;

    __device__ __forceinline__ void init(const uint8_t* data, uint32_t offset) {
        buffer = data;
        pos = offset;
        length = 0xFFFFFFFFU;
        value = (uint32_t)buffer[pos] << 24 | (uint32_t)buffer[pos + 1] << 16 |
            (uint32_t)buffer[pos + 2] << 8 | (uint32_t)buffer[pos + 3];
        pos += 4;
    }

    __device__ __forceinline__ void renorm() {
        while (length < AC__MinLength) {
            value = (value << 8) | buffer[pos++];
            length <<= 8;
        }
    }
	
    __device__ __forceinline__ uint32_t readBits_2(int bit_count) {
        uint32_t ltmp = length >> bit_count;
        uint32_t raw = value / ltmp;
        length = ltmp;
        value -= length * raw;
        renorm();
        return raw;
    }

    // Split into readBits and readBits_2 because recursion prevented the compiler from inlining.
    __device__ __forceinline__ uint32_t readBits(int bit_count) {
        if (bit_count > 19) {
            uint32_t lower = readBits_2(16);
            return lower | (readBits_2(bit_count - 16) << 16);
        }
        return readBits_2(bit_count);
    }
};

// ---------------------------------------------------------------
// 2. Symbol Model
// ---------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#define LASZIP_STATIC_MODEL_SIZE 1
#else
#define LASZIP_STATIC_MODEL_SIZE 0
#endif

template<uint32_t CAPACITY = 256, bool STATIC_SIZE = false>
struct SymbolModel {
    static_assert(CAPACITY >= 2 && CAPACITY <= 256);
    static constexpr bool use_static_size =
        STATIC_SIZE && LASZIP_STATIC_MODEL_SIZE;

    uint16_t symbol_count[CAPACITY];
    uint16_t distribution[CAPACITY];
    uint32_t update_cycle;
    uint32_t symbols_until_update;
    uint32_t num_symbols;

    __device__ void init(uint32_t symbols) {
        const uint32_t symbol_limit = use_static_size ? CAPACITY : symbols;
        if constexpr (!use_static_size) {
            num_symbols = symbols;
        }

        const int lane = threadIdx.x & 31;
        for (uint32_t i = lane; i < symbol_limit; i+=32) symbol_count[i] = 1;
        __syncwarp();
        update_cycle = symbol_limit;
        update_distribution();

        update_cycle = (symbol_limit + 6) / 2;
        symbols_until_update = update_cycle;
    }

    __device__ void update_distribution() {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);

        const unsigned mask = 0xffffffffu;
        const int lane = threadIdx.x & 31;
        const uint32_t symbol_limit =
            use_static_size ? CAPACITY : num_symbols;

        uint32_t local_sum = 0;
        for (uint32_t i = lane; i < symbol_limit; i += 32) {
            local_sum += symbol_count[i];
        }
        uint32_t total_count = cg::reduce(warp, local_sum, cg::plus<uint32_t>());
        total_count = __shfl_sync(mask, total_count, 0);

        if (total_count > 32768u) {
            uint32_t local_sum2 = 0;
            for (uint32_t i = lane; i < symbol_limit; i += 32) {
                uint32_t v = symbol_count[i];
                v = (v + 1u) >> 1;
                symbol_count[i] = v;
                local_sum2 += v;
            }
            total_count = cg::reduce(warp, local_sum2, cg::plus<uint32_t>());
        }

        uint32_t scale = 0x80000000u / total_count;
        scale = __shfl_sync(mask, scale, 0);

        // Prefix sum in tiles of 32 with a running offset.
        uint32_t offset = 0;

        for (uint32_t base = 0; base < symbol_limit; base += 32) {
            uint32_t idx = base + lane;
            uint32_t val = (idx < symbol_limit) ? symbol_count[idx] : 0u;
            uint32_t excl = cg::exclusive_scan(warp, val, cg::plus<uint32_t>());

            uint32_t prefix = offset + excl;

            if (idx < symbol_limit) {
                uint32_t mapped = (uint32_t)(((uint64_t)scale * (uint64_t)prefix) >> 16);
                distribution[idx] = (uint16_t)mapped;
            }

            uint32_t tile_sum = __shfl_sync(mask, excl + val, 31);

            offset += tile_sum;
        }

        if (lane == 0) {
            update_cycle += (update_cycle >> 2);
            uint32_t cap = 8u * (symbol_limit + 6u);
            if (update_cycle > cap) update_cycle = cap;
            symbols_until_update = update_cycle;
        }
    }

    __device__ __forceinline__ uint32_t decode(ArithmeticDecoder& dec) {
        const uint32_t symbol_limit =
            use_static_size ? CAPACITY : num_symbols;

        // Load the update counter before the symbol search
        const uint32_t symbols_until_update_next = symbols_until_update - 1u;

        uint32_t ltmp = dec.length >> 15;

        // Warp-cooperative symbol search
        constexpr unsigned warp_mask = 0xffffffffu;
        const int lane = threadIdx.x & 31;
        int lo = 0;
        int len = symbol_limit;
        int sym = 0;
        if (symbol_limit <= 33) {
            const int idx = lane + 1;
            bool pred = false;
            if (idx < symbol_limit) {
                const uint32_t lower =
                    static_cast<uint32_t>(distribution[idx]) * ltmp;
                pred = lower <= dec.value;
            }
            sym = __popc(__ballot_sync(warp_mask, pred));
        }
        else {
            while (len > 0) {
                int step = (len + 31) / 32;
                int idx = lo + lane * step;

                bool pred = false;
                if (idx < lo + len) {
                    uint32_t lower_mid =
                        static_cast<uint32_t>(distribution[idx]) * ltmp;
                    pred = (lower_mid <= dec.value);
                }

                unsigned mask = __ballot_sync(warp_mask, pred);
                int count = __popc(mask);

                if (count > 0) {
                    lo += (count - 1) * step;
                    len = step;
                    if (step == 1) { sym = lo; break; }
                }
                else {
                    len = step - 1;
                    if (len <= 0) { sym = lo; break; }
                }
            }
        }

        uint32_t lower = (uint32_t)distribution[sym] * ltmp;
        dec.value -= lower;
        if ((uint32_t)sym < symbol_limit - 1) {
            dec.length = ((uint32_t)distribution[sym + 1] * ltmp) - lower;
        }
        else {
            dec.length -= lower;
        }
        dec.renorm();

        symbol_count[sym] += 1;
        if (symbols_until_update_next == 0) {
            update_distribution();
        }
        else {
            symbols_until_update = symbols_until_update_next;
        }

        return sym;
    }
};

struct BitModel {
    uint32_t bit_0_count;
    uint32_t bit_count;
    uint32_t bit_0_prob;
    uint32_t update_cycle;
    uint32_t bits_until_update;

    __device__ void init() {
        bit_0_count = 1;
        bit_count = 2;
        bit_0_prob = 4096;
        update_cycle = 4;
        bits_until_update = 4;
    }

    __device__ void update() {
        bit_count += update_cycle;
        if (bit_count > 8192) {
            bit_count = (bit_count + 1) >> 1;
            bit_0_count = (bit_0_count + 1) >> 1;
            if (bit_0_count == bit_count) bit_count++;
        }

        uint32_t scale = 0x80000000U / bit_count;
        bit_0_prob = (bit_0_count * scale) >> 18;

        update_cycle += (update_cycle >> 2);
        if (update_cycle > 64) update_cycle = 64;
        bits_until_update = update_cycle;
    }

    __device__ __forceinline__ uint32_t decode(ArithmeticDecoder& dec) {
        uint32_t ltmp = dec.length >> 13;
        uint32_t lower = bit_0_prob * ltmp;
        uint32_t bit;

        if (dec.value >= lower) {
            bit = 1;
            dec.value -= lower;
            dec.length -= lower;
        }
        else {
            bit = 0;
            dec.length = lower;
        }
        dec.renorm();

        if (bit == 0) bit_0_count++;
        if (--bits_until_update == 0) update();

        return bit;
    }
};

// ---------------------------------------------------------------
// 3. Integer Compressor
// ---------------------------------------------------------------
template<int BITS, int CONTEXTS, int BITS_HIGH>
struct IntegerCompressor {
    SymbolModel<BITS + 1, LASZIP_STATIC_MODEL_SIZE != 0> model_k[CONTEXTS];

    BitModel model_bit_0;
    SymbolModel<> model_corrector[BITS + 1];

    __device__ void init() {
        for (int i = 0; i < CONTEXTS; i++) {
            model_k[i].init(BITS + 1);
        }
        model_bit_0.init();

        for (int i = 1; i <= BITS_HIGH; i++) {
            model_corrector[i].init(1 << i);
        }
        for (int i = BITS_HIGH + 1; i <= BITS; i++) {
            model_corrector[i].init(1 << BITS_HIGH);
        }
    }

    __device__ __forceinline__ int32_t decode(ArithmeticDecoder& dec, int context, int& out_k) {
        int k = model_k[context].decode(dec);
        out_k = k;

        if (k == 0) return model_bit_0.decode(dec);
        if (k == 32) return INT32_MIN;

        int32_t c;
        if (k <= BITS_HIGH) {
            c = model_corrector[k].decode(dec);
        }
        else {
            int k1 = k - BITS_HIGH;
            c = model_corrector[k].decode(dec);
            int c1 = dec.readBits(k1);
            c = (c << k1) | c1;
        }
        if ((uint32_t)c >= (1u << (k - 1))) c = (int32_t)((uint32_t)c + 1u);
        else c = (int32_t)((uint32_t)c - ((1u << k) - 1u));

        return c;
    }

    __device__ __forceinline__ int32_t decode(ArithmeticDecoder& dec, int context) {
        int dummy_k;
        return decode(dec, context, dummy_k);
    }
};

// ---------------------------------------------------------------
// 4. Chunk State
// ---------------------------------------------------------------
struct alignas(16) ChunkState {
    SymbolModel<64, LASZIP_STATIC_MODEL_SIZE != 0> model_changed_values;
    SymbolModel<256, LASZIP_STATIC_MODEL_SIZE != 0> model_bit_byte[256];
    SymbolModel<256, LASZIP_STATIC_MODEL_SIZE != 0> model_classification[32];
    SymbolModel<256, LASZIP_STATIC_MODEL_SIZE != 0> model_scan_angle[2];
    SymbolModel<256, LASZIP_STATIC_MODEL_SIZE != 0> model_user_data[256];

    IntegerCompressor<16, 4, 8> model_intensity;
    IntegerCompressor<16, 1, 8> model_point_source;
    IntegerCompressor<32, 2, 8> model_X;
    IntegerCompressor<32, 22, 8> model_Y;
    IntegerCompressor<32, 20, 8> model_Z;

    SymbolModel<128, LASZIP_STATIC_MODEL_SIZE != 0> model_rgb_changed;
    SymbolModel<256, LASZIP_STATIC_MODEL_SIZE != 0> rgb_models[6];

    __device__ void init() {
        model_changed_values.init(64);
        model_rgb_changed.init(128);
        for (int i = 0; i < 256; i++) {
            model_bit_byte[i].init(256);
            model_user_data[i].init(256);
        }
        for (int i = 0; i < 32; i++) {
            model_classification[i].init(256);
        }
        for (int i = 0; i < 2; i++) {
            model_scan_angle[i].init(256);
        }
        for (int i = 0; i < 6; i++) {
            rgb_models[i].init(256);
        }
        model_point_source.init();
        model_intensity.init();
        model_X.init();
        model_Y.init();
        model_Z.init();
    }
};

__device__ __noinline__ void reset_chunk_state(
    ChunkState* destination_state,
    const ChunkState* initial_state)
{
    static_assert(sizeof(ChunkState) % sizeof(uint4) == 0);
    const int lane = threadIdx.x & 31;
    auto* destination = reinterpret_cast<uint4*>(destination_state);
    const auto* source = reinterpret_cast<const uint4*>(initial_state);
    constexpr int state_words = sizeof(ChunkState) / sizeof(uint4);
    for (int word = lane; word < state_words; word += 32) {
        destination[word] = source[word];
    }
    __syncwarp();
}

// Branchless replacement for LASzip's StreamingMedian5
struct StreamingMedian5 {
    int32_t v0, v1, v2, v3, v4;
    int32_t remove_largest;

    __device__ __forceinline__ void init() {
        v0 = v1 = v2 = v3 = v4 = 0;
        remove_largest = 1;
    }

    __device__ __forceinline__ void add(int32_t v) {
        int32_t current_median = v2;

        int32_t next_remove = (v < current_median) ? 1 :
            (v > current_median) ? 0 : (1 - remove_largest);

        int32_t new_v0 = remove_largest ? v0 : v;
        int32_t new_v4 = remove_largest ? v : v4;
        v0 = new_v0;
        v4 = new_v4;

        remove_largest = next_remove;

        sort_5();
    }

    __device__ __forceinline__ int32_t get() const {
        return v2;
    }

private:
    __device__ __forceinline__ void cswap(int32_t& a, int32_t& b) {
        int32_t tmp = a;
        a = min(tmp, b);
        b = max(tmp, b);
    }

    __device__ __forceinline__ void sort_5() {
        cswap(v0, v1);
        cswap(v3, v4);
        cswap(v2, v4);
        cswap(v2, v3);
        cswap(v0, v3);
        cswap(v0, v2);
        cswap(v1, v4);
        cswap(v1, v3);
        cswap(v1, v2);
    }
};

// ---------------------------------------------------------------
// 5. Kernels
// ---------------------------------------------------------------
__device__ inline int32_t read_i32(const uint8_t* ptr)
{
    return (int32_t)(
        (uint32_t)ptr[0] |
        ((uint32_t)ptr[1] << 8) |
        ((uint32_t)ptr[2] << 16) |
        ((uint32_t)ptr[3] << 24));
}

__device__ inline uint16_t read_u16(const uint8_t* ptr)
{
    return (uint16_t)(ptr[0] | (ptr[1] << 8));
}
__device__ inline int32_t ISum32(int32_t pred, int32_t diff) {
    return (int32_t)((uint32_t)pred + (uint32_t)diff);
}

__constant__ uint8_t number_return_map[8][8] = {
    {15, 14, 13, 12, 11, 10,  9,  8},
    {14,  0,  1,  3,  6, 10, 10,  9},
    {13,  1,  2,  4,  7, 11, 11, 10},
    {12,  3,  4,  5,  8, 12, 12, 11},
    {11,  6,  7,  8,  9, 13, 13, 12},
    {10, 10, 11, 12, 13, 14, 14, 13},
    { 9, 10, 11, 12, 13, 14, 15, 14},
    { 8,  9, 10, 11, 12, 13, 14, 15}
};

__constant__ uint8_t number_return_level[8][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {1, 0, 1, 2, 3, 4, 5, 6},
    {2, 1, 0, 1, 2, 3, 4, 5},
    {3, 2, 1, 0, 1, 2, 3, 4},
    {4, 3, 2, 1, 0, 1, 2, 3},
    {5, 4, 3, 2, 1, 0, 1, 2},
    {6, 5, 4, 3, 2, 1, 0, 1},
    {7, 6, 5, 4, 3, 2, 1, 0}
};

__device__ inline int clamp255(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
}
#define U8_FOLD(v) ((uint8_t)((v) & 0xFF))

__global__ void laszip_init_state_kernel(ChunkState* states, int total_chunks) {
    int chunk_id = blockIdx.x;
    if (chunk_id >= total_chunks - 1) return;
    states[chunk_id].init();
}

__global__ void laszip_format2_kernel(
    const uint8_t* __restrict__ compressed,
    const uint64_t* __restrict__ chunk_offsets,
    const uint64_t* __restrict__ point_offsets,
    const uint32_t* __restrict__ point_counts,
    PointFormat2* __restrict__ out_points,
    int total_chunks,
    ChunkState* states,
    const ChunkState* __restrict__ initial_state)
{
    constexpr int warps_per_block = 2;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int worker_id = blockIdx.x * warps_per_block + warp_id;
    const int worker_count = gridDim.x * warps_per_block;

    __shared__ StreamingMedian5 sm_median_X[warps_per_block][16];
    __shared__ StreamingMedian5 sm_median_Y[warps_per_block][16];
    __shared__ int32_t          sm_last_Z[warps_per_block][8];
    __shared__ uint16_t         sm_last_Intensity[warps_per_block][16];

    ChunkState& state = states[worker_id];
    ArithmeticDecoder dec;

    for (int chunk_id = worker_id; chunk_id < total_chunks; chunk_id += worker_count) {

        // Every chunk begins with the same uniform adaptive-model state.
        // Copying a prebuilt image avoids recomputing hundreds of identical reductions, scans, and integer divisions for every worker.
        reset_chunk_state(&state, initial_state);

        const uint64_t base_idx = point_offsets[chunk_id];
        const uint32_t chunk_point_count = point_counts[chunk_id];
        uint64_t offset = chunk_offsets[chunk_id];

        // Use a local variable to prevent 32 threads from writing to global memory simultaneously
        PointFormat2 first_local;

        // All threads redundantly read the data
        first_local.X = read_i32(compressed + offset); offset += 4;
        first_local.Y = read_i32(compressed + offset); offset += 4;
        first_local.Z = read_i32(compressed + offset); offset += 4;

        first_local.Intensity = read_u16(compressed + offset); offset += 2;
        first_local.ReturnMask = compressed[offset++];
        first_local.Classification = compressed[offset++] & 0b11111;
        first_local.ScanAngleRank = (int8_t)compressed[offset++];
        first_local.UserData = compressed[offset++];

        first_local.PointSourceID = read_u16(compressed + offset); offset += 2;
        first_local.Red = read_u16(compressed + offset); offset += 2;
        first_local.Green = read_u16(compressed + offset); offset += 2;
        first_local.Blue = read_u16(compressed + offset); offset += 2;

        // Only thread 0 writes the first point to global memory
        if (lane == 0) {
            out_points[base_idx] = first_local;
        }

        dec.init(compressed, offset);

        PointFormat2 prev = first_local;

        if (lane < 16) {
            sm_median_X[warp_id][lane].init();
            sm_median_Y[warp_id][lane].init();
            sm_last_Intensity[warp_id][lane] = 0;
        }
        if (lane < 8) sm_last_Z[warp_id][lane] = 0;
        __syncwarp();

        for (uint32_t i = 1; i < chunk_point_count; i++) {
            PointFormat2 p = prev;
            uint32_t changed = state.model_changed_values.decode(dec);

            if (changed & (1 << 5)) {
                p.ReturnMask = state.model_bit_byte[prev.ReturnMask].decode(dec);
            }
            int r = p.ReturnMask & 7;
            int n = (p.ReturnMask >> 3) & 7;
            int m = number_return_map[n][r];
            int I = number_return_level[n][r];

            if (changed & (1 << 4)) {
                int ctx_int = (m < 3) ? m : 3;
                int32_t dInt = state.model_intensity.decode(dec, ctx_int);
                p.Intensity = (uint16_t)(sm_last_Intensity[warp_id][m] + dInt);
            }
            else {
                p.Intensity = sm_last_Intensity[warp_id][m];
            }
            sm_last_Intensity[warp_id][m] = p.Intensity;

            if (changed & (1 << 3)) p.Classification = state.model_classification[prev.Classification].decode(dec) & 0b11111;
            if (changed & (1 << 2)) p.ScanAngleRank = U8_FOLD(p.ScanAngleRank + state.model_scan_angle[(p.ReturnMask >> 6) & 1].decode(dec));
            if (changed & (1 << 1)) p.UserData = state.model_user_data[prev.UserData].decode(dec);
            if (changed & (1 << 0)) p.PointSourceID = prev.PointSourceID + state.model_point_source.decode(dec, 0);

            int k_x, k_y, k_z;

            int ctx_x = (n == 1) ? 1 : 0;
            int32_t dx = state.model_X.decode(dec, ctx_x, k_x);
            int32_t dxMedian = ISum32(sm_median_X[warp_id][m].get(), dx);
            p.X = ISum32(prev.X, dxMedian);
            sm_median_X[warp_id][m].add(dxMedian);

            int ctx_y = (k_x < 20) ? (k_x & ~1) : 20;
            if (n == 1) ctx_y += 1;
            int32_t dy = state.model_Y.decode(dec, ctx_y, k_y);
            int32_t dyMedian = ISum32(sm_median_Y[warp_id][m].get(), dy);
            p.Y = ISum32(prev.Y, dyMedian);
            sm_median_Y[warp_id][m].add(dyMedian);

            int kXY = (k_x + k_y) / 2;
            int ctx_z = (kXY < 18) ? (kXY & ~1) : 18;
            if (n == 1) ctx_z += 1;
            int32_t dz = state.model_Z.decode(dec, ctx_z, k_z);
            p.Z = ISum32(sm_last_Z[warp_id][I], dz);
            sm_last_Z[warp_id][I] = p.Z;

            uint32_t sym = state.model_rgb_changed.decode(dec);

            uint16_t last_r = prev.Red, last_g = prev.Green, last_b = prev.Blue;
            uint8_t r_low, r_high;

            if (sym & (1 << 0)) r_low = U8_FOLD((uint8_t)state.rgb_models[0].decode(dec) + (last_r & 0xFF));
            else r_low = last_r & 0xFF;

            if (sym & (1 << 1)) r_high = U8_FOLD((uint8_t)state.rgb_models[1].decode(dec) + (last_r >> 8));
            else r_high = (last_r >> 8) & 0xFF;

            uint16_t final_red = (uint16_t)r_low | ((uint16_t)r_high << 8);

            if (sym & (1 << 6)) {
                int32_t diff = (int32_t)r_low - (int32_t)(last_r & 0xFF);
                uint8_t g_low, g_high, b_low, b_high;

                if (sym & (1 << 2)) g_low = U8_FOLD((uint8_t)state.rgb_models[2].decode(dec) + clamp255(diff + (last_g & 0xFF)));
                else g_low = last_g & 0xFF;

                if (sym & (1 << 4)) b_low = U8_FOLD((uint8_t)state.rgb_models[4].decode(dec) + clamp255((diff + ((int32_t)g_low - (int32_t)(last_g & 0xFF))) / 2 + (last_b & 0xFF)));
                else b_low = last_b & 0xFF;

                diff = (int32_t)r_high - (int32_t)(last_r >> 8);

                if (sym & (1 << 3)) g_high = U8_FOLD((uint8_t)state.rgb_models[3].decode(dec) + clamp255(diff + (last_g >> 8)));
                else g_high = (last_g >> 8) & 0xFF;

                if (sym & (1 << 5)) b_high = U8_FOLD((uint8_t)state.rgb_models[5].decode(dec) + clamp255((diff + ((int32_t)g_high - (int32_t)(last_g >> 8))) / 2 + (last_b >> 8)));
                else b_high = (last_b >> 8) & 0xFF;

                p.Red = final_red;
                p.Green = (uint16_t)g_low | ((uint16_t)g_high << 8);
                p.Blue = (uint16_t)b_low | ((uint16_t)b_high << 8);
            }
            else {
                p.Red = p.Green = p.Blue = final_red;
            }

            if (lane == 0) {
                out_points[base_idx + i] = p;
            }
            prev = p;
        }
    }
}

// ---------------------------------------------------------------
// 6. Statistics
// ---------------------------------------------------------------
// Example of consuming the decoded points while they are still in device
// memory. One block reduces one chunk, then merges into the owning file's
// statistics with one set of global atomics per warp.
constexpr int stats_threads_per_block = 256;

__device__ __forceinline__ void atomic_add_u64(uint64_t* address, uint64_t value) {
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    atomicAdd(reinterpret_cast<unsigned long long*>(address),
        static_cast<unsigned long long>(value));
}

__global__ void point_stats_kernel(
    const PointFormat2* __restrict__ points,
    const uint64_t* __restrict__ point_offsets,
    const uint32_t* __restrict__ point_counts,
    const uint32_t* __restrict__ chunk_files,
    PointCloudStats* stats)
{
    __shared__ uint32_t sm_classification_counts[32];
    if (threadIdx.x < 32) sm_classification_counts[threadIdx.x] = 0;
    __syncthreads();

    const PointFormat2* chunk_points = points + point_offsets[blockIdx.x];
    const uint32_t count = point_counts[blockIdx.x];

    int32_t min_x = INT32_MAX, min_y = INT32_MAX, min_z = INT32_MAX;
    int32_t max_x = INT32_MIN, max_y = INT32_MIN, max_z = INT32_MIN;
    uint64_t intensity = 0, red = 0, green = 0, blue = 0;

    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const PointFormat2 p = chunk_points[i];
        min_x = min(min_x, p.X); max_x = max(max_x, p.X);
        min_y = min(min_y, p.Y); max_y = max(max_y, p.Y);
        min_z = min(min_z, p.Z); max_z = max(max_z, p.Z);
        intensity += p.Intensity;
        red += p.Red;
        green += p.Green;
        blue += p.Blue;

        const uint32_t classification = p.Classification & 31;
        const unsigned peers =
            __match_any_sync(__activemask(), classification);
        if ((threadIdx.x & 31) == __ffs(peers) - 1) {
            atomicAdd(&sm_classification_counts[classification],
                static_cast<uint32_t>(__popc(peers)));
        }
    }

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    min_x = cg::reduce(warp, min_x, cg::less<int32_t>());
    min_y = cg::reduce(warp, min_y, cg::less<int32_t>());
    min_z = cg::reduce(warp, min_z, cg::less<int32_t>());
    max_x = cg::reduce(warp, max_x, cg::greater<int32_t>());
    max_y = cg::reduce(warp, max_y, cg::greater<int32_t>());
    max_z = cg::reduce(warp, max_z, cg::greater<int32_t>());
    intensity = cg::reduce(warp, intensity, cg::plus<uint64_t>());
    red = cg::reduce(warp, red, cg::plus<uint64_t>());
    green = cg::reduce(warp, green, cg::plus<uint64_t>());
    blue = cg::reduce(warp, blue, cg::plus<uint64_t>());

    PointCloudStats& file_stats = stats[chunk_files[blockIdx.x]];
    if (warp.thread_rank() == 0) {
        atomicMin(&file_stats.min_X, min_x);
        atomicMin(&file_stats.min_Y, min_y);
        atomicMin(&file_stats.min_Z, min_z);
        atomicMax(&file_stats.max_X, max_x);
        atomicMax(&file_stats.max_Y, max_y);
        atomicMax(&file_stats.max_Z, max_z);
        atomic_add_u64(&file_stats.intensity_sum, intensity);
        atomic_add_u64(&file_stats.red_sum, red);
        atomic_add_u64(&file_stats.green_sum, green);
        atomic_add_u64(&file_stats.blue_sum, blue);
    }

    __syncthreads();
    if (threadIdx.x < 32 && sm_classification_counts[threadIdx.x] != 0) {
        atomic_add_u64(&file_stats.classification_counts[threadIdx.x],
            sm_classification_counts[threadIdx.x]);
    }
    if (threadIdx.x == 0) {
        atomic_add_u64(&file_stats.point_count, count);
    }
}

int decompress_files(
    const std::vector<CulazFileInput>& files,
    bool copy_results,
    int benchmark_iterations,
    uint32_t max_batch_chunks,
    GpuBenchmarkResult* benchmark_result) {
    constexpr int warps_per_block = 2;
    constexpr int threads_per_block = warps_per_block * 32;
    constexpr size_t mib = 1024ull * 1024ull;
    constexpr size_t compressed_read_ahead = 64;
    // Chunk offset, point offset, point count and file index.
    constexpr size_t descriptor_bytes_per_chunk =
        2 * sizeof(uint64_t) + 2 * sizeof(uint32_t);

    struct ChunkJob {
        uint32_t file_index;
        uint64_t file_point_offset;
        uint32_t point_count;
        size_t compressed_start;
        size_t compressed_size;
    };

    std::vector<ChunkJob> jobs;
    uint64_t total_points = 0;
    for (size_t file_index = 0; file_index < files.size(); ++file_index) {
        const CulazFileInput& file = files[file_index];
        if ((!file.raw_file_data && file.filename.empty()) ||
            !file.chunk_offsets || (copy_results && !file.points)) {
            std::cerr << "GPU input " << file_index
                << " is missing required data.\n";
            return 1;
        }
        if (file.num_chunks < 2 || file.total_points == 0 ||
            file.chunk_offsets->size() < file.num_chunks) {
            std::cerr << "GPU input " << file_index
                << " contains no chunks or has a short chunk table.\n";
            return 1;
        }
        if (file.chunk_size == 0) {
            std::cerr << "GPU input " << file_index
                << " has no fixed chunk size.\n";
            return 1;
        }
        const uint32_t points_per_chunk = file.chunk_size;

        const size_t raw_byte_size = file.raw_file_data
            ? file.raw_file_data->size()
            : file.raw_byte_size;
        const std::vector<uint64_t>& offsets = *file.chunk_offsets;
        for (uint32_t i = 0; i < file.num_chunks; ++i) {
            if (offsets[i] > raw_byte_size ||
                (i > 0 && offsets[i] < offsets[i - 1])) {
                std::cerr << "GPU input " << file_index
                    << " contains an invalid chunk offset.\n";
                return 1;
            }
        }

        const uint32_t data_chunk_count = file.num_chunks - 1;
        const uint64_t expected_chunk_count =
            (file.total_points + points_per_chunk - 1) / points_per_chunk;
        if (data_chunk_count != expected_chunk_count) {
            std::cerr << "GPU input " << file_index << " has "
                << data_chunk_count << " chunks, but "
                << file.total_points << " points in " << points_per_chunk
                << "-point chunks require " << expected_chunk_count << ".\n";
            return 1;
        }

        for (uint32_t chunk = 0; chunk < data_chunk_count; ++chunk) {
            const uint64_t file_point_offset =
                uint64_t(chunk) * points_per_chunk;
            const uint32_t point_count = static_cast<uint32_t>(
                std::min<uint64_t>(
                    points_per_chunk,
                    file.total_points - file_point_offset));
            const size_t compressed_start =
                static_cast<size_t>(offsets[chunk]);
            const size_t logical_end =
                static_cast<size_t>(offsets[chunk + 1]);
            const size_t compressed_end = std::min(
                raw_byte_size,
                logical_end + std::min(
                    compressed_read_ahead, raw_byte_size - logical_end));
            jobs.push_back({
                static_cast<uint32_t>(file_index),
                file_point_offset,
                point_count,
                compressed_start,
                compressed_end - compressed_start
            });
        }

        if (copy_results) {
            *file.points = std::vector<PointFormat2>(file.total_points);
        }
        total_points += file.total_points;
    }

    if (jobs.empty()) {
        std::cerr << "No chunks were supplied for GPU decompression.\n";
        return 1;
    }

    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceProp device_prop;
    gpuErrchk(cudaGetDeviceProperties(&device_prop, device));

    int blocks_per_sm = 0;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, laszip_format2_kernel, threads_per_block, 0));
    cudaFuncAttributes attributes;
    gpuErrchk(cudaFuncGetAttributes(&attributes, laszip_format2_kernel));

    printf("GPU: %s (%d SMs, compute capability %d.%d)\n",
        device_prop.name, device_prop.multiProcessorCount,
        device_prop.major, device_prop.minor);
    printf("Kernel: %d registers, %zu B shared memory, %d resident warps per SM\n",
        attributes.numRegs, attributes.sharedSizeBytes,
        blocks_per_sm * warps_per_block);

    const int max_resident_blocks =
        device_prop.multiProcessorCount * blocks_per_sm;
    const uint32_t resident_worker_count =
        static_cast<uint32_t>(max_resident_blocks * warps_per_block);
    const uint32_t batch_chunk_cap = max_batch_chunks == 0
        ? static_cast<uint32_t>(
            std::min<size_t>(jobs.size(), resident_worker_count))
        : static_cast<uint32_t>(
            std::min<size_t>(jobs.size(), max_batch_chunks));
    if (jobs.size() < resident_worker_count) {
        printf("Note: only %zu chunks for %u resident warps; each chunk is "
            "decoded by one warp, so most of the GPU stays idle.\n",
            jobs.size(), resident_worker_count);
    }

    size_t free_device_bytes = 0;
    size_t total_device_bytes = 0;
    gpuErrchk(cudaMemGetInfo(&free_device_bytes, &total_device_bytes));
    const uint64_t estimated_blocks =
        (uint64_t(batch_chunk_cap) + warps_per_block - 1) /
        warps_per_block;
    const int estimated_launch_blocks = static_cast<int>(
        std::min<uint64_t>(estimated_blocks, max_resident_blocks));
    const size_t estimated_state_bytes =
        size_t(estimated_launch_blocks * warps_per_block) *
        sizeof(ChunkState);
    const size_t reserve_bytes =
        std::max<size_t>(512ull * mib, free_device_bytes / 10);
    if (free_device_bytes <= estimated_state_bytes + reserve_bytes) {
        std::cerr << "Not enough free GPU memory for decoder state.\n";
        return 1;
    }
    const size_t batch_memory_budget =
        free_device_bytes - estimated_state_bytes - reserve_bytes;

    struct Batch {
        size_t first_job;
        uint32_t chunk_count;
        uint64_t point_count;
        size_t compressed_size;
        size_t device_bytes;
    };

    std::vector<Batch> batches;
    size_t first_job = 0;
    while (first_job < jobs.size()) {
        uint32_t chunk_count = 0;
        uint64_t point_count = 0;
        size_t compressed_size = 0;
        size_t device_bytes = 0;

        while (first_job + chunk_count < jobs.size() &&
               chunk_count < batch_chunk_cap) {
            const ChunkJob& job = jobs[first_job + chunk_count];
            const uint32_t prospective_chunks = chunk_count + 1;
            const uint64_t prospective_points =
                point_count + job.point_count;
            const size_t prospective_compressed =
                compressed_size + job.compressed_size;
            const uint64_t required_bytes =
                uint64_t(prospective_compressed) +
                prospective_points * sizeof(PointFormat2) +
                uint64_t(prospective_chunks) * descriptor_bytes_per_chunk;
            if (required_bytes > batch_memory_budget) {
                break;
            }
            chunk_count = prospective_chunks;
            point_count = prospective_points;
            compressed_size = prospective_compressed;
            device_bytes = static_cast<size_t>(required_bytes);
        }

        if (chunk_count == 0) {
            std::cerr << "One LAZ chunk does not fit in GPU memory.\n";
            return 1;
        }
        batches.push_back({
            first_job, chunk_count, point_count, compressed_size, device_bytes
        });
        first_job += chunk_count;
    }

    uint32_t largest_batch_chunks = 0;
    size_t largest_batch_bytes = 0;
    for (const Batch& batch : batches) {
        largest_batch_chunks =
            std::max(largest_batch_chunks, batch.chunk_count);
        largest_batch_bytes =
            std::max(largest_batch_bytes, batch.device_bytes);
    }
    const uint64_t state_blocks =
        (uint64_t(largest_batch_chunks) + warps_per_block - 1) /
        warps_per_block;
    const int launch_state_blocks = static_cast<int>(
        std::min<uint64_t>(state_blocks, max_resident_blocks));
    const int state_count = launch_state_blocks * warps_per_block;

    std::cout << "Combined GPU queue: " << files.size() << " files / "
        << jobs.size() << " chunks\n"
        << "GPU batches: " << batches.size()
        << ", largest: " << largest_batch_chunks << " chunks / "
        << (largest_batch_bytes / double(mib)) << " MiB"
        << ", memory budget: " << (batch_memory_budget / double(mib))
        << " MiB\n";

    std::vector<std::ifstream> streams(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        if (!files[i].raw_file_data) {
            streams[i].open(files[i].filename, std::ios::binary);
            if (!streams[i].is_open()) {
                std::cerr << "Failed to open '" << files[i].filename
                    << "' for streaming.\n";
                return 1;
            }
        }
    }

    ChunkState* d_states = nullptr;
    ChunkState* d_initial_state = nullptr;
    gpuErrchk(cudaMalloc(
        &d_states, size_t(state_count) * sizeof(ChunkState)));
    gpuErrchk(cudaMalloc(&d_initial_state, sizeof(ChunkState)));
    gpuErrchk(cudaMemset(d_initial_state, 0, sizeof(ChunkState)));
    laszip_init_state_kernel<<<1, 32>>>(d_initial_state, 2);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const bool compute_stats = std::any_of(files.begin(), files.end(),
        [](const CulazFileInput& file) { return file.stats != nullptr; });
    PointCloudStats* d_stats = nullptr;
    if (compute_stats) {
        const std::vector<PointCloudStats> initial_stats(files.size());
        gpuErrchk(cudaMalloc(
            &d_stats, files.size() * sizeof(PointCloudStats)));
        gpuErrchk(cudaMemcpy(
            d_stats, initial_stats.data(),
            files.size() * sizeof(PointCloudStats),
            cudaMemcpyHostToDevice));
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    const int iteration_count = std::max(1, benchmark_iterations);
    std::vector<double> iteration_ms(iteration_count, 0.0);
    double stats_ms = 0.0;
    const dim3 block_dimensions(threads_per_block);

    auto cleanup_persistent = [&]() {
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
        gpuErrchk(cudaFree(d_initial_state));
        gpuErrchk(cudaFree(d_states));
        if (d_stats) gpuErrchk(cudaFree(d_stats));
    };

    auto make_batch = [&](size_t first, uint32_t count) {
        uint64_t points = 0;
        size_t compressed_bytes = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const ChunkJob& job = jobs[first + i];
            points += job.point_count;
            compressed_bytes += job.compressed_size;
        }
        const size_t descriptor_bytes =
            size_t(count) * descriptor_bytes_per_chunk;
        return Batch{
            first,
            count,
            points,
            compressed_bytes,
            compressed_bytes +
                size_t(points) * sizeof(PointFormat2) +
                descriptor_bytes
        };
    };

    for (size_t batch_index = 0; batch_index < batches.size();) {
        const Batch batch = batches[batch_index];
        if (batches.size() > 1) {
            std::cout << "Processing GPU batch " << (batch_index + 1)
                << " / " << batches.size() << "\n";
        }

        std::vector<uint8_t> compressed(batch.compressed_size);
        std::vector<uint64_t> chunk_offsets(batch.chunk_count);
        std::vector<uint64_t> point_offsets(batch.chunk_count);
        std::vector<uint32_t> point_counts(batch.chunk_count);
        std::vector<uint32_t> chunk_files(batch.chunk_count);
        size_t compressed_cursor = 0;
        uint64_t point_cursor = 0;

        for (uint32_t i = 0; i < batch.chunk_count; ++i) {
            const ChunkJob& job = jobs[batch.first_job + i];
            const CulazFileInput& file = files[job.file_index];
            chunk_offsets[i] = compressed_cursor;
            point_offsets[i] = point_cursor;
            point_counts[i] = job.point_count;
            chunk_files[i] = job.file_index;
            uint8_t* destination =
                compressed.data() + compressed_cursor;

            if (file.raw_file_data) {
                std::memcpy(
                    destination,
                    file.raw_file_data->data() + job.compressed_start,
                    job.compressed_size);
            }
            else {
                std::ifstream& stream = streams[job.file_index];
                stream.clear();
                stream.seekg(
                    static_cast<std::streamoff>(job.compressed_start));
                stream.read(
                    reinterpret_cast<char*>(destination),
                    static_cast<std::streamsize>(job.compressed_size));
                if (!stream) {
                    std::cerr << "Failed reading a chunk from '"
                        << file.filename << "'.\n";
                    cleanup_persistent();
                    return 1;
                }
            }
            compressed_cursor += job.compressed_size;
            point_cursor += job.point_count;
        }

        uint8_t* d_compressed = nullptr;
        uint64_t* d_chunk_offsets = nullptr;
        uint64_t* d_point_offsets = nullptr;
        uint32_t* d_point_counts = nullptr;
        uint32_t* d_chunk_files = nullptr;
        PointFormat2* d_points = nullptr;
        cudaError_t allocation_error =
            cudaMalloc(&d_compressed, batch.compressed_size);
        if (allocation_error == cudaSuccess) {
            allocation_error = cudaMalloc(
                &d_chunk_offsets,
                size_t(batch.chunk_count) * sizeof(uint64_t));
        }
        if (allocation_error == cudaSuccess) {
            allocation_error = cudaMalloc(
                &d_point_offsets,
                size_t(batch.chunk_count) * sizeof(uint64_t));
        }
        if (allocation_error == cudaSuccess) {
            allocation_error = cudaMalloc(
                &d_point_counts,
                size_t(batch.chunk_count) * sizeof(uint32_t));
        }
        if (allocation_error == cudaSuccess) {
            allocation_error = cudaMalloc(
                &d_chunk_files,
                size_t(batch.chunk_count) * sizeof(uint32_t));
        }
        if (allocation_error == cudaSuccess) {
            allocation_error = cudaMalloc(
                &d_points,
                size_t(batch.point_count) * sizeof(PointFormat2));
        }

        if (allocation_error != cudaSuccess) {
            if (d_compressed) cudaFree(d_compressed);
            if (d_chunk_offsets) cudaFree(d_chunk_offsets);
            if (d_point_offsets) cudaFree(d_point_offsets);
            if (d_point_counts) cudaFree(d_point_counts);
            if (d_chunk_files) cudaFree(d_chunk_files);
            if (d_points) cudaFree(d_points);
            cudaGetLastError();

            if (batch.chunk_count == 1) {
                std::cerr << "A single chunk allocation failed: "
                    << cudaGetErrorString(allocation_error) << ".\n";
                cleanup_persistent();
                return 1;
            }

            const uint32_t left_count = batch.chunk_count / 2;
            const uint32_t right_count =
                batch.chunk_count - left_count;
            const Batch left = make_batch(batch.first_job, left_count);
            const Batch right = make_batch(
                batch.first_job + left_count, right_count);
            batches[batch_index] = left;
            batches.insert(batches.begin() + batch_index + 1, right);
            std::cout << "CUDA could not allocate the "
                << (batch.device_bytes / double(mib))
                << " MiB batch (" << cudaGetErrorString(allocation_error)
                << "); split it into " << left_count << " + "
                << right_count << " chunks and retrying.\n";
            continue;
        }

        gpuErrchk(cudaMemcpy(
            d_compressed, compressed.data(), batch.compressed_size,
            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(
            d_chunk_offsets, chunk_offsets.data(),
            size_t(batch.chunk_count) * sizeof(uint64_t),
            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(
            d_point_offsets, point_offsets.data(),
            size_t(batch.chunk_count) * sizeof(uint64_t),
            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(
            d_point_counts, point_counts.data(),
            size_t(batch.chunk_count) * sizeof(uint32_t),
            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(
            d_chunk_files, chunk_files.data(),
            size_t(batch.chunk_count) * sizeof(uint32_t),
            cudaMemcpyHostToDevice));

        const int required_blocks =
            (int(batch.chunk_count) + warps_per_block - 1) /
            warps_per_block;
        const int launch_blocks =
            std::min(required_blocks, max_resident_blocks);
        const dim3 grid_dimensions(launch_blocks);

        laszip_format2_kernel<<<grid_dimensions, block_dimensions>>>(
            d_compressed, d_chunk_offsets, d_point_offsets, d_point_counts,
            d_points, int(batch.chunk_count), d_states, d_initial_state);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        for (int iteration = 0;
             iteration < iteration_count;
             ++iteration) {
            gpuErrchk(cudaEventRecord(start));
            laszip_format2_kernel<<<grid_dimensions, block_dimensions>>>(
                d_compressed, d_chunk_offsets, d_point_offsets,
                d_point_counts, d_points, int(batch.chunk_count),
                d_states, d_initial_state);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaEventRecord(stop));
            gpuErrchk(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
            iteration_ms[iteration] += elapsed_ms;
        }

        if (compute_stats) {
            gpuErrchk(cudaEventRecord(start));
            point_stats_kernel<<<batch.chunk_count, stats_threads_per_block>>>(
                d_points, d_point_offsets, d_point_counts, d_chunk_files,
                d_stats);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaEventRecord(stop));
            gpuErrchk(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
            stats_ms += elapsed_ms;
        }

        if (copy_results) {
            uint32_t first_copy_job = 0;
            while (first_copy_job < batch.chunk_count) {
                const ChunkJob& first =
                    jobs[batch.first_job + first_copy_job];
                uint32_t copy_jobs = 1;
                uint64_t copy_points = first.point_count;
                while (first_copy_job + copy_jobs < batch.chunk_count) {
                    const ChunkJob& previous =
                        jobs[batch.first_job + first_copy_job +
                            copy_jobs - 1];
                    const ChunkJob& next =
                        jobs[batch.first_job + first_copy_job + copy_jobs];
                    if (next.file_index != first.file_index ||
                        next.file_point_offset !=
                            previous.file_point_offset +
                                previous.point_count) {
                        break;
                    }
                    copy_points += next.point_count;
                    ++copy_jobs;
                }
                const CulazFileInput& destination =
                    files[first.file_index];
                gpuErrchk(cudaMemcpy(
                    destination.points->data() +
                        size_t(first.file_point_offset),
                    d_points + point_offsets[first_copy_job],
                    size_t(copy_points) * sizeof(PointFormat2),
                    cudaMemcpyDeviceToHost));
                first_copy_job += copy_jobs;
            }
        }

        gpuErrchk(cudaFree(d_compressed));
        gpuErrchk(cudaFree(d_chunk_offsets));
        gpuErrchk(cudaFree(d_point_offsets));
        gpuErrchk(cudaFree(d_point_counts));
        gpuErrchk(cudaFree(d_chunk_files));
        gpuErrchk(cudaFree(d_points));
        ++batch_index;
    }

    if (compute_stats) {
        std::vector<PointCloudStats> stats(files.size());
        gpuErrchk(cudaMemcpy(
            stats.data(), d_stats, files.size() * sizeof(PointCloudStats),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < files.size(); ++i) {
            if (files[i].stats) {
                *files[i].stats = stats[i];
            }
        }
    }

    double sum_ms = 0.0;
    double sum_squared_ms = 0.0;
    for (const double ms : iteration_ms) {
        sum_ms += ms;
        sum_squared_ms += ms * ms;
    }
    const double average_ms = sum_ms / double(iteration_count);
    const double variance_ms =
        (sum_squared_ms / double(iteration_count)) -
        (average_ms * average_ms);
    const double stddev_ms =
        variance_ms > 0.0 ? std::sqrt(variance_ms) : 0.0;
    const double points_per_ms =
        average_ms > 0.0 ? double(total_points) / average_ms : 0.0;

    if (benchmark_result) {
        benchmark_result->points = total_points;
        benchmark_result->iteration_ms = iteration_ms;
        benchmark_result->stats_ms = stats_ms;
    }
    std::cout << "Decoded " << total_points << " points\n"
        << "Kernel time (avg over " << iteration_count
        << " runs): " << average_ms << " ms"
        << " (stddev: " << stddev_ms << " ms), throughput: "
        << points_per_ms << " points/ms\n";
    if (compute_stats) {
        std::cout << "Statistics kernel time: " << stats_ms << " ms\n";
    }

    cleanup_persistent();
    return 0;
}

int decompress(const std::vector<uint8_t>& raw_file_data,
    uint32_t num_chunks,
    uint32_t chunk_size,
    const std::vector<uint64_t>& chunk_offsets,
    uint64_t actual_total_points,
    std::vector<PointFormat2>& points,
    bool copy_results,
    int benchmark_iterations,
    uint32_t max_batch_chunks,
    GpuBenchmarkResult* benchmark_result) {
    const CulazFileInput file = {
        {},
        &raw_file_data,
        raw_file_data.size(),
        &chunk_offsets,
        num_chunks,
        chunk_size,
        actual_total_points,
        &points
    };
    return decompress_files(
        {file},
        copy_results,
        benchmark_iterations,
        max_batch_chunks,
        benchmark_result);
}
