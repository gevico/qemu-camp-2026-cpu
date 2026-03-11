/*
 * RISC-V Xg233ai Extension Helpers
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "cpu.h"
#include "internals.h"
#include "accel/tcg/cpu-ldst.h"
#include "exec/helper-proto.h"

static int g233_mmu_idx(CPURISCVState *env)
{
    return riscv_env_mmu_index(env, false);
}

static uint32_t g233_ld32(CPURISCVState *env, target_ulong addr, uintptr_t ra)
{
    MemOpIdx oi = make_memop_idx(MO_LEUL, g233_mmu_idx(env));
    return cpu_ldl_mmu(env, addr, oi, ra);
}

static void g233_st32(CPURISCVState *env, target_ulong addr,
                      uint32_t val, uintptr_t ra)
{
    MemOpIdx oi = make_memop_idx(MO_LEUL, g233_mmu_idx(env));
    cpu_stl_mmu(env, addr, val, oi, ra);
}

static uint8_t g233_ld8(CPURISCVState *env, target_ulong addr, uintptr_t ra)
{
    MemOpIdx oi = make_memop_idx(MO_UB, g233_mmu_idx(env));
    return cpu_ldb_mmu(env, addr, oi, ra);
}

static void g233_st8(CPURISCVState *env, target_ulong addr,
                     uint8_t val, uintptr_t ra)
{
    MemOpIdx oi = make_memop_idx(MO_UB, g233_mmu_idx(env));
    cpu_stb_mmu(env, addr, val, oi, ra);
}

/* dma: FP32 matrix transpose */
void HELPER(dma)(CPURISCVState *env, target_ulong dst,
                 target_ulong src, target_ulong grain)
{
    uintptr_t ra = GETPC();
    static const int sizes[] = {8, 16, 32};
    int n = sizes[grain];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            uint32_t val = g233_ld32(env, src + (i * n + j) * 4, ra);
            g233_st32(env, dst + (j * n + i) * 4, val, ra);
        }
    }
}

/* sort: INT32 bubble sort */
void HELPER(sort)(CPURISCVState *env, target_ulong addr,
                  target_ulong total, target_ulong sort_count)
{
    uintptr_t ra = GETPC();
    int k = (int)sort_count;

    for (int i = 0; i < k - 1; i++) {
        for (int j = 0; j < k - i - 1; j++) {
            int32_t a = (int32_t)g233_ld32(env, addr + j * 4, ra);
            int32_t b = (int32_t)g233_ld32(env, addr + (j + 1) * 4, ra);
            if (a > b) {
                g233_st32(env, addr + j * 4, (uint32_t)b, ra);
                g233_st32(env, addr + (j + 1) * 4, (uint32_t)a, ra);
            }
        }
    }
}

/* crush: 8-bit to 4-bit compression */
void HELPER(crush)(CPURISCVState *env, target_ulong dst,
                   target_ulong src, target_ulong n)
{
    uintptr_t ra = GETPC();
    int count = (int)n;

    for (int i = 0; i < count / 2; i++) {
        uint8_t lo = g233_ld8(env, src + 2 * i, ra) & 0x0F;
        uint8_t hi = g233_ld8(env, src + 2 * i + 1, ra) & 0x0F;
        g233_st8(env, dst + i, lo | (hi << 4), ra);
    }
    if (count & 1) {
        uint8_t lo = g233_ld8(env, src + count - 1, ra) & 0x0F;
        g233_st8(env, dst + count / 2, lo, ra);
    }
}

/* expand: 4-bit to 8-bit decompression */
void HELPER(expand)(CPURISCVState *env, target_ulong dst,
                    target_ulong src, target_ulong n)
{
    uintptr_t ra = GETPC();
    int count = (int)n;

    for (int i = 0; i < count; i++) {
        uint8_t val = g233_ld8(env, src + i, ra);
        g233_st8(env, dst + 2 * i, val & 0x0F, ra);
        g233_st8(env, dst + 2 * i + 1, (val >> 4) & 0x0F, ra);
    }
}

/* vdot: INT32 vector dot product */
target_ulong HELPER(vdot)(CPURISCVState *env, target_ulong addr_a,
                          target_ulong addr_b)
{
    uintptr_t ra = GETPC();
    int64_t acc = 0;

    for (int i = 0; i < 16; i++) {
        int32_t a = (int32_t)g233_ld32(env, addr_a + i * 4, ra);
        int32_t b = (int32_t)g233_ld32(env, addr_b + i * 4, ra);
        acc += (int64_t)a * (int64_t)b;
    }
    return (target_ulong)acc;
}

/* vrelu: INT32 vector ReLU */
void HELPER(vrelu)(CPURISCVState *env, target_ulong dst,
                   target_ulong src, target_ulong n)
{
    uintptr_t ra = GETPC();
    int count = (int)n;

    for (int i = 0; i < count; i++) {
        int32_t val = (int32_t)g233_ld32(env, src + i * 4, ra);
        g233_st32(env, dst + i * 4, (uint32_t)(val > 0 ? val : 0), ra);
    }
}

/* vscale: INT32 vector scalar multiply */
void HELPER(vscale)(CPURISCVState *env, target_ulong dst,
                    target_ulong src, target_ulong scale)
{
    uintptr_t ra = GETPC();

    for (int i = 0; i < 16; i++) {
        int32_t val = (int32_t)g233_ld32(env, src + i * 4, ra);
        int32_t result = (int32_t)((int64_t)val * (int64_t)(target_long)scale);
        g233_st32(env, dst + i * 4, (uint32_t)result, ra);
    }
}

/* vmax: INT32 vector max reduction */
target_ulong HELPER(vmax)(CPURISCVState *env, target_ulong src,
                          target_ulong n)
{
    uintptr_t ra = GETPC();
    int count = (int)n;
    int32_t max = (int32_t)g233_ld32(env, src, ra);

    for (int i = 1; i < count; i++) {
        int32_t val = (int32_t)g233_ld32(env, src + i * 4, ra);
        if (val > max) {
            max = val;
        }
    }
    /* sign-extend INT32 to XLEN */
    return (target_ulong)(target_long)(int32_t)max;
}

/* gemm: INT32 4x4 matrix multiply */
void HELPER(gemm)(CPURISCVState *env, target_ulong dst,
                  target_ulong a, target_ulong b)
{
    uintptr_t ra = GETPC();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int64_t acc = 0;
            for (int k = 0; k < 4; k++) {
                int32_t va = (int32_t)g233_ld32(env, a + (i * 4 + k) * 4, ra);
                int32_t vb = (int32_t)g233_ld32(env, b + (k * 4 + j) * 4, ra);
                acc += (int64_t)va * (int64_t)vb;
            }
            g233_st32(env, dst + (i * 4 + j) * 4, (uint32_t)(int32_t)acc, ra);
        }
    }
}

/* vadd: INT32 vector element-wise addition */
void HELPER(vadd)(CPURISCVState *env, target_ulong dst,
                  target_ulong a, target_ulong b)
{
    uintptr_t ra = GETPC();

    for (int i = 0; i < 16; i++) {
        int32_t va = (int32_t)g233_ld32(env, a + i * 4, ra);
        int32_t vb = (int32_t)g233_ld32(env, b + i * 4, ra);
        g233_st32(env, dst + i * 4, (uint32_t)(va + vb), ra);
    }
}
