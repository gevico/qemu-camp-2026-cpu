static bool trans_dma(DisasContext *ctx, arg_dma *a)
{
    TCGv dst = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv src = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv gran = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_dma(tcg_env, dst, src, gran);
    return true;
}

static bool trans_sort(DisasContext *ctx, arg_sort *a)
{
    TCGv addr = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv sort_num = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv array_num = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_sort(tcg_env, addr, array_num, sort_num);
    return true;
}

static bool trans_crush(DisasContext *ctx, arg_crush *a)
{
    TCGv dst = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv src = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv num = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_crush(tcg_env, dst, src, num);
    return true;
}

static bool trans_expand(DisasContext *ctx, arg_expand *a)
{
    TCGv dst = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv src = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv num = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_expand(tcg_env, dst, src, num);
    return true;
}

static bool trans_vdot(DisasContext *ctx, arg_vdot *a)
{
    TCGv addr_a = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv addr_b = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dst = dest_gpr(ctx, a->rd);
    gen_helper_vdot(dst, tcg_env, addr_a, addr_b);
    gen_set_gpr(ctx, a->rd, dst);
    return true;
}

static bool trans_vrelu(DisasContext *ctx, arg_vrelu *a)
{
    TCGv dst = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv src = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv num = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_vrelu(tcg_env, dst, src, num);
    return true;
}

static bool trans_vscale(DisasContext *ctx, arg_vscale *a)
{
    TCGv dst = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv src = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv scale = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_vscale(tcg_env, dst, src, scale);
    return true;
}

static bool trans_vmax(DisasContext *ctx, arg_vmax *a)
{
    TCGv src = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv num = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dst = dest_gpr(ctx, a->rd);
    gen_helper_vmax(dst, tcg_env, src, num);
    gen_set_gpr(ctx, a->rd, dst);
    return true;
}

static bool trans_gemm(DisasContext *ctx, arg_gemm *a)
{
    TCGv dst = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv a_addr = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv b_addr = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_gemm(tcg_env, dst, a_addr, b_addr);
    return true;
}

static bool trans_vadd(DisasContext *ctx, arg_vadd *a)
{
    TCGv dst = get_gpr(ctx, a->rd, EXT_NONE);
    TCGv a_addr = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv b_addr = get_gpr(ctx, a->rs2, EXT_NONE);
    gen_helper_vadd(tcg_env, dst, a_addr, b_addr);
    return true;
}
