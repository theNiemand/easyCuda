# pyright: reportUnreachable=false
import torch
import triton
import triton.language as tl


@triton.jit
def prefillAttn_kernel(
    Q: torch.Tensor,     # [num_tokens, q_head_num, head_dim]
    K: torch.Tensor,     # [num_tokens, kv_head_num, head_dim]
    V: torch.Tensor,     # [num_tokens, kv_head_num, head_dim]
    O: torch.Tensor,     # [num_tokens, q_head_num, head_dim]

    seqs_start_loc: torch.Tensor,   # [batch_size]
    seqs_len: torch.Tensor,         # [batch_size]

    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    head_dim: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,

    Tile_Q: tl.constexpr,
    Tile_KV: tl.constexpr
):
    batch_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    tile_q_id = tl.program_id(2)

    kv_head_id = q_head_id // GQA_GROUP_SIZE

    # get the start loc of the current batch
    start_loc = tl.load(seqs_start_loc + batch_id)

    # get the seq len of the current batch
    seq_len = tl.load(seqs_len + batch_id)

    # get the start loc of the current tile
    q_tile_start_loc = start_loc + tile_q_id * Tile_Q

    # if the tile is out of the seq len, return
    if q_tile_start_loc >= seq_len:
        return
    
    # seq offet
    Q_tile_offset = q_tile_start_loc * (q_head_num * head_dim)
    KV_seq_offset = start_loc * (kv_head_num * head_dim)

    # head offet
    Q_head_offset = q_head_id * head_dim
    KV_head_offset = kv_head_id * head_dim

    # tile range
    Q_tile_range = tl.arange(0, Tile_Q)
    KV_tile_range = tl.arange(0, Tile_KV)

    # head_dim range
    head_dim_range = tl.arange(0, head_dim)

    # ptrs
    Q_ptr = Q + Q_tile_offset + Q_head_offset + Q_tile_range[:, None] * (q_head_num * head_dim) + head_dim_range[None, :]
    O_ptr = O + Q_tile_offset + Q_head_offset + Q_tile_range[:, None] * (q_head_num * head_dim) + head_dim_range[None, :]
    K_base_ptr = K + KV_seq_offset + KV_head_offset + KV_tile_range[None, :] * (kv_head_num * head_dim) + head_dim_range[:, None]
    V_base_ptr = V + KV_seq_offset + KV_head_offset + KV_tile_range[:, None] * (kv_head_num * head_dim) + head_dim_range[None, :]

    # mask
    q_mask = q_tile_start_loc + Q_tile_range < seq_len


    # load Q
    Q_tile = tl.load(Q_ptr, mask=q_mask[:, None], other=0.0, cache_modifier=".cg")
    
    m = tl.full([Tile_Q], value=float("-1e20"), dtype=tl.float32)
    l = tl.zeros([Tile_Q], dtype=tl.float32)
    O_acc = tl.zeros([Tile_Q, head_dim], dtype=tl.float32)

    # Cal non-diagonal attn, no mask needed
    for tile_kv_loc  in range(0, tile_q_id * Tile_Q, Tile_KV):
        KV_tile_offset = tile_kv_loc * (kv_head_num * head_dim)
       
        # load K
        K_ptr = K_base_ptr + KV_tile_offset
        K_tile = tl.load(K_ptr, cache_modifier=".cg")
       
        # Cal attn weight (no denominator)
        scale = tl.sqrt(tl.cast(head_dim, tl.float32))
        qk = tl.dot(Q_tile, K_tile).to(tl.float32) / scale

        # Cal max for safe softmax
        m_new = tl.maximum(m, tl.max(qk, axis=1))

        # Cal exp(qk - m_new)
        exp_qk = tl.exp(qk - m_new[:, None])

        # load V
        V_ptr = V_base_ptr + KV_tile_offset
        V_tile = tl.load(V_ptr, cache_modifier=".cg")
        
        # Cal m_compensation and l
        m_compensation = tl.exp(m - m_new)
        l = l * m_compensation + tl.sum(exp_qk, axis=1)
        m = m_new

        # Cal attn score
        attn_score = tl.dot(exp_qk.to(tl.bfloat16), V_tile)

        # compensate O_acc
        O_acc = O_acc * m_compensation[:, None] + attn_score
    
    # Cal diagonal attn
    for tile_kv_loc in range(tile_q_id * Tile_Q, (tile_q_id + 1) * Tile_Q, Tile_KV):
        KV_tile_offset = tile_kv_loc * (kv_head_num * head_dim)
        
        # Cal KV mask
        kv_mask = start_loc + tile_kv_loc + KV_tile_range < seq_len

        # load K
        K_ptr = K_base_ptr + KV_tile_offset
        K_tile = tl.load(K_ptr, mask=kv_mask[None, :], other=0.0, cache_modifier=".cg")

        # Cal attn weight (no denominator)
        scale = tl.sqrt(tl.cast(head_dim, tl.float32))
        qk = tl.dot(Q_tile, K_tile).to(tl.float32) / scale

        # mask for casual-attn
        causal_mask = (start_loc + tile_kv_loc + KV_tile_range)[None, :] <= (q_tile_start_loc + Q_tile_range)[:, None]
        seq_mask = (start_loc + tile_kv_loc + KV_tile_range)[None, :] < seq_len
        combined_mask = causal_mask & seq_mask
        qk = tl.where(combined_mask, qk, -1e30)

        # Cal max for safe softmax
        m_new = tl.maximum(m, tl.max(qk, axis=1))

        # Cal exp(qk - m_new)
        exp_qk = tl.exp(qk - m_new[:, None])

        # load V
        V_ptr = V_base_ptr + KV_tile_offset
        V_tile = tl.load(V_ptr, mask=kv_mask[:, None], other=0.0, cache_modifier=".cg")
        
        # Cal m_compensation and l
        m_compensation = tl.exp(m - m_new)
        l = l * m_compensation + tl.sum(exp_qk, axis=1)
        m = m_new

        # Cal attn score
        attn_score = tl.dot(exp_qk.to(tl.bfloat16), V_tile)

        # compensate O_acc
        O_acc = O_acc * m_compensation[:, None] + attn_score
    
    # store O
    tl.store(O_ptr, O_acc / l[:, None], mask=q_mask[:, None])


def prefillAttn(
    Q: torch.Tensor,     # [num_tokens, q_head_num, head_dim]
    K: torch.Tensor,     # [num_tokens, kv_head_num, head_dim]
    V: torch.Tensor,     # [num_tokens, kv_head_num, head_dim]

    seqs_start_loc: torch.Tensor,
    seqs_len: torch.Tensor
):
    q_head_num = 128
    kv_head_num = 16
    head_dim = 128
    GQA_GROUP_SIZE = 8

    assert(Q.shape[1] == q_head_num, "q_head_num mismatch")
    assert(K.shape[1] == kv_head_num, "kv_head_num mismatch")
    assert(V.shape[1] == kv_head_num, "kv_head_num mismatch")

    assert(Q.shape[2] == K.shape[2], "head_dim mismatch")
    assert(Q.shape[2] == V.shape[2], "head_dim mismatch")

    assert(Q.shape[0] == K.shape[0], "num_tokens mismatch")
    assert(Q.shape[0] == V.shape[0], "num_tokens mismatch")

    # alloc O tensor
    O = torch.empty_like(Q)

    # get batch size
    batch_size = seqs_len.shape[0]

    # get the max seq len
    max_seq_len = seqs_len.max()

    Tile_Q = 128
    Tile_KV = 128

    grid = (batch_size, q_head_num, triton.cdiv(max_seq_len, Tile_Q))

    num_warps = 8
    num_stages = 3
    prefillAttn_kernel[grid](
        Q, K, V, O,
        seqs_start_loc, seqs_len,
        q_head_num, kv_head_num, head_dim, GQA_GROUP_SIZE,
        Tile_Q, Tile_KV,
        num_warps = num_warps,
        num_stages = num_stages
    )

    return O