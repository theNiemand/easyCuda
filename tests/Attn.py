import torch
import torch.nn.functional as F
import unittest
from easyCuda.triton_kernel.prefillAttn import prefillAttn
from utils.basic import snr_error, benchmark


class prefillAttnTest(unittest.TestCase):
    def setUp(self):
        self.seq_len = 1024
        self.q_head_num = 128
        self.kv_head_num = 16
        self.head_dim = 128
        self.softmax_scale = 1.0 / (self.head_dim ** 0.5)
        
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16

    def prefillAttn_torch(
        self,
        Q: torch.Tensor,     # [num_tokens, q_head_num, head_dim]
        K: torch.Tensor,     # [num_tokens, kv_head_num, head_dim]
        V: torch.Tensor,     # [num_tokens, kv_head_num, head_dim]
        seqs_start_loc: torch.Tensor,
        seqs_len: torch.Tensor
    ):
        batch_size = seqs_start_loc.shape[0]
        num_tokens, q_head_num, head_dim = Q.shape
        _, kv_head_num, _ = K.shape
        
        # 计算GQA组大小
        GQA_GROUP_SIZE = q_head_num // kv_head_num
        
        # 创建输出张量
        O = torch.zeros_like(Q)
        
        # 遍历每个batch
        for batch_id in range(batch_size):
            start_loc = seqs_start_loc[batch_id].item()
            seq_len = seqs_len[batch_id].item()
            
            # 提取当前序列的QKV
            Q_seq = Q[start_loc:start_loc + seq_len]  # [seq_len, q_head_num, head_dim]
            K_seq = K[start_loc:start_loc + seq_len]  # [seq_len, kv_head_num, head_dim]
            V_seq = V[start_loc:start_loc + seq_len]  # [seq_len, kv_head_num, head_dim]
            
            # 对每个query head进行计算
            for q_head_id in range(q_head_num):
                # 确定对应的KV head
                kv_head_id = q_head_id // GQA_GROUP_SIZE
                
                # 获取当前head的Q, K, V
                q = Q_seq[:, q_head_id, :]  # [seq_len, head_dim]
                k = K_seq[:, kv_head_id, :]  # [seq_len, head_dim]
                v = V_seq[:, kv_head_id, :]  # [seq_len, head_dim]
                
                # 计算注意力分数: Q @ K^T / sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)  # [seq_len, seq_len]
                
                # 应用因果掩码：每个位置只能看到自己和之前的位置
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal_mask, float('-inf'))
                
                # 计算softmax得到注意力权重
                attn_weights = F.softmax(scores, dim=-1)  # [seq_len, seq_len]
                
                # 计算加权后的输出: attn_weights @ V
                output = torch.matmul(attn_weights, v)  # [seq_len, head_dim]
                
                # 存储结果
                O[start_loc:start_loc + seq_len, q_head_id, :] = output
        
        return O

    
    def test_accuracy(self):
        Q = torch.randn(self.seq_len, self.q_head_num, self.head_dim, dtype=self.dtype, device=self.device)
        K = torch.randn(self.seq_len, self.kv_head_num, self.head_dim, dtype=self.dtype, device=self.device)
        V = torch.randn(self.seq_len, self.kv_head_num, self.head_dim, dtype=self.dtype, device=self.device)
        
        seqs_start_loc = torch.tensor([0], dtype=torch.int32, device=self.device)
        seqs_len = torch.tensor([self.seq_len], dtype=torch.int32, device=self.device)
        O_pred = prefillAttn(Q, K, V, seqs_start_loc, seqs_len)
        O_real = self.prefillAttn_torch(Q, K, V, seqs_start_loc, seqs_len)

        print(f"{snr_error(O_pred, O_real) = }")

        self.assertTrue(
            snr_error(O_pred, O_real) < 1e-2,
            "accuracy test failed, got {}, expected {}".format(O_pred, O_real)
        )
    
    def test_performance(self):
        Q = torch.randn(self.seq_len, self.q_head_num, self.head_dim, dtype=self.dtype, device=self.device)
        K = torch.randn(self.seq_len, self.kv_head_num, self.head_dim, dtype=self.dtype, device=self.device)
        V = torch.randn(self.seq_len, self.kv_head_num, self.head_dim, dtype=self.dtype, device=self.device)

        shape = [self.seq_len, self.q_head_num, self.head_dim]
        flop = 2 * self.seq_len * (self.q_head_num * self.head_dim) * self.seq_len + 2 * self.seq_len * self.seq_len * (self.q_head_num * self.head_dim)

        seqs_start_loc = torch.tensor([0], dtype=torch.int32, device=self.device)
        seqs_len = torch.tensor([self.seq_len], dtype=torch.int32, device=self.device)
        benchmark(prefillAttn, shape, flop, Q, K, V, seqs_start_loc, seqs_len)
        benchmark(self.prefillAttn_torch, shape, flop, Q, K, V, seqs_start_loc, seqs_len)


if __name__ == "__main__":
    unittest.main()