import unittest
import torch
from easyCuda.cuda_kernel import vector_add
from utils.basic import snr_error, benchmark

class vectorAddTest(unittest.TestCase):
    def setUp(self):
        self.sizes = [8, 16, 32, 64, 128, 1024, 2048, 16 * 1024, 128 * 1024, 256 * 1024]
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda:0')

    def test_accuracy(self):
        for size in self.sizes:
            with self.subTest(size=size):
                x = torch.randn(size, dtype=self.dtype, device=self.device)
                y = torch.randn(size, dtype=self.dtype, device=self.device)

                z_pred = vector_add(x, y)
                z_real = torch.add(x, y)

                self.assertTrue(
                    snr_error(z_pred, z_real) < 1e-2,
                    "accuracy test failed, got {}, expected {}".format(z_pred, z_real)
                )
    
    def test_performance(self):
        for size in self.sizes:
            with self.subTest(size=size):
                x = torch.randn(size, dtype=self.dtype, device=self.device)
                y = torch.randn(size, dtype=self.dtype, device=self.device)
                
                shape = [size]
                flop = size

                benchmark(vector_add, shape, flop, x, y)
                benchmark(torch.add, shape, flop, x, y)

if __name__ == "__main__":
    unittest.main()