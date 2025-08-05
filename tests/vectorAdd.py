import unittest
import torch
from easyCuda.kernel import vector_add

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
                z_real = x + y

                self.assertTrue(
                    torch.allclose(z_pred, z_real, atol=1e-4, rtol=1e-4),
                    "accuracy test failed, expected {}, got {}".format(z_real, z_pred)
                )

if __name__ == "__main__":
    unittest.main()