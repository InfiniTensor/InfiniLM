import infini.ops
import torch

m, n, k = 2, 3, 4

x = torch.randn(m, k, device="cpu")
y = torch.randn(k, n, device="cpu")
z = torch.empty(m, n, device="cpu")

infini.ops.gemm(x, y, z)

print(x)
print(y)
print(z)
print(torch.mm(x, y))
