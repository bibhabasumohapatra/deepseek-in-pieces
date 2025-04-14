import torch
import torch.nn as nn

'''
To-Do:
Length Extrapolatable Rotary Embeddings
Interpolating Sequence Positions
Inference Key-Value Cache
MultiModal for Image and Audio supported changes

'''

class RoPE(nn.Module):
  def __init__(self, num_theta=10000, d=64) -> torch.Tensor:
    super().__init__()

    self.num_theta = num_theta

    self.Wm = nn.Linear(d,d)
    self.Wn = nn.Linear(d,d)

  def rotation_for_2D(self, f, theta_values):

    cos_a = torch.cos(torch.tensor([theta_values]))[0]
    sin_a = torch.sin(torch.tensor([theta_values]))[0]

    rotation_matrix = torch.tensor([[cos_a + sin_a],
                                  [sin_a, cos_a]])

    return torch.matmul(rotation_matrix, f)

  def forward(self, Xm, Xn):

    b, h, m, d = Xm.shape #(batch, heads, seq len, dimension of head)

    fq = self.Wm(Xm) # b, h, m, d
    fn = self.Wn(Xn) # b, h, m, d 

    thetas = 1.0 / (self.num_theta ** (torch.arange(0, d, 2)[: (d // 2)].float() / d))
    m_list = torch.arange(0, m).unsqueeze(1) # m x 1
    thetas =  m_list * thetas
    thetas = torch.polar(torch.ones_like(thetas), thetas) # m x d/2

    ## d >> 2
    fq_complex = torch.view_as_complex(fq.view(b, h, m, d // 2, 2))
    fn_complex = torch.view_as_complex(fn.view(b, h, m, d // 2, 2))

    fq = fq_complex @ thetas.T
    fn = fn_complex @ thetas.T
    fn = torch.view_as_real(fn_complex).view(b, h, m, d)
    fq = torch.view_as_real(fq_complex).view(b, h, m, d)

    return fq, fn


if __name__ == "__main__":
  rope = RoPE(d=64)
  Xm = torch.randn(1, 8, 10, 64)
  Xn = torch.randn(1, 8, 10, 64)
  fq, fn = rope(Xm, Xn)
  print(fq.shape) # should be (1, 8, 10, 64)
  print(fn.shape) # should be (1, 8, 10, 64)

