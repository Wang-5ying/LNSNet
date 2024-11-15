import torch
import torch.nn.functional as F
x = torch.randn((3,4,5))
y = torch.randn((3,4,5))
logp_x = F.log_softmax(x,dim=-1)
p_y = F.softmax(y, dim=-1)
kl_sum = F.kl_div(logp_x, p_y, reduction='sum')
print(kl_sum)