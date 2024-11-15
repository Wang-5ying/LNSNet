from __future__ import print_function
import torch
import torch.nn as nn

CL = nn.TripletMarginWithDistanceLoss().cuda()
CL1 = nn.MultiMarginLoss().cuda()


def compute_cos_dis(x_sup, x_que):
    x_sup = x_sup.view(x_sup.size()[0], x_sup.size()[1], -1)
    x_que = x_que.view(x_que.size()[0], x_que.size()[1], -1)

    x_que_norm = torch.norm(x_que, p=2, dim=1, keepdim=True)
    x_sup_norm = torch.norm(x_sup, p=2, dim=1, keepdim=True)

    x_que_norm = x_que_norm.permute(0, 2, 1)
    x_qs_norm = torch.matmul(x_que_norm, x_sup_norm)

    x_que = x_que.permute(0, 2, 1)

    x_qs = torch.matmul(x_que, x_sup)
    x_qs = x_qs / (x_qs_norm + 1e-5)
    return x_qs

def sclloss(x, xt, xb):
    cosc = (1+compute_cos_dis(x, xt))*0.5
    cosb = (1+compute_cos_dis(x, xb))*0.5
    loss = -torch.log(cosc+1e-5)-torch.log(1-cosb+1e-5)
    return loss.sum()


if __name__ == "__main__":
    x_student = torch.randn(5, 640, 8, 8).cuda()
    x_student_a = torch.randn(3, 5, 384, 8, 8).cuda()
    x_teacher = torch.randn(5, 640, 8, 8).cuda()
    x_teacher1 = torch.randn(5, 640, 8, 8).cuda()
    x_teacher_a = torch.randn(3, 5, 640, 8, 8).cuda()
    # model = SupConLoss().cuda()
    out = CL(x_student, x_teacher, x_teacher1)
    print(out)