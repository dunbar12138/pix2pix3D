import torch
import torch.nn.functional as F

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if (h != ht) or (w != wt):
        # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode='bilinear', align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    loss = F.cross_entropy(input, target, weight=weight, reduction='mean')

    return loss