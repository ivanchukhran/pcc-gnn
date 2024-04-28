#!/usr/bin/env python3
from torch.autograd import Function
from losses.structural.StructuralLossesBackend import NNDistanceFunction, NNDistanceGrad

class NNDistanceFunction(Function):
    @staticmethod
    def forward(ctx, seta, setb):
        ctx.save_for_backward(seta, setb)
        dist1, idx1, dist2, idx2 = NNDistanceFunction(seta, setb)
        ctx.idx1 = idx1
        ctx.idx2 = idx2
        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        seta, setb = ctx.saved_tensors
        idx1 = ctx.idx1
        idx2 = ctx.idx2
        grada, dgradb = NNDistanceGrad(seta, setb, idx1, idx2, grad_dist1, grad_dist2)
        return grada, dgradb

nn_distance = NNDistanceFunction.apply
