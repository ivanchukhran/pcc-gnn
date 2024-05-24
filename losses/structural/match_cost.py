from torch.autograd import Function
from .StructuralLossesBackend import ApproxMatch, MatchCost, MatchCostGrad

class MatchCostFunction(Function):
    @staticmethod
    def forward(ctx, seta, setb):
        ctx.save_for_backward(seta, setb)
        match, temp = ApproxMatch(seta, setb)
        cost = MatchCost(seta, setb, match)
        return cost

    @staticmethod
    def backward(ctx, grad_output):
        seta, setb = ctx.saved_tensors
        grada, gradb = MatchCostGrad(seta, setb, grad_output)
        grad_output_expand = grad_output.unsqueeze(1).unsqueeze(2)
        return grada * grad_output_expand, gradb * grad_output_expand

match_cost = MatchCostFunction.apply
