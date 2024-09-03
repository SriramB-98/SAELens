from typing import Any, Tuple

import torch


def rectangle_pt(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class Step(torch.autograd.Function):
    BANDWIDTH = 0.001

    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: Tuple[torch.Tensor, torch.Tensor], output: torch.Tensor
    ) -> None:
        x, threshold = inputs
        del output
        ctx.save_for_backward(x, threshold)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, threshold = ctx.saved_tensors
        x_grad = 0.0 * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(1.0 / Step.BANDWIDTH)
            * rectangle_pt((x - threshold) / Step.BANDWIDTH)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad


class JumpReLU(torch.autograd.Function):
    BANDWIDTH = 0.001

    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return x * (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: Tuple[torch.Tensor, torch.Tensor], output: torch.Tensor
    ) -> None:
        x, threshold = inputs
        del output
        ctx.save_for_backward(x, threshold)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / JumpReLU.BANDWIDTH)
            * rectangle_pt((x - threshold) / JumpReLU.BANDWIDTH)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad
