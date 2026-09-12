"""LatEq: anatomical label action, paired supervision, and aligned projection.

This module implements the three stages in Fig. 1 of the LatEq manuscript.
The caller supplies classifier logits; LatEq introduces no trainable parameters.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["LatEq"]


class LatEq(nn.Module):
    """Apply a known anatomical reflection action to targets and predictions.

    Args:
        permutation: Class mapping with ``permutation[c]`` equal to the class
            of a reflected example from class ``c``. It must be a permutation
            of ``range(K)`` and an involution: ``P[P[c]] == c``.
        flip_dim: Spatial dimension of a batched input tensor to reflect.
            Inputs use ``[N, C, H, W]`` or ``[N, C, D, H, W]``. Specify the
            dimension from the input's anatomical coordinate convention.

    The permutation is stored as a buffer. Move this module to the same device
    as the inputs, logits, and targets with ``.to(device)``.
    """

    def __init__(self, permutation: Sequence[int], flip_dim: int) -> None:
        super().__init__()
        perm = torch.as_tensor(permutation).detach().clone()
        if perm.ndim != 1 or perm.numel() == 0:
            raise ValueError("permutation must be a nonempty one-dimensional sequence.")
        if perm.dtype not in (torch.int32, torch.int64):
            raise ValueError("permutation must contain integer class indices.")
        perm = perm.long()
        identity = torch.arange(perm.numel(), device=perm.device)
        if not torch.equal(perm.sort().values, identity):
            raise ValueError("permutation must contain each class index exactly once.")
        if not torch.equal(perm[perm], identity):
            raise ValueError("permutation must satisfy P[P[c]] == c.")
        if not isinstance(flip_dim, int) or isinstance(flip_dim, bool):
            raise TypeError("flip_dim must be an integer tensor dimension.")
        self.register_buffer("permutation", perm)
        self.flip_dim = flip_dim

    def reflect(self, inputs: Tensor) -> Tensor:
        """Return the input reflection F(x), preserving values and shape.

        Args:
            inputs: Batched 2-D images or 3-D volumes in channel-first format.

        Returns:
            A tensor reflected along the configured spatial dimension.
        """
        if inputs.ndim not in (4, 5):
            raise ValueError("inputs must have shape [N, C, H, W] or [N, C, D, H, W].")
        dim = self.flip_dim if self.flip_dim >= 0 else inputs.ndim + self.flip_dim
        if not 2 <= dim < inputs.ndim:
            raise ValueError("flip_dim must refer to a spatial dimension.")
        return inputs.flip(dim)

    def transform_targets(self, targets: Tensor) -> Tensor:
        """Return the reflected targets P(y), as in Fig. 1(a).

        Args:
            targets: Integer class IDs of shape ``[N]`` and dtype ``long``, or
                floating-point soft targets of shape ``[N, K]``. Soft targets
                contain nonnegative probabilities with rows summing to one.

        Returns:
            Targets with the same shape and dtype. Soft-target permutation
            preserves the mixing coefficients used by MixUp or CutMix.
        """
        if targets.ndim == 1 and targets.dtype == torch.long:
            return self.permutation.index_select(0, targets)
        if (
            targets.ndim == 2
            and targets.shape[1] == self.permutation.numel()
            and targets.is_floating_point()
        ):
            # P equals its inverse, so indexing implements the same class action.
            return targets.index_select(1, self.permutation)
        raise ValueError("targets must be long [N] class IDs or floating [N, K] probabilities.")

    def loss(self, logits: Tensor, reflected_logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the ActionTrain objective in Fig. 1(b) and Eq. (3).

        Args:
            logits: Unnormalized classifier outputs f_theta(x), shape ``[N, K]``.
            reflected_logits: Outputs f_theta(F(x)) from the same classifier,
                with the same shape as ``logits``.
            targets: Hard or soft targets for the original inputs.

        Returns:
            Scalar mean loss: 0.5 * CE(f_theta(x), y)
            + 0.5 * CE(f_theta(F(x)), P(y)). Both branches retain gradients.
        """
        self._check_logits(logits, reflected_logits)
        reflected_targets = self.transform_targets(targets)
        return 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(reflected_logits, reflected_targets)
        )

    def forward(self, logits: Tensor, reflected_logits: Tensor) -> Tensor:
        """Return aligned probabilities q(x), as in Fig. 1(c) and Eq. (4).

        Args:
            logits: Classifier outputs f_theta(x), shape ``[N, K]``.
            reflected_logits: Outputs f_theta(F(x)) from the same classifier.

        Returns:
            Probabilities ``[N, K]`` in the original class order. For a
            deterministic classifier and the specified involutive action,
            q(F(x)) = P(q(x)). Compute classifier logits in evaluation mode.
        """
        self._check_logits(logits, reflected_logits)
        probabilities = logits.softmax(dim=1)
        reflected_probabilities = reflected_logits.softmax(dim=1)
        # Align probability coordinates before averaging the two views.
        aligned = reflected_probabilities.index_select(1, self.permutation)
        return 0.5 * (probabilities + aligned)

    def _check_logits(self, logits: Tensor, reflected_logits: Tensor) -> None:
        """Check the batch and class dimensions shared by loss and projection."""
        if (
            logits.ndim != 2
            or logits.shape != reflected_logits.shape
            or logits.shape[1] != self.permutation.numel()
        ):
            raise ValueError("both logits tensors must have shape [N, len(permutation)].")
