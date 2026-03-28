"""
Epic 3 — Image pool for stable CycleGAN discriminator training.

Implements the image buffer trick from Shrivastava et al. (2017) where a
pool of previously generated images is maintained.  When the discriminator
is updated, images are randomly drawn from this pool instead of always
using the latest generator output.  This reduces oscillation and improves
training stability.

Usage::

    from udm_epic3.data.image_pool import ImagePool

    pool = ImagePool(pool_size=50)
    fake_images_for_disc = pool.query(fake_images)
"""

from __future__ import annotations

import random

import torch


class ImagePool:
    """Buffer of previously generated images for discriminator training.

    For each image in a query batch, with 50% probability the image is
    returned unchanged; otherwise it is swapped with a random image from
    the pool (and the new image is stored in its place).

    Parameters
    ----------
    pool_size : int
        Maximum number of images to store.  When ``pool_size=0`` the pool
        is disabled and ``query`` is a no-op passthrough.
    """

    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self._images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return a batch of images, mixing current and previously stored ones.

        Args:
            images: Batch tensor of shape ``[B, C, H, W]``.

        Returns:
            Tensor of the same shape, with some images potentially replaced
            by older pool entries.
        """
        if self.pool_size == 0:
            return images

        result: list[torch.Tensor] = []
        for img in images:
            img = img.unsqueeze(0)  # (1, C, H, W)

            if len(self._images) < self.pool_size:
                # Pool not full yet — store and return the image as-is.
                self._images.append(img.clone())
                result.append(img)
            else:
                if random.random() > 0.5:
                    # Swap with a random pool entry.
                    idx = random.randint(0, self.pool_size - 1)
                    stored = self._images[idx].clone()
                    self._images[idx] = img.clone()
                    result.append(stored)
                else:
                    # Return current image unchanged.
                    result.append(img)

        return torch.cat(result, dim=0)

    def __len__(self) -> int:
        return len(self._images)

    def __repr__(self) -> str:
        return f"ImagePool(pool_size={self.pool_size}, stored={len(self._images)})"
