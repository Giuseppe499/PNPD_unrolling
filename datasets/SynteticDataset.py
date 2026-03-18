import torch
from torch.utils.data import Dataset
from typing import Callable


class SynteticDataset(Dataset):
    def __init__(
        self, ground_truth: Dataset, transform: Callable = torch.nn.Identity()
    ):
        self.ground_truth = ground_truth
        self.transform = transform

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        gt = self.ground_truth[idx]
        observed = self.transform(gt)
        return observed, gt


if __name__ == "__main__":
    from BSDS500 import BSDS500
    from torchvision import transforms
    from math_extras import blur_image, gaussian_psf
    from torch.fft import rfft2
    import matplotlib.pyplot as plt

    size = (256, 256)

    psf_rfft2 = rfft2(gaussian_psf(sigma=2, size=size))

    def blur_operator(image):
        return blur_image(image, psf_rfft2)

    dataset = SynteticDataset(
        ground_truth=BSDS500(
            path="data/BSDS500/train", transform=transforms.RandomCrop(size)
        ),
        transform=blur_operator,
    )

    f, axs = plt.subplots(2, 2)
    for ax1, ax2 in zip(axs[:, 0], axs[:, 1]):
        observed, gt = dataset[0]

        ax1.imshow(gt.squeeze(), cmap="gray")
        ax1.axis("off")
        ax1.set_title("Ground Truth")

        ax2.imshow(observed.squeeze(), cmap="gray")
        ax2.axis("off")
        ax2.set_title("Observed")

    plt.show()
