import torch
import torchvision
import os
from PIL import Image


class BSDS500(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, device="cpu"):
        self.path = path
        self.transform = transform
        self.device = device

        # Assuming all files in the directory are images
        self.images_paths = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images_paths)

    def load_image(self, img_path):
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        image = self.load_image(img_path).to(self.device)

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    dataset = BSDS500(
        path="data/BSDS500/train",
        transform=torchvision.transforms.RandomCrop((256, 256)),
    )

    import matplotlib.pyplot as plt

    f, axs = plt.subplots(2, 2)
    for ax in axs.flatten():
        img = dataset[0]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")

    plt.show()
