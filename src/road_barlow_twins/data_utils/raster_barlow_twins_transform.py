import torch
import torchvision.transforms as transforms


class BarlowTwinsTransform:
    """Src: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/barlow-twins.html"""

    def __init__(
        self,
        train=True,
        input_height=224,
        gaussian_blur=True,
        jitter_strength=1.0,
        normalize=None,
    ):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5
                )
            )

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose(
                [transforms.ToTensor(), normalize]
            )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(self.input_height, scale=(0.6, 1.0)), # default 0.08, 1.0
                #transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform,
            ]
        )

    def __call__(self, sample):
        return (
            self.transform(torch.from_numpy(sample)),
            self.transform(torch.from_numpy(sample)),
        )