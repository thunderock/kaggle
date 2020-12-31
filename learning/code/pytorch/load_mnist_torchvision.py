import torch
import torchvision

mnist = torchvision.datasets.MNIST(
    '/tmp', download=True, transform=torchvision.transforms.ToTensor()
    )

mnist[0]
batches = torch.utils.data.DataLoader(mnist, batch_size=32)
batch_averages = torch.Tensor([batch[0].mean() for batch in batches])

batch_averages.mean()
all_images = torch.cat([image for image, label in mnist])

all_images.shape, all_images.mean()
