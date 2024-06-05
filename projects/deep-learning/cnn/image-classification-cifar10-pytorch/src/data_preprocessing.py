from torchvision.transforms import transforms

from utils import NORMALIZE


class CIFAR10Preprocessor:
    def __init__(self):
        self.mean, self.std = NORMALIZE
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def preprocess(self, data):
        return self.transform(data)
