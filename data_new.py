import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class OxfordIIITPetDataModule(L.LightningDataModule):
    def __init__(self, data_path:str, batch_size:int, num_workers:int, train_val_split:tuple, preprocess):
        super(OxfordIIITPetDataModule, self).__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            preprocess
        ])

        self.prepare_data_per_node = True

    def setup(self, stage=None):
        dataset = ImageFolder(root=self.data_path, transform=self.transforms)
        train_size = int(len(dataset) * self.train_val_split[0])
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)