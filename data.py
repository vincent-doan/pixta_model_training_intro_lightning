import os
import pandas as pd
import torch
import lightning as L
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

class PascalVOCDataset(Dataset):
    def __init__(self, images_path:str, labels_path:str, transforms, preprocess):
        super(PascalVOCDataset, self).__init__()
        
        # LABELS
        df = pd.read_csv(labels_path, index_col=0)
        keys = df.index.tolist()
        data_tensor = torch.tensor(df.iloc[:, :].values.flatten(), dtype=torch.float32)
        self.image_path_to_label_dict = {key: data_tensor[i:i+len(df.columns)] for i, key in enumerate(keys)}
        self.num_to_label_dict = {i: key for i, key in enumerate(df.columns)}
        self.num_labels = len(self.num_to_label_dict)

        # IMAGES
        self.images_path = [images_path + path for path in sorted(os.listdir(images_path)) if path in self.image_path_to_label_dict]

        # PREPROCESS
        self.transforms = transforms
        self.preprocess = preprocess
    
    def __getitem__(self, idx):
        path = self.images_path[idx]
        image_id = path.split('/')[-1]
        label = self.image_path_to_label_dict[image_id]

        image = Image.open(path)
        image = ToTensor()(image)
        if self.transforms:
            image = self.transforms(image)
        image = self.preprocess(image)

        return image, label

    def __len__(self):
        return len(self.images_path)  

class PascalVOCDataModule(L.LightningDataModule):
    def __init__(self, train_images_path:str, val_images_path:str,
                 train_labels_path:str, val_labels_path:str, 
                 preprocess, batch_size:int=32):
        super(PascalVOCDataModule, self).__init__()

        self.train_images_path = train_images_path
        self.val_images_path = val_images_path
        self.train_labels_path = train_labels_path
        self.val_labels_path = val_labels_path
        
        self.batch_size = batch_size

        self.preprocess = preprocess
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2))
        ])

        self.prepare_data_per_node = True

    def setup(self, stage):
        self.train_dataset = PascalVOCDataset(images_path=self.train_images_path,
                                              labels_path=self.train_labels_path,
                                              transforms=self.transforms,
                                              preprocess=self.preprocess)
        self.val_dataset = PascalVOCDataset(images_path=self.val_images_path,
                                            labels_path=self.val_labels_path,
                                            transforms=None,
                                            preprocess=self.preprocess)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)