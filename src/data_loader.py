import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np

class MedicalDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Read the image path from 
        img_path = self.data.iloc[idx]['image_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) # Standard size for EfficientNet
        
        # Scale image to 0-1 and convert to Tensor
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        
        # Clinical feature
        clinical = self.data.iloc[idx][['age', 'glucose', 'blood_pressure']].values.astype(np.float32)
        
        # Label
        label = self.data.iloc[idx]['label']

        return image, torch.tensor(clinical), torch.tensor(label, dtype=torch.float32)