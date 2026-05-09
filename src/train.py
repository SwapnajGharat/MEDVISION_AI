import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import MedicalDataset
from src.models import MultimodalFusionNet

def train_model():
    # 1. Setup Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Initialize Dataset & Loader
    # We assume 3 clinical features: age, glucose, blood_pressure
    dataset = MedicalDataset(csv_file="data/master_manifest.csv")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. Initialize Model
    model = MultimodalFusionNet(tabular_dim=3).to(device)
    
    # 4. Loss Function & Optimizer
    criterion = nn.BCELoss() # Binary Cross Entropy for 0/1 classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. The Training Loop
    model.train()
    epochs = 5 # Start with 5 to test
    
    print("Starting Training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, clinical, labels in train_loader:
            images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, clinical).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass (Learning)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    # 6. Save the trained weights
    torch.save(model.state_dict(), "models_saved/final_model.pth")
    print("Training complete! Model saved to models_saved/final_model.pth")

if __name__ == "__main__":
    train_model()