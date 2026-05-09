import torch
from src.models import MultimodalFusionNet

# Assume we have 3 clinical features: Age, Glucose, Blood Pressure
model = MultimodalFusionNet(tabular_dim=3)
model.eval() # Set to evaluation mode

print("--- Model Initialized Successfully ---")

# Create dummy inputs(1 image, 1 patient record)
dummy_img = torch.randn(1, 3, 224, 224) 
dummy_tab = torch.randn(1, 3)

# Run prediction
with torch.no_grad():
    prediction = model(dummy_img, dummy_tab)

print(f"Test Prediction Probability: {prediction.item():.4f}")
print("System is ready for real data!")