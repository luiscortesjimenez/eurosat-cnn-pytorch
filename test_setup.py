import torch

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check if Apple Silicon GPU acceleration is available
if torch.backends.mps.is_available():
    print("Success: Apple MPS (GPU acceleration) is available.")
    device = torch.device("mps")
else:
    print("MPS not available, defaulting to CPU.")
    device = torch.device("cpu")
