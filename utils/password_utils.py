import torch
import hashlib

def generate_password_vector(password, dim=512):
    sha = hashlib.sha256(password.encode()).digest()
    seed = [b / 255.0 for b in sha]
    while len(seed) < dim:
        seed += seed  # repeat if needed
    return torch.tensor(seed[:dim], dtype=torch.float32).unsqueeze(0)


