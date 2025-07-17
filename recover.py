import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add project root to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.identity_encoder import load_identity_encoder
from models.attribute_encoder import AttributeEncoder
from models.generator import Generator
from utils.password_utils import generate_password_vector
from config import *

def recover_image(image_path, password, d):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load and preprocess image ===
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    img_tensor = tf(img).unsqueeze(0).to(device)

    # === Load models ===
    Eid = load_identity_encoder()  # returns a function, not a model
    Eattr = AttributeEncoder().to(device)
    G = Generator().to(device)

    # === Load trained weights ===
    Eattr.load_state_dict(torch.load("/data/Bibek/Face_deid_project/checkpoints/attribute_encoder.pth", map_location=device))
    G.load_state_dict(torch.load("/data/Bibek/Face_deid_project/checkpoints/generator.pth", map_location=device))

    # === Extract features from protected image ===
    znew = Eid(img_tensor)
    zattr = Eattr(img_tensor)

    # === Generate password reference vector ===
    zref = generate_password_vector(password).to(device)

    # === Invert identity rotation ===
    zref = zref.to(znew.device)
    theta = torch.tensor((70 + 5 * d) * 3.14159 / 180).to(znew.device)
    A = torch.sum(znew * zref, dim=1, keepdim=True)
    zid = (znew - zref * torch.sin(theta)) / (torch.cos(theta) - A * torch.sin(theta))

    # === Reconstruct original image ===
    restored_img = G(zattr, zid)

    # === Save result ===
    out_img = transforms.ToPILImage()(restored_img.squeeze().cpu())
    out_img.save("recovered.jpg")
    print("✅ Recovered image saved as recovered.jpg")

# === Command-line interface ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to protected image')
    parser.add_argument('--password', required=True, help='Password used during protection')
    parser.add_argument('--privacy', type=int, default=7, help='Same privacy level used in protection')
    args = parser.parse_args()

    recover_image(args.image, args.password, args.privacy)
