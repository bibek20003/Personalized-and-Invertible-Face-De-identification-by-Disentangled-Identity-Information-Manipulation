import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.identity_encoder import load_identity_encoder
from models.attribute_encoder import AttributeEncoder
from models.generator import Generator
from models.identity_modifier import rotate_identity
from utils.password_utils import generate_password_vector
from config import *

def protect_image(img_path, output_path, password, d, Eid, Eattr, G, device):
    # === Load and preprocess image ===
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to open {img_path}: {e}")
        return

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    img_tensor = tf(img).unsqueeze(0).to(device)

    # === Extract features ===
    zid = Eid(img_tensor)
    zattr = Eattr(img_tensor)
    zref = generate_password_vector(password).to(device)
    znew = rotate_identity(zid, zref, d)

    # === Generate protected image ===
    deid_img = G(zattr, znew)

    # === Save output ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_img = transforms.ToPILImage()(deid_img.squeeze(0).cpu())
    output_img.save(output_path)
    print(f"✅ Saved: {output_path}")

def protect_folder(input_dir, output_dir, password, privacy_level):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load models and weights once ===
    Eid = load_identity_encoder()
    Eattr = AttributeEncoder().to(device)
    G = Generator().to(device)
    Eattr.load_state_dict(torch.load("/data/Bibek/Face_deid_project/checkpoints/attribute_encoder.pth", map_location=device))
    G.load_state_dict(torch.load("/data/Bibek/Face_deid_project/checkpoints/generator.pth", map_location=device))

    # === Loop through all images ===
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            protect_image(input_path, output_path, password, privacy_level, Eid, Eattr, G, device)

# === CLI ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='/data/Bibek/Face_deid_project/Input_Img/fake_class')
    parser.add_argument('--output_dir', default='protected', help='/data/Bibek/Zace_deid_project/protected')
    parser.add_argument('--password', required=True, help='Password for identity rotation')
    parser.add_argument('--privacy', type=int, default=7, help='Privacy level (5–9 recommended)')
    args = parser.parse_args()

    protect_folder(args.input_dir, args.output_dir, args.password, args.privacy)
    
