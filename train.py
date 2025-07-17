import os
import sys
import torch
from torch import nn, optim
from torchvision.utils import save_image

# === Ensure local imports work ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.identity_encoder import load_identity_encoder
from models.attribute_encoder import AttributeEncoder
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataloader import get_dataloader
from utils.losses import identity_loss, attribute_loss, reconstruction_loss, adversarial_loss
from config import *

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Models ===
Eid = load_identity_encoder()  # function, not .to(device)
Eattr = AttributeEncoder().to(device)
G = Generator().to(device)
D = Discriminator().to(device)

# === Optimizers ===
optimizer_G = optim.Adam(
    list(Eattr.parameters()) + list(G.parameters()),
    lr=LEARNING_RATE, betas=(0.0, 0.999)
)
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.999))

# === Loss weights ===
lambda_id = 5
lambda_attr = 10
lambda_rec = 10
lambda_adv = 0.1

# === Data loader ===
dataloader = get_dataloader("/data/Bibek/Face_deid_project/Input_Img", IMAGE_SIZE, BATCH_SIZE)

# === Output directories ===
# === Paths ===
OUTPUT_DIR = "/data/Bibek/Face_deid_project/outputs"
CHECKPOINT_DIR = "/data/Bibek/Face_deid_project/checkpoints"

# === Ensure save directories exist ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# === Training loop ===
EPOCHS = 50

for epoch in range(EPOCHS):
    for step, (images, _) in enumerate(dataloader):
        images = images.to(device)

        # Identity (fixed)
        with torch.no_grad():
            zid = Eid(images)

        # Attribute & Reconstruction
        zattr = Eattr(images)
        rec_imgs = G(zattr, zid)

        # Generator losses
        loss_id = identity_loss(Eid, rec_imgs, zid)
        loss_attr = attribute_loss(zattr, Eattr(rec_imgs))
        loss_rec = reconstruction_loss(rec_imgs, images)
        loss_adv_g = adversarial_loss(D, rec_imgs, target_is_real=True)

        loss_G = lambda_id * loss_id + lambda_attr * loss_attr + lambda_rec * loss_rec + lambda_adv * loss_adv_g

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Discriminator loss
        loss_D_real = adversarial_loss(D, images, target_is_real=True)
        loss_D_fake = adversarial_loss(D, rec_imgs.detach(), target_is_real=False)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Logging
        if step % 50 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Step [{step}] | Loss_G: {loss_G.item():.4f} | Loss_D: {loss_D.item():.4f}")
            save_image(rec_imgs[:8], f"{OUTPUT_DIR}/epoch{epoch}_step{step}.png", normalize=True)


# === Save model checkpoints ===
torch.save(Eattr.state_dict(), f"{CHECKPOINT_DIR}/attribute_encoder.pth")
torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/generator.pth")
torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/discriminator.pth")


print("✅ Training complete. Models saved in 'checkpoints/'.")
