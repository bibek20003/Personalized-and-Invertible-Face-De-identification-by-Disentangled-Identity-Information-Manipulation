import torch.nn.functional as F
import torch


def identity_loss(Eid, img_fake, z_real):
    z_fake = Eid(img_fake)
    return 1 - F.cosine_similarity(z_fake, z_real).mean()

def attribute_loss(z_real, z_fake):
    return F.mse_loss(z_fake, z_real)

def reconstruction_loss(recon, real):
    return F.mse_loss(recon, real)

def adversarial_loss(D, x, target_is_real=True):
    pred = D(x)
    labels = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, labels)
