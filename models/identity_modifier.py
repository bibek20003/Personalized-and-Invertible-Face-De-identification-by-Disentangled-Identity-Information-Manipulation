import torch
import torch.nn.functional as F
import numpy as np

def rotate_identity(z_id, password_vec, d):
    import numpy as np
    import torch.nn.functional as F

    theta = np.deg2rad(70 + 5 * d)
    
    # 🔁 Match device
    password_vec = password_vec.to(z_id.device)

    z_id_norm = F.normalize(z_id, dim=1)
    z90 = password_vec - torch.sum(password_vec * z_id_norm, dim=1, keepdim=True) * z_id_norm
    z90 = F.normalize(z90, dim=1)

    return z_id * np.cos(theta) + z90 * np.sin(theta)
