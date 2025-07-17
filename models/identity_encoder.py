import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
import numpy as np

def load_identity_encoder():
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # or 'CUDAExecutionProvider' for GPU
    app.prepare(ctx_id=0)

    def encode_identity(batch_tensor):
        embeddings = []
        for img_tensor in batch_tensor:
            img = (img_tensor.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            faces = app.get(img)
            if faces:
                emb = torch.tensor(faces[0].normed_embedding, dtype=torch.float32).unsqueeze(0)
            else:
                emb = torch.zeros(1, 512)
            embeddings.append(emb)
        return F.normalize(torch.cat(embeddings, dim=0), dim=1)

    return encode_identity
