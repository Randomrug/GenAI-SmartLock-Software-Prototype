from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
import torch

device = 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)

    if face is None:
        return None

    with torch.no_grad():
        embedding = model(face.unsqueeze(0))

    embedding = embedding.numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)  # normalize

    return embedding
