import numpy as np
from tqdm import tqdm
import torch
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from torchvision.models import ResNet50_Weights


class ImageEmbedder:
    def __init__(self):
        # Upload the pretrained ResNet model
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
         # Remove classification layer to get embeddings
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()  # Se the model on eval mode

        # Prepare image transformation
        # We are scaling all images and normalizng them
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image_path: str):
        """
        Generate embeddings for the single image
        """
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            embedding = self.model(batch_t)

        return embedding.squeeze().cpu().numpy()

    def generate_embeddings_for_paths(self, image_paths: list[str]) -> dict:
        """
        Genera embedding per un elenco di percorsi immagine.
        Restituisce un dizionario {image_id: embedding_array}.
        """
        embeddings = {}
        for img_path in tqdm(image_paths, desc="Generating embeddings"):
            try:
                img_id = os.path.basename(img_path).split('.')[0]
                embedding = self.get_embedding(img_path)
                embeddings[img_id] = embedding
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        return embeddings



# if __name__ == '__main__':
#     # Example of usage: create test embeddings for two images
#     from pathlib import Path
#     image_dir = Path('./images')
#     if not image_dir.exists():
#         print(f"La directory {image_dir} non esiste. Crea alcune immagini di esempio.")
#         exit()

#     image_paths = [str(f) for f in image_dir.glob("*.png")]
#     if not image_paths:
#         print(f"Nessuna immagine trovata in {image_dir}. Aggiungi immagini .png per testare.")
#         exit()

#     embedder = ImageEmbedder()
#     test_embeddings = embedder.generate_embeddings_for_paths(image_paths[:2]) # Genera embedding per le prime 2 immagini
#     print(f"Generated {len(test_embeddings)} test embeddings.")
#     if test_embeddings:
#         first_embedding_id = list(test_embeddings.keys())[0]
#         print(f"Example embedding shape for {first_embedding_id}: {test_embeddings[first_embedding_id].shape}")