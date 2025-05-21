from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
from pathlib import Path
import os

class QdrantManager:
    def __init__(self, db_path: str = "./qdrant_data"):
        self.client = QdrantClient(path=db_path)

    def create_and_upload_collection(self, collection_name: str, embeddings: dict, image_dir: Path, batch_size: int = 10):
        """
        Search or creates a Qdrant collection and upload embeddings
        """
        if not embeddings:
            print("No embdedding provided.")
            return

        # Ottieni la dimensione del vettore dal primo embedding
        vector_size = next(iter(embeddings.values())).shape[0]

        print(f"Creating or re-creating collection '{collection_name}' with vector size {vector_size}...")
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created.")

        points = []
        point_id_counter = 1 # Initialize a counter for PoinStructs
        for img_id, embedding in embeddings.items():
            points.append(PointStruct(
                id=point_id_counter,
                vector=embedding.tolist(), # Convert NumPy array to Python list
                payload={"image_path": str(image_dir / f"{img_id}.png"), "name": img_id}
            ))
            point_id_counter += 1

        print(f"Uploading {len(points)} embeddings to Qdrant in batches...")
        # Upload in batches
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch_points,
                wait=True # Attendi che l'operazione sia completata
            )
        print(f"Uploaded {len(points)} embeddings to Qdrant collection '{collection_name}'.")

    def search_similar_images(self, collection_name: str, query_embedding: np.ndarray, limit: int = 5):
        """
        Cerca immagini simili nella collezione Qdrant.
        """
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        return search_result

# if __name__ == '__main__':
#     # Esempio di utilizzo (solo per testare il QdrantManager)
#     q_manager = QdrantManager()
#     # Per testare questo, avresti bisogno di alcuni embedding fittizi
#     dummy_embeddings = {
#         "img_a": np.random.rand(2048),
#         "img_b": np.random.rand(2048)
#     }
#     dummy_image_dir = Path('./images') # Assicurati che esista o creane una fittizia
#     if not dummy_image_dir.exists():
#         os.makedirs(dummy_image_dir) # Crea la directory se non esiste
#         # Crea anche dei file immagine fittizi per i payload
#         with open(dummy_image_dir / "img_a.png", "w") as f: f.write("")
#         with open(dummy_image_dir / "img_b.png", "w") as f: f.write("")


#     q_manager.create_and_upload_collection(
#         collection_name="test_collection",
#         embeddings=dummy_embeddings,
#         image_dir=dummy_image_dir
#     )

#     # Esempio di ricerca
#     if dummy_embeddings:
#         query_vec = dummy_embeddings["img_a"]
#         results = q_manager.search_similar_images("test_collection", query_vec)
#         print("\nSearch Results (dummy):")
#         for hit in results:
#             print(f"ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")