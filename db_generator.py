import os
from pathlib import Path
from embedder import ImageEmbedder
from qdrant_db import QdrantManager

def main():
    # 1. Config
    image_dir = Path('./images')
    collection_name = "wool_samples"

    image_paths = []
    supported_extensions = ["*.png", "*.jpg", "*.jpeg"]

    # Check if the images dir exists
    if not image_dir.exists():
        print(f"Error: the images directory '{image_dir}' doesn't exists. Creste it and upload your images.")
        return

    # Get images path
    for ext in supported_extensions:
        image_paths.extend([str(f) for f in image_dir.glob(ext)])

    if not image_paths:
        print(f"No images found in '{image_dir}'.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # 2. Generation of embeddings
    print("\n--- Generating Embeddings ---")
    embedder = ImageEmbedder()
    embeddings = embedder.generate_embeddings_for_paths(image_paths)
    print(f"Generated embeddings for {len(embeddings)} images.")
    if not embeddings:
        print("Non embeddings generated. Cannot create the database.")
        return

    # 3. Cration of Qdrant database, images upload
    print("\n--- Initializing Qdrant Database ---")
    qdrant_manager = QdrantManager(db_path="./qdrant_data") # path to the folder where Qdrant files are saved


    qdrant_manager.create_and_upload_collection(
        collection_name=collection_name,
        embeddings=embeddings,
        image_dir=image_dir 
    )

    

if __name__ == '__main__':
    main()