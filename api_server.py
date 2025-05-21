import uvicorn
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import tempfile

# ASSUMING you have get_image_embedding and client (QdrantClient) available here
# You'll need to import or define these from your embedder.py and qdrant_db.py
from embedder import ImageEmbedder # Assuming ImageEmbedder is initialized globally or passed
from qdrant_db import QdrantManager # Assuming QdrantManager is initialized globally or passed

# Inizializza l'embedder e il client Qdrant globalmente per l'API
# Oppure, considera di passare queste dipendenze all'interno delle funzioni degli endpoint
# Per semplicità in questo esempio, li inizializziamo qui.
image_embedder = ImageEmbedder()
qdrant_manager = QdrantManager(db_path="./qdrant_data")
# È buona pratica definire la collection_name qui o come variabile d'ambiente
COLLECTION_NAME = "wool_samples" # Usa la stessa collection name definita in db_generator.py

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider to restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search_similar(
    file: UploadFile = File(...),
    limit: int = 5
):
    # Save uploaded file temporarily
    # Usa Pathlib per una gestione dei percorsi più robusta
    temp_dir = tempfile.gettempdir()
    temp_file_path = Path(temp_dir) / file.filename # Mantieni il nome originale del file
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Generate embedding for uploaded image
        query_embedding = image_embedder.get_embedding(str(temp_file_path)) # get_embedding accetta str
        
        # Search for similar images
        search_results = qdrant_manager.search_similar_images( # Usa l'istanza del manager
            collection_name=COLLECTION_NAME,
            query_embedding=query_embedding, # Passa l'array numpy direttamente, il manager lo convertirà
            limit=limit
        )
        
        # Format results
        results = []
        for res in search_results:
            results.append({
                "image_id": res.id,
                "image_path": res.payload.get("image_path"), # Usa .get() per sicurezza
                "similarity": res.score
            })
        
        return {"results": results}
    finally:
        # Clean up
        if temp_file_path.exists():
            os.unlink(temp_file_path)

# Function to start the FastAPI server (No need for threading for simple execution)
# You would typically run this from the command line
# uvicorn your_app_file:app --reload --host 0.0.0.0 --port 8000
# For a script that starts itself:
def run_fastapi(host="0.0.0.0", port=8000): # Usa 0.0.0.0 per rendere accessibile da altre macchine/docker
    print(f"FastAPI running on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port) # Questo blocco è sincrono e non ritorna finché il server non si ferma

if __name__ == '__main__':
    # Rimuovi la parte di threading e IPython.display se non in Jupyter
    # Se vuoi avviare il server direttamente da questo script, chiama run_fastapi()
    run_fastapi()

