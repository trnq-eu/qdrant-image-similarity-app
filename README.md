# Image Similarity Search Application

This application allows you to upload an image and find similar images based on visual content using a FastAPI backend for image embedding and similarity search, a Qdrant vector database for storing embeddings, and a Streamlit frontend for a user-friendly interface.

## Prerequisites

*   [Docker](https://www.docker.com/) installed on your system
*   Python 3.12
*   [Streamlit](https://streamlit.io/) installed
*   A CUDA enabled GPU is recommended for faster inference.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate # Windows
    ```

3.  **Create a `requirements.txt` file:**

    Create a `requirements.txt` file in the root directory of the project with the following content:

    ```
    fastapi
    uvicorn
    python-multipart
    requests
    Pillow
    torch
    torchvision
    qdrant-client
    streamlit
    tqdm
    ```

    Alternatively, generate it automatically from your virtual environment:

    ```bash
    pip freeze > requirements.txt
    ```

4.  **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Images Folder:**

    Create a folder named `images` in the root directory of the project. Place the images you want to search through in this folder.

6.  **Generate Embeddings and Load into Qdrant:**

    Run the `db_generator.py` script to generate embeddings for the images in the `images` folder and upload them to the Qdrant database:

    ```bash
    python db_generator.py
    ```

    This will create a `qdrant_data` directory.

7.  **Build the Docker Image (for FastAPI Backend):**

    ```bash
    docker build -t image-similarity-search .
    ```

8.  **Run the FastAPI Backend in Docker:**

    ```bash
    docker run -p 8000:8000 image-similarity-search
    ```

9.  **Run the Streamlit Frontend Locally:**

    Open a new terminal, activate your virtual environment, and run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Access the Streamlit Application:**

    Open your web browser and go to `http://localhost:8501`.

2.  **Upload an Image:**

    Use the file uploader to select an image from your local machine.

3.  **Find Similar Images:**

    Click the "Find Similar Images" button. The application will:

    *   Send the image to the FastAPI backend (running in Docker) at `http://localhost:8000`.
    *   Generate an embedding for the uploaded image.
    *   Search for similar images in the Qdrant database.
    *   Display the results in a grid, showing the similar images and their similarity scores.

## Important Notes:

*   The FastAPI backend runs in a Docker container.
*   The Streamlit frontend runs directly on your host machine (not in Docker).
*   Ensure that the FastAPI server is running *before* you run the Streamlit app.

## Troubleshooting

*   **"Connection refused" error:** Make sure the FastAPI server (in Docker) is running *before* accessing the Streamlit app.
*   **Image loading errors:** Ensure that the image paths in the Qdrant database are correct and that the images exist in the `images` directory.
*   **Slow performance:** If the application is running slowly, consider using a GPU for faster image embedding.
*   **CUDA errors**: Verify that your Docker container has access to your host CUDA enabled GPU.

## Credits

This is my personal take on the code written in this [tutorial](https://blog.veskovujovic.me/posts/image-similarity-with-vector-db/?utm_source=pocket_shared).
[Origina repo](https://github.com/vesko-vujovic/image-similarity-vector-db) 

*   This application uses [FastAPI](https://fastapi.tiangolo.com/) for the backend.
*   [Streamlit](https://streamlit.io/) for the frontend.
*   [Qdrant](https://qdrant.tech/) for the vector database.
*   [PyTorch](https://pytorch.org/) and a pretrained ResNet50 model for image embedding.