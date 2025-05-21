# Use a Python 3.12 base image (slim version for smaller size)
FROM python:3.12-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies from requirements.txt
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files to the working directory
COPY . .

# Download the ResNet50 model weights.
# This ensures the model is available in the container and avoids downloading it at runtime.
RUN python -c "import torchvision.models as models; models.resnet50(pretrained=True)"

# Command to run when the container starts.  Starts the FastAPI server.
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]