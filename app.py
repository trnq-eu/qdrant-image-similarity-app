# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "streamlit",
# ]
# ///
import streamlit as st
import requests
from PIL import Image
import io

def main():
    st.title("Image Similarity Search")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Create a copy of the file for sending to the API
        bytes_data = io.BytesIO()
        image.save(bytes_data, format='JPEG')
        bytes_data.seek(0)
        
        # Search for similar images
        if st.button("Find Similar Images"):
            with st.spinner("Searching..."):
                files = {"file": ("image.jpg", bytes_data, "image/jpeg")}
                response = requests.post("http://localhost:8000/search", files=files)
                
                if response.status_code == 200:
                    results = response.json()["results"]
                    
                    if results:
                        st.success(f"Found {len(results)} similar images")
                        
                        # Display results in a grid
                        cols = st.columns(3)
                        for i, result in enumerate(results):
                            img = Image.open(result["image_path"])
                            cols[i % 3].image(
                                img, 
                                caption=f"Similarity: {result['similarity']:.2f}",
                                width=200
                            )
                    else:
                        st.warning("No similar images found")
                else:
                    print(response.text)
                    st.error("Error searching for similar images")

# Hacky stuff required to be added to make it work without `streamlit run`:
# See: https://github.com/streamlit/streamlit/issues/9450

if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        from streamlit.web.cli import main
        import sys
        sys.argv = ['streamlit', 'run', __file__]
        main()
    main()