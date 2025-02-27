import streamlit as st
import torch
import torch.nn as nn
import gdown
import os
import pathlib
import platform
import random
import sys
from fastai.collab import load_learner

# Fix PosixPath issue on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Set up model download
@st.cache_resource
def download_model():
    """Download model from Google Drive with caching to avoid re-downloading each run"""
    model_url = 'https://drive.google.com/uc?id=1zMMcKOZF608BBqmokXFX-XvtOXM7E4o7'
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    
    # Only download if model doesn't exist
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... This may take a minute."):
            gdown.download(model_url, model_path, quiet=False)
    
    return str(model_path)

# Set page config
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🍿",
    layout="wide"
)

# Main application
def main():
    st.title("🎬 Anime Recommendation System")
    st.write("Select an anime and get recommendations for similar content!")
    
    try:
        # Get model path with download if needed
        model_path = download_model()
        
        # Load model
        with st.spinner("Loading model..."):
            learn = load_learner(model_path)
            dls = learn.dls
        
        # Initialize randomized anime list in session state if it doesn't exist
        if 'randomized_anime_list' not in st.session_state:
            anime_list = list(dls.classes['Anime Title'])
            random.shuffle(anime_list)
            st.session_state.randomized_anime_list = anime_list
        
        # Use the stored randomized list
        selected_anime = st.selectbox("Select an anime:", st.session_state.randomized_ani
