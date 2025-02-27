import streamlit as st
import torch.nn as nn
import torch
import gdown
import os
import pathlib
import fastai
import platform
from fastai.collab import load_learner

# Fix PosixPath issue on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Download model from Google Drive
model_url = 'https://drive.google.com/uc?id=1zMMcKOZF608BBqmokXFX-XvtOXM7E4o7'
model_path = "model.pkl"
gdown.download(model_url, model_path, quiet=False)

# Ensure compatibility with Windows
model_path = os.path.abspath(model_path)

# Convert model path to string before loading
learn = load_learner(str(model_path))
dls = learn.dls

# Streamlit UI
st.title("Anime Recommendation System")

# Dropdown for user to select an anime
anime_list = dls.classes['Anime Title']
selected_anime = st.selectbox("Select an anime:", anime_list)

if st.button("Get Recommendations"):
    # Compute similarities
    anime_factors = learn.model.i_weight.weight
    idx = dls.classes['Anime Title'].o2i[selected_anime]
    distances = nn.CosineSimilarity(dim=1)(anime_factors, anime_factors[idx][None])
    idx = distances.argsort(descending=True)[1:11]
    recommendations = [dls.classes['Anime Title'][i] for i in idx]
    
    # Display recommendations
    st.write("### Top 10 Similar Anime:")
    for anime in recommendations:
        st.write(f"- {anime}")
