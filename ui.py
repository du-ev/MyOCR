import streamlit as st
import os
import dotenv
import numpy as np
from textgen import generate_summary, generate_response

import torch
import glob, re
import cv2
from crnn import CRNN
from preprocess_datasets import load_image, convert_image
from train import ctc_decode, load_ocr_model, blank_idx, num_classes, ctc_decode_beam


# how to run: 
#   install README requirements.
#   in terminal, run "streamlit run ui.py". a local server should pop up in your browser.
#   if code changes in ui.py, the server will automatically refresh to update those changes to your browser.
dotenv.load_dotenv()

def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes).to(device).eval()
    load_ocr_model(model, "checkpoints", force=True)

    return model, device

model, device = get_model()

allowed_types = ['jpg', 'png', 'jpeg'] # these are the only file types allowed.

st.title("Text De-TEXT-ion")

uploaded_img = st.file_uploader("Upload an Image:", type=allowed_types) # take in image from the UI

if uploaded_img is not None:
    os.makedirs("uploads", exist_ok=True) # folder named 'uploads' for storing images uploaded in streamlit
    save_path = os.path.join("uploads", "input.png") # this will save the image as a path within your 'uploads' folder
    
    with open(save_path, 'wb') as f: # this will write the image bytes to your file
        f.write(uploaded_img.read())
    
    #Display image
    st.image(save_path, caption="Uploaded Image", use_container_width=True)

    arr = load_image(save_path)
    cleaned_img = convert_image(arr)

    st.image(cleaned_img, use_container_width=True, clamp=True)

    h, w = cleaned_img.shape[:2]
    padding = max(5, w // 20)
    print(h, w)
    st.image(cleaned_img, use_column_width=True, clamp=True)
    cleaned_img = cv2.copyMakeBorder(cleaned_img, 0, 0, padding, padding, borderType=cv2.BORDER_CONSTANT, value=0)

    img_tensor = torch.from_numpy(cleaned_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)

    text = ctc_decode(logits, blank_idx)
    # texts = text = ctc_decode_beam(logits, beam_width=10)
    # text = texts[0]

    #Display text detected
    st.subheader("Extracted Text")
    st.write(text)

    #Display summary generated
    definition = generate_response(text, instructions = "define the following word (The user is a little bit stupid and may misspell word. do your best to correct it):", model_name = "gemini-2.0-flash")
    st.subheader("Definition: ")
    st.write(definition)

