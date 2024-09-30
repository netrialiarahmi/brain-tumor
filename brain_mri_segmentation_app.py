
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

# Title
st.title('Brain MRI Segmentation App')

# Introduction
st.write('This app uses a segmentation model to perform brain MRI segmentation.')

# Upload MRI Image
uploaded_file = st.file_uploader("Choose a brain MRI image...", type="jpg")

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load the pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # Resize logits to the original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    segmentation = torch.argmax(upsampled_logits, dim=1).squeeze(0)

    # Plot segmentation result
    st.write("Segmentation Result")
    fig, ax = plt.subplots()
    ax.imshow(segmentation.cpu().numpy(), cmap='gray')
    st.pyplot(fig)

# Footer
st.write("Developed using the Segment Anything Model for brain MRI segmentation.")
