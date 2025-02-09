from typing import List, Tuple
import gradio as gr
from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
import time
import json
import json

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Categories for each model
with open('categories.json', 'r', encoding='utf-8') as f1:
    categories = json.load(f1)

# Loading the Category Synopsis
with open('categories_synopsis.json', 'r', encoding='utf-8') as f2:
    categories_synopsis = json.load(f2)

# Loading the Parishes
with open('parishes.json', 'r', encoding='utf-8') as f3:
    parishes = json.load(f3)

# Default model
default_model = "Model v2"

# Model URLs
models = {
    "Model v1": YOLO("https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classify.pt").to(device),
    "Model v2": YOLO("https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classifyv2.pt").to(device),
    "Model v3 (Fast)": YOLO("https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classifyv3-Fast.pt").to(device),
    "Model v3 (Accurate)": YOLO("https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classifyv3-Accurate.pt").to(device)
}

parish_model_paths = {
    "Model v1": "https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classify-Parishv1.pt",
    "Model v2": "https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classify-Parishv2.pt"
}

# Loading the respective Parishes Model and Categories
parishes_model_path = "Model v2"
parishes_model = YOLO(parish_model_paths[parishes_model_path]).to(device)
parishes_categories = parishes[parishes_model_path]

def predict_image(image, model_name: str, size=(244, 244)) -> List[Tuple[str, str, float]]:
    """Predict the class of a given image and return sorted probabilities with categories."""
    if model_name is None:
        model_name = default_model
    
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_img = cv2.resize(image, size)
    resized_img = resized_img / 255.0  # Normalize
    resized_img = resized_img.transpose(2, 0, 1)  # Convert to (C, H, W)
    resized_img = resized_img[None, ...]  # Add batch dimension

    # Run prediction
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found.")

    results = model.predict(image)
    pred_probs = results[0].probs.data.cpu().numpy()

    # Sort predictions by probability
    sorted_indices = np.argsort(pred_probs)[::-1]  # Descending order
    english_categories = categories[model_name]["english"]
    maltese_categories = categories[model_name]["maltese"]
    sorted_predictions = [
        (
            english_categories[str(i)],
            maltese_categories[str(i)],
            round(pred_probs[i] * 100, 2)  # Convert to percentage
        )
        for i in sorted_indices
    ]

    return sorted_predictions

def predict_parish(image, size=(244, 244)) -> List[Tuple[str, float]]:
    """Predict the parish of a given image and return sorted probabilities with categories."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_img = cv2.resize(image, size)
    resized_img = resized_img / 255.0  # Normalize
    resized_img = resized_img.transpose(2, 0, 1)  # Convert to (C, H, W)
    resized_img = resized_img[None, ...]  # Add batch dimension

    # Run prediction
    results = parishes_model.predict(image)
    pred_probs = results[0].probs.data.cpu().numpy()

    # Sort predictions by probability
    sorted_indices = np.argsort(pred_probs)[::-1]  # Descending order
    sorted_predictions = [
        (
            parishes_categories[str(i)],
            round(pred_probs[i] * 100, 2)  # Convert to percentage
        )
        for i in sorted_indices
    ]

    return sorted_predictions

def classify_image(input_image, model_name):
    # Check if model_name is None
    if model_name is None:
        model_name = default_model

    start_time = time.time()

    # Get predictions from the model
    predictions = predict_image(input_image, model_name)

    # Predict the parish
    parish_predictions = predict_parish(input_image)
    
    # Format predictions into a dictionary with confidence scores
    formatted_predictions = {
        f"{label} / {maltese_label}": confidence / 100
        for label, maltese_label, confidence in predictions[:5]
    }

    # Format parish predictions into a dictionary with confidence scores
    formatted_parish_predictions = {
        f"{label}": confidence / 100
        for label, confidence in parish_predictions[:5]
    }

    # Modify the first formatted prediction to include "From the Parish of ..."
    first_label, first_confidence = parish_predictions[0]
    formatted_parish_predictions[f"From the Parish of / Mill-Parroċċa ta' {first_label}"] = formatted_parish_predictions.pop(first_label)

    # Get the label with the highest confidence
    highest_confidence_label = predictions[0][0]  # Assuming predictions are sorted by confidence
    highest_confidence_synopsis = categories_synopsis.get(highest_confidence_label, "No synopsis available.")

    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1.0 / elapsed_time

    return (
        formatted_predictions,
        formatted_parish_predictions,
        highest_confidence_synopsis,
        round(fps, 2)
    )

# Metadata
title = "Maltese Christian Statue Image Classification ✝"
description = (
    "This project aims to classify Maltese Christian statues and religious figures depicted in images. "
    "Choose a model to classify images into categories of Maltese Christian statues."
)
article = (
    "The YOLO classification models are trained on datasets of Maltese Christian statues and religious figures. "
    "The MCS Dataset is open-source and available for access through https://github.com/mbar0075/Maltese-Christian-Statue-Classifier.\n"
    "\n © Matthias Bartolo, Miriam Bartolo Abela 2025. Licensed under the MIT License."
)

# Load examples
example_folder = "examples"  # Single folder for all examples
examples = [[f"{example_folder}/{example}"] for example in os.listdir(example_folder) if example.endswith((".png", ".jpg", ".jpeg"))]

# For the list of examples, add the model name
for example in examples:
    example.append(default_model)

# Create the Gradio demo
demo = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil", label="Upload an image"),
        gr.Dropdown(
            choices=list(models.keys()),
            value=default_model,
            label="Select Model",
            interactive=True
        )
    ],
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions (English / Maltese)"),
        gr.Label(num_top_classes=5, label="Parish Predictions"),
        gr.Textbox(label="Synopsis / Aktar Tagħrif"),
        gr.Number(label="Prediction speed (FPS)")
    ],
    title=title,
    description=description,
    article=article,
    examples=examples
)

# Launch the demo
demo.launch()#share=True
