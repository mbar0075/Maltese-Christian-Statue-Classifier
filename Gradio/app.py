from typing import List, Tuple
import gradio as gr
from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Categories in English and Maltese
categories_english = {
    0: 'Jesus has Risen', 1: 'Jesus praying in Gethsemane', 2: 'Saint Philip of Agira',
    3: 'Simon of Cyrene', 4: 'The Betrayal of Judas', 5: 'The Cross',
    6: 'The Crucifixion', 7: 'The Ecce Homo', 8: 'The Flogged',
    9: 'The Lady of Sorrows', 10: 'The Last Supper', 11: 'The Monument',
    12: 'The Redeemer', 13: 'The Veronica'
}

categories_maltese = {
    0: 'L-Irxoxt', 1: 'Ġesù fl-Ort tal-Ġetsemani', 2: 'San Filep ta’ Aġġira',
    3: 'Xmun min Ċireni', 4: 'It-Tradiment ta’ Ġuda', 5: 'Is-Salib',
    6: 'Il-Vara Il-Kbira', 7: 'L-Ecce Homo', 8: 'Il-Marbut',
    9: 'Id-Duluri', 10: 'L-Aħħar Ċena', 11: 'Il-Monument',
    12: 'Ir-Redentur', 13: 'Il-Veronica'
}

def predict_image(image, model, size=(244, 244)) -> List[Tuple[str, str, float]]:
    """Predict the class of a given image and return sorted probabilities with categories."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_img = cv2.resize(image, size)
    resized_img = resized_img / 255.0  # Normalize
    resized_img = resized_img.transpose(2, 0, 1)  # Convert to (C, H, W)
    resized_img = resized_img[None, ...]  # Add batch dimension

    # Run prediction
    results = model.predict(image)
    pred_probs = results[0].probs.data.cpu().numpy()

    # Sort predictions by probability
    sorted_indices = np.argsort(pred_probs)[::-1]  # Descending order
    sorted_predictions = [
        (
            categories_english[i],
            categories_maltese[i],
            round(pred_probs[i] * 100, 2)  # Convert to percentage
        )
        for i in sorted_indices
    ]

    return sorted_predictions

# Load the model
model = YOLO("https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classify.pt").to(device)

example_list = [["examples/" + example] for example in os.listdir("examples")]

def classify_image(input_image):
    predictions = predict_image(input_image, model)
    formatted_predictions = {
        f"{label} / {maltese_label}": confidence / 100
        for label, maltese_label, confidence in predictions[:5]
    }
    return formatted_predictions, round(1.0 / 0.1, 2)  # Example FPS calculation

# Metadata
title = "Maltese Christian Statue Image Classification ✝"
description = (
    "This project aims to classify Maltese Christian statues and religious figures depicted in images. The model is trained "
    "to recognise 14 unique categories related to the Maltese Christian heritage, presented in both English and Maltese. "
    "Upload an image to receive predictions of the statue's category, sorted by likelihood. This initiative promotes the "
    "preservation of Maltese culture and traditions through AI technology."
)
article = (
    "The YOLO11m classification model was trained on a dataset of Maltese Christian statues and religious figures during the period of Lent.\n"
    "The MCS Dataset is open-source and available for access through https://github.com/mbar0075/Maltese-Christian-Statue-Classifier.\n"
    "Matthias Bartolo 2024-2025"
)

# Create the Gradio demo
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions (English / Maltese)"),
        gr.Number(label="Prediction speed (FPS)")
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

# Launch the demo
demo.launch()
