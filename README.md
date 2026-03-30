# 🦠 Malaria Detection using Vision Transformer

##  Project Overview
This project detects malaria infection from microscopic blood cell images using a Vision Transformer (ViT) model.

##  Features
- Image classification (Parasitized / Uninfected)
- Confidence score
- Probability visualization
- Explainable AI (Grad-CAM heatmap)
- PDF report generation
- Web interface using Streamlit

##  Model
- Vision Transformer (ViT)
- Trained on NIH malaria dataset

## Dataset
NIH Malaria Dataset (27,558 images)

Download from:
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

##  Installation

```bash
pip install -r requirements.txt
```
Run Application
streamlit run malaria_app.py

Output
Prediction: Parasitized / Uninfected
Confidence score
Heatmap visualization
Downloadable PDF report

Mathematical Concepts Used
Softmax function
Cross Entropy Loss
Attention mechanism
GELU activation

Future Improvements
Multi-cell detection
Mobile app integration
Real-time microscope input

👨‍💻 Author
C.Mohith Reddy
