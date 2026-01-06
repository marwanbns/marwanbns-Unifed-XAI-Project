# Unified Explainable AI Interface
**Marwan BENNIS A5DIA2**

## Project Overview
This projects aim to refactor and integrates two existing Explainable AI systems into a single unifed one with an interactive interface, capable of handling multi-modal inputs :
1. Deepfake Audio Detection
2. Chest X-ray Analysis (Lung cancer detection)

The application allow the user to:
Upload an audio (.wav) or image (.png / .jpg) files
Select a compatible classification model
Apply one or multiple Explainable AI (XAI) techniques between (Lime, Shap and GradCam)
Compare explanations side by side in an interactive interface
The goal is not only to perform classification, but also to understand and visualize why models make their decisions.

## Installation 
```bash
git clone https://github.com/marwanbns/marwanbns-Unifed-XAI-Project
cd marwanbns-Unifed-XAI-Project
pip install -r requirements.txt
```

## Project Architecture
```bash
C:.
│   .gitignore
│   app.py
│   README.md
│   requirements.txt
│
├───configs
├───data
│   ├───cache
│   ├───outputs
│   └───uploads
├───input
│   ├───audio
│   │       2026-01-05 15-46-35.mp3
│   │       2026-01-05-15-46-35.wav
│   │       a0b5ebf906a67d59a3c99a4d29fe13040c6bff4cfc0899e93fc0418c.jpg
│   │       Donald-Trump_-4K-Original_Deep_Fake-Example-_mp3cut.net_.wav
│   │
│   └───image
│           view1_frontal.jpg
│           view2_frontal.jpg
│           view3_frontal.jpg
│           view4_frontal.jpg
│           view6_frontal.jpg
│
├───models
│   ├───audio
│   │   └───saved_model
│   │       └───model
│   │           │   keras_metadata.pb
│   │           │   saved_model.pb
│   │           │
│   │           └───variables
│   │                   variables.data-00000-of-00001
│   │                   variables.index
│   │
│   └───image
│           best_lunglesion_alexnet.pt
│           best_lunglesion_densenet121.pt
│
└───src
    ├───audio
    │   │   models.py
    │   │   preprocess.py
    │   │
    │   └───xai
    │           gradcam_audio.py
    │           lime_audio.py
    │           shap_audio.py
    │
    ├───common
    │       io.py
    │       registry.py
    │       utils.py
    │
    └───image
        │   models.py
        │   preprocess.py
        │
        └───xai
                gradcam_image.py
```

## Start the application


## How to Use the Interface
Open a command invite (cmd)
```bash
streamlit run app.py
```

1. Single Run Tab
- Upload an audio or image file
- The system automatically detects the input type
- Select a compatible model
- Select one XAI method
- Run inference and visualize the explanation

2. Comparison Tab
- Upload a single input
- Select a model
- Select multiple XAI techniques
- Compare explanations side by side
- Incompatible XAI methods are automatically hidden to avoid errors.