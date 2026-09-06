# MLForge

<p align="center">
  <b>Learn. Build. Train. Understand. — No Code Required.</b>
</p>

<p align="center">
  A web-based no-code Machine Learning learning platform for beginners.
</p>

<p align="center">

  <img src="https://img.shields.io/badge/Machine%20Learning-Educational-blue" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JavaScript-orange" alt="Frontend">
  <img src="https://img.shields.io/badge/Backend-FastAPI-green" alt="Backend">
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-yellow" alt="Machine Learning Framework">
  <img src="https://img.shields.io/badge/AI%20Assisted-IBM%20Bob-purple" alt="IBM Bob">

</p>

---

## 🌐 Live Demo

> 🚀 **Coming Soon**

<!-- Add your Vercel deployment URL here -->

<!-- Example:
🌐 Live Website: https://your-project.vercel.app
-->

---

# 📖 About MLForge

**MLForge** is a web-based **No-Code Machine Learning Learning Platform** designed to help beginners understand and experiment with Machine Learning without requiring them to write Machine Learning code manually.

The platform provides a visual workflow where users can:

* Select a dataset
* Configure classification classes
* Build a neural network architecture
* Configure hyperparameters
* Train a Machine Learning model
* Monitor training progress
* Visualize training results
* Evaluate model performance
* Perform predictions
* Export trained models

MLForge focuses on helping users understand the Machine Learning workflow rather than simply generating predictions.

The learning workflow is:

```text
Learn
  ↓
Choose Dataset
  ↓
Design Model
  ↓
Configure Training
  ↓
Train Model
  ↓
Evaluate Results
  ↓
Make Predictions
  ↓
Export Model
```

---

# 🎓 Capstone Project

This project was developed as a **Capstone Design Project** for:

## IBM SkillsBuild University Education × Hacktiv8

### Course

**IT-AI Agent for Programming**

MLForge was developed using an AI-assisted programming approach with **IBM Bob**.

IBM Bob was used to assist the development process, including:

* Project planning
* System architecture
* Code generation
* Frontend development
* Backend development
* Code debugging
* Code refinement
* Technical problem solving
* Iterative software development

> **Important:** IBM Bob is used as an AI Agent for Programming during the development of MLForge. IBM Bob is not integrated as a chatbot, API, backend service, or runtime feature inside the MLForge application.

The development workflow can be described as:

```text
Developer
    │
    ▼
IBM Bob
    │
    │ AI-Assisted Programming
    ▼
MLForge Source Code
    │
    ▼
MLForge Web Application
    │
    ▼
End User
```

---

# 🎯 Problem Statement

Learning Machine Learning can be challenging for beginners because it often requires knowledge of:

* Programming
* Python
* Machine Learning libraries
* TensorFlow
* Neural network architecture
* Dataset preprocessing
* Hyperparameter configuration
* Model training
* Model evaluation

As a result, beginners may focus more on copying existing code than understanding the Machine Learning workflow.

MLForge aims to address this problem by providing a visual and beginner-friendly environment where users can experiment with Machine Learning concepts without manually writing Machine Learning training code.

---

# 💡 Solution

MLForge provides a web-based visual interface that allows users to configure and experiment with Machine Learning models through a structured workflow.

Users can:

1. Select a dataset.
2. Select classification classes.
3. Build a neural network architecture.
4. Configure training parameters.
5. Train the model.
6. Monitor training progress.
7. Evaluate model performance.
8. Perform predictions.
9. Export the trained model.

The platform is designed to make Machine Learning experimentation more accessible to beginners and non-programmers.

---

# ✨ Features

## 🏠 Home

The Home page introduces:

* MLForge platform
* Platform objectives
* Machine Learning workflow
* Core features
* Educational purpose

---

## 📊 Dataset Selection

Users can select datasets for Machine Learning experiments.

The current implementation focuses on:

### MNIST Dataset

MLForge supports image classification using the MNIST dataset.

Users can configure classification as:

### Binary Classification

Example:

```text
Class 0 vs Class 1
```

or:

```text
Class 3 vs Class 8
```

### Multiclass Classification

Example:

```text
Class 0
Class 1
Class 2
Class 3
Class 4
...
```

The number of output classes is determined based on the user's dataset configuration.

---

## 🧠 No-Code Model Builder

MLForge provides a visual interface for building neural network architectures.

Users can configure layers without writing TensorFlow or Keras code manually.

Supported layers include:

* Conv2D
* MaxPooling2D
* AveragePooling2D
* Flatten
* Dense
* Dropout
* BatchNormalization

Users can:

* Add layers
* Remove layers
* Configure layer parameters
* Arrange neural network architectures
* Customize model configuration

Example architecture:

```text
Input
  │
  ▼
Conv2D
  │
  ▼
MaxPooling2D
  │
  ▼
Conv2D
  │
  ▼
Flatten
  │
  ▼
Dense
  │
  ▼
Output
```

The frontend sends the model configuration to the backend using structured data.

Example:

```json
{
  "layers": [
    {
      "type": "Conv2D",
      "filters": 32,
      "kernel_size": 3,
      "activation": "relu"
    },
    {
      "type": "MaxPooling2D",
      "pool_size": 2
    },
    {
      "type": "Flatten"
    },
    {
      "type": "Dense",
      "units": 128,
      "activation": "relu"
    }
  ]
}
```

---

## ⚙️ Training Configuration

Users can configure Machine Learning hyperparameters.

Available configurations include:

* Epochs
* Batch Size
* Learning Rate
* Optimizer
* Validation Split

Supported optimizers include:

* Adam
* SGD
* RMSprop

Supported activation functions include:

* ReLU
* Sigmoid
* Tanh

---

## 🚀 Model Training

MLForge provides a web-based Machine Learning training workflow.

The frontend sends:

* Dataset configuration
* Selected classes
* Model architecture
* Layer configuration
* Hyperparameters

to the backend.

The backend handles:

```text
Load Dataset
      ↓
Validate Configuration
      ↓
Build Model
      ↓
Compile Model
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Save Results
```

Machine Learning operations are handled by the backend rather than requiring users to write or run Machine Learning code manually.

---

## 📈 Training Progress

MLForge provides training information during the Machine Learning workflow.

Training information may include:

* Current Epoch
* Total Epoch
* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

Training status includes:

```text
Queued
   ↓
Training
   ↓
Completed
```

or:

```text
Failed
```

---

## 📉 Training Visualization

MLForge provides visualization of training performance.

### Accuracy Curve

Displays:

* Training Accuracy
* Validation Accuracy

### Loss Curve

Displays:

* Training Loss
* Validation Loss

The visualization helps users understand:

* Learning progress
* Overfitting
* Underfitting
* Training stability
* Validation performance

---

## 📊 Model Evaluation

MLForge provides Machine Learning evaluation results.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score

The platform can also display:

* Confusion Matrix

These metrics help users understand how well their Machine Learning model performs.

---

## 🔍 Prediction

Users can test trained models through image prediction.

The workflow is:

```text
Upload Image
      ↓
Image Processing
      ↓
Model Inference
      ↓
Prediction
      ↓
Result
```

Prediction results may include:

* Predicted Class
* Confidence Score
* Class Probabilities

---

## 📦 Model Export

MLForge supports exporting trained Machine Learning models.

Supported formats include:

* `.keras`
* `.tflite`

Additional exported information may include:

* Model Architecture
* Training History
* Evaluation Results

TensorFlow Lite export provides a possible path for deploying trained models to edge and embedded devices.

---

# 🏗️ System Architecture

MLForge uses a frontend and backend architecture.

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │    Frontend     │
                  │                 │
                  │ HTML            │
                  │ CSS             │
                  │ JavaScript      │
                  └────────┬────────┘
                           │
                           │ API Request
                           ▼
                  ┌─────────────────┐
                  │    Backend      │
                  │                 │
                  │ Python          │
                  │ FastAPI         │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  ML Components  │
                  │                 │
                  │ Model Builder   │
                  │ Dataset Loader  │
                  │ Trainer         │
                  │ Evaluator       │
                  │ Exporter        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ TensorFlow      │
                  │ Keras           │
                  └─────────────────┘
```

---

# 🧩 MLForge Workflow

The MLForge workflow is designed as a step-by-step Machine Learning learning experience.

```text
┌─────────────────┐
│   1. Dataset    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Build Model │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Configure    │
│    Training     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   4. Training   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Evaluation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Prediction  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    7. Export    │
└─────────────────┘
```

---

# 📁 Project Structure

```text
MLForge/
│
├── frontend/
│   │
│   ├── css/
│   │   └── main.css
│   │
│   ├── js/
│   │   ├── dataset.js
│   │   ├── evaluation.js
│   │   ├── export.js
│   │   ├── mlforge.js
│   │   ├── model-builder.js
│   │   ├── prediction.js
│   │   └── training.js
│   │
│   ├── dataset.html
│   ├── evaluation.html
│   ├── export.html
│   ├── index.html
│   ├── model-builder.html
│   ├── prediction.html
│   └── training.html
│
├── backend/
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── evaluation.py
│   │   ├── export.py
│   │   ├── prediction.py
│   │   └── training.py
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   ├── exporter.py
│   │   └── trainer.py
│   │
│   ├── models/
│   │
│   └── main.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🛠️ Technology Stack

## Frontend

* HTML5
* CSS3
* JavaScript

## Backend

* Python
* FastAPI

## Machine Learning

* TensorFlow
* Keras

## Development

* Visual Studio Code
* Git
* GitHub
* IBM Bob

## Deployment

* Vercel

---

# 🤖 AI-Assisted Development with IBM Bob

MLForge was developed using an AI-assisted programming workflow.

**IBM Bob** was used as an AI Agent for Programming to support the development process.

The development workflow included:

```text
Project Idea
      │
      ▼
Requirement Definition
      │
      ▼
System Architecture
      │
      ▼
Prompt Engineering
      │
      ▼
IBM Bob
      │
      ▼
Code Generation
      │
      ▼
Code Review
      │
      ▼
Code Refinement
      │
      ▼
Testing
      │
      ▼
MLForge
```

The project followed an iterative development approach.

The development cycle included:

```text
Identify Problem
       │
       ▼
Define Requirement
       │
       ▼
Generate / Modify Code
       │
       ▼
Test Implementation
       │
       ▼
Evaluate Result
       │
       ▼
Refine
```

> IBM Bob was used to assist the software development process. The final MLForge application does not require IBM Bob to run and does not include IBM Bob as a user-facing feature.

---

# 🎓 Educational Objectives

MLForge is designed to help beginners understand the Machine Learning development process.

Users can learn about:

* Dataset selection
* Binary classification
* Multiclass classification
* Neural networks
* Convolutional Neural Networks
* Conv2D layers
* Pooling layers
* Dense layers
* Dropout
* Batch normalization
* Hyperparameters
* Optimizers
* Epochs
* Batch size
* Learning rate
* Training curves
* Model evaluation
* Model prediction
* Model export

The platform emphasizes experimentation and understanding.

---

# ⚠️ Responsible AI

MLForge is designed for educational and experimental purposes.

Users should understand that:

* Model performance depends on the dataset.
* Model performance depends on the selected architecture.
* Hyperparameters affect training results.
* High accuracy does not guarantee universal reliability.
* Predictions should not automatically be interpreted as certainty.
* Models should not be used for high-stakes decisions without proper validation.
* Sensitive personal information should not be unnecessarily uploaded.

MLForge is not intended to replace professional Machine Learning validation or production-grade model development.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/MLForge.git
```

Move into the project directory:

```bash
cd MLForge
```

---

# 💻 Frontend

The frontend is located inside:

```text
frontend/
```

The frontend uses:

* HTML
* CSS
* JavaScript

Open the project using a local development server.

For example, using VS Code:

```text
Open frontend/index.html
```

You can also use a local development extension such as Live Server.

---

# ⚙️ Backend Setup

Move to the backend directory:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Run the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

---

# 🌐 Deployment

The MLForge frontend can be deployed using **Vercel**.

The deployment architecture is:

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │     Vercel      │
                  │                 │
                  │    Frontend     │
                  │ HTML / CSS / JS │
                  └────────┬────────┘
                           │
                           │ API
                           ▼
                  ┌─────────────────┐
                  │     Backend     │
                  │                 │
                  │    FastAPI      │
                  │    Python       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Machine Learning│
                  │ TensorFlow/Keras│
                  └─────────────────┘
```

## Frontend Deployment

1. Push the project to GitHub.
2. Create a Vercel account.
3. Import the MLForge GitHub repository.
4. Configure the Root Directory as:

```text
frontend
```

5. Deploy the project.

---

# 📌 Project Status

**Status: Active Development**

MLForge is currently developed as:

* Educational Machine Learning Platform
* No-Code Machine Learning Tool
* Capstone Design Project
* AI-Assisted Programming Project

---

# 🔮 Future Development

Future improvements may include:

* Fashion-MNIST support
* CIFAR-10 support
* Additional datasets
* Custom dataset upload
* More neural network layers
* Drag-and-drop model builder
* Model architecture visualization
* Model comparison
* Training history management
* Model versioning
* Advanced hyperparameter configuration
* Hyperparameter optimization
* Training queue management
* Cloud model storage
* ONNX export
* Advanced TensorFlow Lite export
* TinyML deployment workflow
* Edge AI integration

---

# 📜 License

This project is currently developed for educational, portfolio, and Capstone Design purposes.

A specific open-source license may be added in the future.

---

# 👨‍💻 Author

Developed by:

**[Your Name]**

Biomedical Engineering Graduate
Machine Learning and Computer Vision Enthusiast

---

# 🙏 Acknowledgment

This project was developed as part of:

## IBM SkillsBuild University Education × Hacktiv8

### Course

**IT-AI Agent for Programming**

Special acknowledgment to:

**IBM Bob**

for supporting the AI-assisted programming and iterative development process used in the development of MLForge.

---

# ⭐ Support

If you find this project interesting:

⭐ Star the repository
🐛 Report issues
💡 Suggest improvements

---

<p align="center">

## MLForge

### Learn. Build. Train. Understand.

**No Code Required.**

</p>
