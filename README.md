# YouTube Comment Sentiment Analyzer
A full-stack machine learning application that analyzes YouTube comments in real-time to provide sentiment insights for content creators and social media influencers.

![Demo GIF](path/to/your/demo.gif)

## 🚀 Features

- **Real-time Sentiment Analysis**: Analyze YouTube comments instantly through Chrome extension
- **High Accuracy Model**: Logistic Regression with 87.98% accuracy and 0.8777 F1-score
- **Complete MLOps Pipeline**: Full ML lifecycle with experiment tracking and model versioning
- **Scalable Architecture**: Containerized Flask API with automated CI/CD deployment
- **Data Version Control**: Reproducible ML workflows with DVC and AWS S3 backend
- **Chrome Extension**: Seamless YouTube integration for real-time analysis

## 🏗️ System Architecture

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Chrome Extension│ │ Flask REST API  │ │ ML Pipeline     │
│ (Frontend UI)   │ │ (Backend)       │ │ (Scikit-learn)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ MLflow Registry │ │ DVC Pipeline    │ │ AWS S3 Storage  │
│ (Model Tracking)│ │ (Data Versioning│ │ (Artifacts)     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 🛠️ Technology Stack

### Machine Learning & MLOps
- **Scikit-learn**: Model training and inference
- **NLTK**: Natural language processing
- **MLflow**: Experiment tracking, model registry, and lifecycle management
- **DVC**: Data version control and pipeline orchestration
- **Optuna**: Hyperparameter optimization with Tree-structured Parzen Estimator

### Backend & Deployment
- **Flask**: RESTful API development
- **Docker**: Containerization for consistent deployments
- **AWS S3**: Cloud storage for datasets and model artifacts
- **GitHub Actions**: Automated CI/CD pipeline
- **Render**: Cloud deployment platform

### Frontend & Integration
- **Chrome Extension**: Browser integration for YouTube
- **JavaScript**: Frontend logic and DOM manipulation
- **HTML/CSS**: User interface design

