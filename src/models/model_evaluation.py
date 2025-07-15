from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

import dagshub

# Logging setup
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    
    Args:
        params_path (str): Path to the YAML file.
        
    Returns:
        dict: Loaded parameters.
    """
    try: 
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded from {params_path}: {params}")
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise

def load_model(model_path: str):
    """Load the saved model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except pd.errors.EmptyDataError:
        logger.error('No data found in file: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': str(model_path)  # Convert Path to string
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm, accuracy
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save confusion matrix plot as a file and log it to MLflow
        cm_file_path = f'confusion_matrix_{dataset_name.replace(" ", "_")}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
        
        # Clean up the file
        if os.path.exists(cm_file_path):
            os.remove(cm_file_path)
            
    except Exception as e:
        logger.error('Error logging confusion matrix: %s', e)
        raise

def main():
    logger.info("Starting model evaluation...")
    
    try:
        # Initialize DagsHub
        logger.info("Initializing DagsHub...")
        dagshub.init(repo_owner='AMR-ITH', repo_name='yt-comment-analyzer', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/AMR-ITH/yt-comment-analyzer.mlflow/")
        mlflow.set_experiment("yt_comment_analyzer_experiment")
        
        logger.info("Starting MLflow run...")
        with mlflow.start_run() as run:
            logger.info(f"MLflow run started with ID: {run.info.run_id}")
            
            # Get project root
            project_root = Path.cwd()
            logger.info(f"Project root: {project_root}")
            print(f"Project root: {project_root}")
            # Load parameters
            logger.info("Loading parameters...")
            params = load_params(project_root / 'params.yaml')
            
            # Log parameters
            logger.info("Logging parameters to MLflow...")
            for key, value in params.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_param(f"{key}_{sub_key}", sub_value)
                else:
                    mlflow.log_param(key, value)
            
            # Load model and vectorizer
            logger.info("Loading model and vectorizer...")
            model_path = project_root / 'models' / 'logistic_model.pkl'
            vectorizer_path = project_root / 'models' / 'tfidf_vectorizer.pkl'
            
            model = load_model(model_path)
            vectorizer = load_vectorizer(vectorizer_path)
            
            # Load test data
            logger.info("Loading test data...")
            test_data_path = project_root / 'data' / 'interim' / 'test_processed.csv'
            
            if not test_data_path.exists():
                raise FileNotFoundError(f"Test data file not found: {test_data_path}")
            
            test_data = load_data(test_data_path)
            
            # Prepare the test data
            logger.info("Preparing test data...")
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values
            
            logger.info(f"Test data shape: {X_test_tfidf.shape}")
            logger.info(f"Test labels shape: {y_test.shape}")
            
            # Log model as artifact (DagsHub compatible)
            logger.info("Logging model as artifact...")
            mlflow.log_artifact(str(model_path), "model")
            
            # Save model info
            logger.info("Saving model info...")
            model_info_path = project_root / 'models' / 'logistic_model'
            save_model_info(run.info.run_id, model_info_path, 'experiment_info.json')
            
            # Log the vectorizer as an artifact
            logger.info("Logging vectorizer as artifact...")
            mlflow.log_artifact(str(vectorizer_path), "model")
            
            # Evaluate model and get metrics
            logger.info("Evaluating model...")
            report, cm, accuracy = evaluate_model(model, X_test_tfidf, y_test)
            
            # Debug: Print what's in the classification report
            logger.info("Classification report keys:")
            for key in report.keys():
                logger.info(f"  - {key}: {type(report[key])}")
                if key == 'accuracy':
                    logger.info(f"    Accuracy value: {report[key]}")
            
            # Log accuracy metric (calculated separately)
            logger.info(f"Test accuracy: {accuracy:.4f}")
            mlflow.log_metric("test_accuracy", accuracy)
            
            # Log classification report metrics for the test data
            logger.info("Logging classification metrics to MLflow...")
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })
            
            # Log overall metrics from the report
            if 'macro avg' in report:
                mlflow.log_metrics({
                    "test_macro_avg_precision": report['macro avg']['precision'],
                    "test_macro_avg_recall": report['macro avg']['recall'],
                    "test_macro_avg_f1-score": report['macro avg']['f1-score']
                })
            
            if 'weighted avg' in report:
                mlflow.log_metrics({
                    "test_weighted_avg_precision": report['weighted avg']['precision'],
                    "test_weighted_avg_recall": report['weighted avg']['recall'],
                    "test_weighted_avg_f1-score": report['weighted avg']['f1-score']
                })
            
            # Log accuracy from classification report if available
            if 'accuracy' in report:
                logger.info(f"Classification report accuracy: {report['accuracy']:.4f}")
                mlflow.log_metric("test_accuracy_from_report", report['accuracy'])
            
            # Log confusion matrix
            logger.info("Logging confusion matrix...")
            log_confusion_matrix(cm, "Test_Data")
            
            # Add important tags
            logger.info("Adding tags...")
            mlflow.set_tag("model_type", "Logistic Regression")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            
            logger.info("Model evaluation completed successfully!")
            
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()