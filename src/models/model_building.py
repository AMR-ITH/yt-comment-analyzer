import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import yaml
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



# Logging setup 
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_params(params_path:str)-> dict:
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


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        # train_data.dropna(subset=['clean_comment'], inplace=True)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        # Save the vectorizer in the root directory
        project_root = Path.cwd()
        with open(project_root / "models" / "tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def train_logistic(X_train_tfidf : np.ndarray, y_train: np.ndarray, penalty:str, C:float, solver:str, max_iter :int) -> LogisticRegression:
    try:
        model = LogisticRegression(penalty=penalty, C=C, 
                                   solver=solver,
                                     max_iter=max_iter,
                                     class_weight ="balanced",
                                     multi_class= "auto",
                                     random_state=42)
        model.fit(X_train_tfidf, y_train)
        logger.debug('Logistic Regression model trained successfully')
        return model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise

def save_model(model: LogisticRegression, model_path: str) -> None:
    """
    Save the trained model to a file.
    
    Args:
        model (LogisticRegression): The trained model.
        model_path (str): Path to save the model.
    """
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved to {model_path}")
    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise

def main():
    try:
        # Fetch the data from data/raw
        project_root = Path.cwd()

        # Load parameters from params.yaml
        params = load_params(project_root / 'params.yaml')
        ngram_range = tuple(params["model_building"]["ngram_range"])
        max_features = params['model_building']['max_features']
        penalty = params['model_building']['penalty']
        C = params['model_building']['C']
        solver = params['model_building']['solver']
        max_iter = params['model_building']['max_iter']

        # load the data
        train_data = load_data(project_root / 'data' / 'interim' / 'train_processed.csv')

        # Apply TF-IDF
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_logistic(X_train_tfidf, y_train, penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        # Save the trained model
        model_path = project_root / 'models' / 'logistic_model.pkl'
        save_model(best_model, model_path)

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()



