import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import logging

# logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        
        df = df.dropna()
        # Removing duplicates
        df =df.drop_duplicates()
        df = df.dropna(subset=['category', 'clean_comment'])
        # Removing rows with empty strings
        df = df[~(df['clean_comment'].str.strip() == '')]

        
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, train_path: str, test_path: str):
    """
    Save the training and testing data to CSV files.
    
    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        train_path (str): Path to save the training data.
        test_path (str): Path to save the testing data.
    """
    try:
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.debug(f"Data saved: Train data at {train_path}, Test data at {test_path}")
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise



def main():
    try:
        # Get the current working directory
        project_root = Path.cwd()
        logger.info(f"Current working directory: {project_root}")
        

        # Load parameters from params.yaml
        params = load_params(project_root / 'params.yaml')
        test_size = params['data_ingestion']['test_size']
        logger.info(f"Test size: {test_size}")
        print(f"Test size: {test_size}")
    
        # Load the dataset
        data_path = project_root / 'data' / 'raw' / 'reddit.csv'
        df = load_data(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        # Preprocess the data
        final_df = preprocess_data(df)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42, stratify=final_df['category'])

        # Save the processed data
        train_data_path = project_root / 'data' / 'raw' / 'train_data.csv'
        test_data_path = project_root / 'data' / 'raw' / 'test_data.csv'
        save_data(train_data, test_data, train_data_path, test_data_path)
        logger.info("Data ingestion completed successfully.")


    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")

if __name__ == "__main__":
    main()

