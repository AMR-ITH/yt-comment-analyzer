from pathlib import Path
import pandas as pd
import re
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Logging setup (you can configure this globally if not already done)
logger = logging.getLogger("text_preprocessing")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)




def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment


def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)

        # Remove NaNs
        df.dropna(subset=['clean_comment'], inplace=True)
        
        # Remove empty strings
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: Path) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = data_path / 'interim'
        logger.debug(f"Creating directory {interim_data_path}")
        interim_data_path.mkdir(parents=True, exist_ok=True)

        train_data.to_csv(interim_data_path / "train_processed.csv", index=False)
        test_data.to_csv(interim_data_path / "test_processed.csv", index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Fetch the data from data/raw
        project_root = Path.cwd()
        raw_data_path = project_root / 'data' / 'raw'
        train_data = pd.read_csv(raw_data_path / 'train_data.csv')
        test_data = pd.read_csv(raw_data_path / 'test_data.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path=project_root / 'data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()





