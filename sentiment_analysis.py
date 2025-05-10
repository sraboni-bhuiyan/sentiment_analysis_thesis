import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv('data/amazon_reviews_large.csv')
print(df.columns.tolist())
print(df.head())

""" # Dataset configuration
DATASET_CONFIG = {
    'sentiment140': {
        'path': 'data/sentiment140.csv',
        'columns': ['sentiment', 'id', 'date', 'query', 'user', 'text'],
        'load_args': {'encoding': 'latin-1', 'header': None},
        'rename_columns': {},
        'text_preprocessor': 'social_media',
        'sentiment_mapper': 'sentiment140'
    },
    'reddit': {
        'path': 'data/reddit_comments.csv',
        'columns': None,
        'load_args': {},
        'rename_columns': {'body': 'text', 'score': 'sentiment'},
        'text_preprocessor': 'social_media',
        'sentiment_mapper': 'reddit'
    },
    'amazon': {
        'path': 'data/amazon_reviews_large.csv',
        'columns': None,
        'load_args': {},
        'rename_columns': {'reviewText': 'text', 'rating': 'sentiment'},  # Updated
        'text_preprocessor': 'reviews',
        'sentiment_mapper': 'amazon'
    },
    'imdb': {
        'path': 'data/imdb.csv',
        'columns': None,
        'load_args': {},
        'rename_columns': {'review': 'text'},
        'text_preprocessor': 'reviews',
        'sentiment_mapper': 'imdb'
    }
}

# Text preprocessing functions
def preprocess_social_media(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    slang_dict = {
        'u': 'you', 'lol': 'laughing out loud', 'brb': 'be right back',
        'idk': 'i do not know', 'smh': 'shaking my head', 'thx': 'thanks'
    }
    words = text.split()
    text = ' '.join(slang_dict.get(word.lower(), word) for word in words)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def preprocess_reviews(text, max_length=512):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split()[:max_length])
    return text.strip()

# Sentiment mapping functions (binary classification)
def map_sentiment140(sentiment):
    return {0: 0, 4: 1}.get(sentiment, 0)

def map_reddit_sentiment(score):
    return 1 if score >= 3 else 0

def map_amazon_sentiment(rating):
    if pd.isna(rating):
        return 0
    rating = float(rating)
    return 1 if rating >= 4 else 0

def map_imdb_sentiment(sentiment):
    if isinstance(sentiment, str):
        return 1 if sentiment.lower() == 'positive' else 0
    return 1 if sentiment == 1 else 0

# Load datasets
def load_datasets():
    datasets = {}
    try:
        for name, config in DATASET_CONFIG.items():
            print(f"\nLoading {name} dataset...")
            df = pd.read_csv(config['path'], **config['load_args'])
            print(f"Raw columns in {name}: {df.columns.tolist()}")
            if config['columns']:
                df.columns = config['columns']
                print(f"Assigned columns in {name}: {df.columns.tolist()}")
            if config['rename_columns']:
                df = df.rename(columns=config['rename_columns'])
                print(f"Renamed columns in {name}: {df.columns.tolist()}")
            required_cols = ['text', 'sentiment']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns {missing_cols} in {name} dataset")
            datasets[name] = df
            print(f"Sample {name} data:")
            print(df[['text', 'sentiment']].head())
        return datasets
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)

# Preprocess datasets
def preprocess_datasets(datasets):
    try:
        text_preprocessors = {
            'social_media': preprocess_social_media,
            'reviews': preprocess_reviews
        }
        sentiment_mappers = {
            'sentiment140': map_sentiment140,
            'reddit': map_reddit_sentiment,
            'amazon': map_amazon_sentiment,
            'imdb': map_imdb_sentiment
        }
        splits = {}
        for name, df in datasets.items():
            df['text'] = df['text'].apply(text_preprocessors[DATASET_CONFIG[name]['text_preprocessor']])
            df['sentiment'] = df['sentiment'].apply(sentiment_mappers[DATASET_CONFIG[name]['sentiment_mapper']])
            print(f"\n{name} - Missing values:")
            print("Text:", df['text'].isnull().sum())
            print("Sentiment:", df['sentiment'].isnull().sum())
            df.dropna(subset=['text', 'sentiment'], inplace=True)
            class_counts = Counter(df['sentiment'])
            if class_counts:
                min_size = min(class_counts.values())
                df = pd.concat([
                    df[df['sentiment'] == label].sample(min_size, random_state=42)
                    for label in class_counts
                ])
            train, temp = train_test_split(df, test_size=0.2, stratify=df['sentiment'], random_state=42)
            val, test = train_test_split(temp, test_size=0.5, stratify=temp['sentiment'], random_state=42)
            splits[name] = (train, val, test)
            print(f"\n{name} - Sample preprocessed data:")
            print(df[['text', 'sentiment']].head())
            print(f"{name} - Class distribution:")
            print("Train:", Counter(train['sentiment']))
            print("Val:", Counter(val['sentiment']))
            print("Test:", Counter(test['sentiment']))
        return splits
    except Exception as e:
        print(f"Error preprocessing datasets: {e}")
        exit(1)

# Main execution
if __name__ == "__main__":
    datasets = load_datasets()
    splits = preprocess_datasets(datasets)
    for name, (train, val, test) in splits.items():
        train.to_csv(f'data/{name}_train.csv', index=False)
        val.to_csv(f'data/{name}_val.csv', index=False)
        test.to_csv(f'data/{name}_test.csv', index=False) """