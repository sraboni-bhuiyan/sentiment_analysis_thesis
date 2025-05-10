import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv('data/amazon_reviews_large.csv')
print(df.columns.tolist())