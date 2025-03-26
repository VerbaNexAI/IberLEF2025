import re
import string
import unicodedata

import emoji
import pandas as pd
from bs4 import BeautifulSoup
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class DatasetColumnTransformer:
    """
    A class for transforming dataset text columns by cleaning and preprocessing the text.
    """

    def __init__(self, language='spanish'):
        """
        Initialize the DatasetColumnTransformer with patterns and a lemmatizer.

        Args:
            language (str): Language used for stopwords (default is 'spanish').
                            Stopwords removal is commented out in this version.
        """
        # Initialize the WordNet lemmatizer.
        self.lemmatizer = WordNetLemmatizer()

        # Pre-compile regular expressions for efficiency.
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.user_pattern = re.compile(r'@\w+')
        self.html_pattern = re.compile(r'<.*?>')
        self.markdown_pattern = re.compile(r'\[.*?\]\(.*?\)')
        self.punctuation_pattern = re.compile(r'([!?.,]){7,}')
        self.double_quotes_pattern = re.compile(r'[“”]')
        self.single_quotes_pattern = re.compile(r'[‘’]')
        # Uncomment and initialize stopwords if needed:
        # self.stop_words = set(stopwords.words(language))

    def clean_text(self, text):
        """
        Clean and preprocess a given text string.

        The text is lowercased, and common patterns such as URLs, emails,
        user handles, HTML tags, markdown links, and excessive punctuation
        are replaced or removed. Additionally, emojis are converted to text,
        extra spaces are removed, quotes standardized, and words are tokenized
        and lemmatized.

        Args:
            text (str): The text string to be cleaned.

        Returns:
            str: The cleaned and preprocessed text.
        """
        # If the input is not a string, return an empty string.
        if not isinstance(text, str):
            return ""

        # Convert text to lowercase.
        text = text.lower()

        # Replace URLs, emails, and user mentions with placeholders.
        text = self.url_pattern.sub("[URL]", text)
        text = self.email_pattern.sub("[EMAIL]", text)
        text = self.user_pattern.sub("[USER]", text)

        # Remove HTML tags and markdown links.
        text = self.html_pattern.sub("", text)
        text = self.markdown_pattern.sub("", text)

        # Convert emojis to their text representation in Spanish.
        text = emoji.demojize(text, language='es')
        text = re.sub(r':(\w+):', r' \1 ', text)

        # Remove extra spaces.
        text = re.sub(r'\s+', ' ', text).strip()

        # Standardize double and single quotes.
        text = self.double_quotes_pattern.sub('"', text)
        text = self.single_quotes_pattern.sub("'", text)

        # Limit repeated punctuation to three consecutive occurrences.
        text = self.punctuation_pattern.sub(r'\1\1\1', text)

        # Tokenize the text into words.
        words = word_tokenize(text)

        # Optionally remove stopwords (commented out in this version).
        # words = [word for word in words if word not in self.stop_words]

        # Lemmatize each token.
        words = [self.lemmatizer.lemmatize(word) for word in words]

        # Rejoin tokens into a single string.
        return " ".join(words)

    def transform_column(self, df, column_name):
        """
        Apply the clean_text transformation to a specific column in a DataFrame.

        The method creates a copy of the DataFrame and applies the text cleaning
        function to the specified column.

        Args:
            df (pandas.DataFrame): The DataFrame containing the column to transform.
            column_name (str): The name of the column to clean.

        Returns:
            pandas.DataFrame: A copy of the DataFrame with the transformed column.
        """
        # Create a copy of the DataFrame to avoid modifying the original.
        df = df.copy()

        # Apply the clean_text function to the specified column.
        df.loc[:, column_name] = df[column_name].apply(self.clean_text)
        return df

