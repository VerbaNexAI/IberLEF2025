import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator

# Download necessary NLTK resources (only required on first execution)
nltk.download('wordnet')
nltk.download('omw-1.4')


class DataAugmenter:
    """
    A class to apply data augmentation techniques to Spanish text to balance class distributions.

    Implemented techniques:
        - Back translation: Spanish to English and back to Spanish using deep_translator.
        - Synonym replacement using WordNet.
        - Paraphrasing by combining the above techniques.

    This class allows specifying the label to be balanced (either numeric or textual)
    and the target number of samples to achieve balance.
    """

    def __init__(self, seed=42):
        """
        Initializes the DataAugmenter class.

        Args:
            seed (int): Seed value for reproducibility. Default is 42.
        """
        self.seed = seed
        random.seed(self.seed)

    def back_translate(self, text, src='es', mid='en'):
        """
        Applies back translation: translates the text from Spanish to an intermediate language
        (default is English) and then translates it back to Spanish.

        Args:
            text (str): The input text to translate.
            src (str): Source language (default: 'es' for Spanish).
            mid (str): Intermediate language (default: 'en' for English).

        Returns:
            str: The back-translated text.
        """
        try:
            translated = GoogleTranslator(source=src, target=mid).translate(text)
            back_translated = GoogleTranslator(source=mid, target=src).translate(translated)
            return back_translated
        except Exception as e:
            print(f"Error in back_translate: {e}")
            return text

    def get_synonyms(self, word):
        """
        Retrieves a set of synonyms for a given Spanish word using WordNet.

        Args:
            word (str): The input word.

        Returns:
            set: A set of synonyms for the given word.
        """
        synonyms = set()
        for syn in wordnet.synsets(word, lang='spa'):
            for syn_word in syn.lemma_names('spa'):
                if syn_word.lower() != word.lower():
                    synonyms.add(syn_word)
        return synonyms

    def synonym_substitution(self, text, n=1):
        """
        Randomly replaces n words in the text with one of their synonyms (in Spanish).

        Args:
            text (str): The input text.
            n (int): Number of words to replace. Default is 1.

        Returns:
            str: The text with synonyms replaced.
        """
        words = text.split()
        new_words = words.copy()
        indices = list(range(len(words)))
        random.shuffle(indices)
        num_replaced = 0
        for i in indices:
            word = words[i]
            synonyms = self.get_synonyms(word)
            if synonyms:
                new_word = random.choice(list(synonyms))
                new_words[i] = new_word
                num_replaced += 1
            if num_replaced >= n:
                break
        return ' '.join(new_words)

    def paraphrase(self, text):
        """
        Generates a paraphrased version of the text by combining techniques:
        first applying back translation and then synonym substitution.

        Args:
            text (str): The input text.

        Returns:
            str: The paraphrased text.
        """
        paraphrased_text = self.back_translate(text)
        paraphrased_text = self.synonym_substitution(paraphrased_text, n=1)
        return paraphrased_text

    def augment_text(self, text):
        """
        Generates multiple augmented versions of the original text using different techniques.

        Args:
            text (str): The input text.

        Returns:
            list: A list of unique augmented texts.
        """
        augmented_texts = [text]
        augmented_texts.append(self.back_translate(text))
        augmented_texts.append(self.synonym_substitution(text, n=1))
        augmented_texts.append(self.paraphrase(text))
        return list(set(augmented_texts))

    def balance_dataset(self, df, text_col='message', label_col='label', target_labels=None,
                        balance_by='label', target_count=None):
        """
        Balances the dataset by generating augmented examples for minority classes.

        Args:
            df (pd.DataFrame): The input dataset.
            text_col (str): Column containing the text.
            label_col (str): Column containing the numerical labels.
            target_labels (list or int, optional): Labels to balance. Can be a single value or a list.
            balance_by (str): Column used to identify the label to balance ('label' or 'label_text').
            target_count (int, optional): The number of samples to balance up to. If not specified,
                                         it balances up to the maximum sample count in the dataset.

        Returns:
            pd.DataFrame: A new balanced dataset. Classes not specified in target_labels remain unchanged.
        """
        if target_labels is not None and not isinstance(target_labels, list):
            target_labels = [target_labels]

        class_counts = df['label_text'].value_counts() if balance_by == 'label_text' else df[label_col].value_counts()
        default_target = class_counts.max() if target_count is None else target_count
        augmented_rows = []

        for current_label, count in class_counts.items():
            if target_labels is not None and current_label not in target_labels:
                continue
            if count >= default_target:
                continue

            df_class = df[df['label_text'] == current_label] if balance_by == 'label_text' else df[
                df[label_col] == current_label]
            texts = df_class[text_col].tolist()
            num_needed = default_target - count

            for _ in range(num_needed):
                sample = random.choice(texts)
                augmented_texts = self.augment_text(sample)
                candidate = sample
                for variant in augmented_texts:
                    if variant != sample:
                        candidate = variant
                        break
                new_row = df_class.iloc[0].copy()
                new_row[text_col] = candidate
                augmented_rows.append(new_row)

        df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True) if augmented_rows else df.copy()
        return df_augmented


