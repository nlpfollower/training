import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from transformers import pipeline, AutoTokenizer
from typing import List, Dict, Union
from src.types.conversation import Thread
from src.utils.logger import log
from config import get_config

nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    def __init__(self):
        config = get_config()
        self.sia = SentimentIntensityAnalyzer()
        self.roberta_model = pipeline("sentiment-analysis", model=config.sentiment.model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.sentiment.model)
        self.max_length = config.sentiment.max_length

    def preprocess(self, text: str) -> str:
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def truncate_text(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        return self.tokenizer.decode(tokens)

    def split_into_chunks(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.max_length - 2):  # -2 for [CLS] and [SEP] tokens
            chunk = tokens[i:i + self.max_length - 2]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk))
        return chunks

    def score(self, text: str) -> Dict[str, Union[float, str]]:
        nltk_scores = self.sia.polarity_scores(text)

        preprocessed_text = self.preprocess(text)
        chunks = self.split_into_chunks(preprocessed_text)

        roberta_scores = []
        for chunk in chunks:
            result = self.roberta_model(chunk)[0]
            roberta_scores.append((result['label'], result['score']))

        if roberta_scores:
            roberta_label, roberta_score = max(roberta_scores, key=lambda x: x[1])
        else:
            roberta_label, roberta_score = 'neutral', 0.0

        if roberta_label == 'negative':
            roberta_score = -roberta_score
        elif roberta_label == 'neutral':
            roberta_score = -roberta_score + 2 / 3

        roberta_score = max(-1, min(1, roberta_score))

        return {
            'nltk_compound': nltk_scores['compound'],
            'roberta_score': roberta_score,
            'roberta_label': roberta_label
        }

class SentimentStats:
    def __init__(self, threads: List[Thread]):
        self.threads = threads
        self.analyzer = SentimentAnalyzer()
        self.sentiment_scores = []

    def compute_sentiment_scores(self) -> List[Dict[str, float]]:
        log.info(f"Computing sentiment scores for {len(self.threads)} threads")
        self.sentiment_scores = []
        for thread in self.threads:
            for chat in thread.chats:
                if chat.is_assistant():
                    continue  # Skip assistant messages
                try:
                    self.sentiment_scores.append(self.analyzer.score(chat.message))
                except Exception as e:
                    log.error(f"Error processing message: {e}")
                    self.sentiment_scores.append({'nltk_compound': None, 'roberta_score': None, 'roberta_label': None})
        log.info(f"Computed sentiment scores for {len(self.sentiment_scores)} messages")
        return self.sentiment_scores

    def get_sentiment_distribution(self) -> Dict[str, List[int]]:
        nltk_scores = [score['nltk_compound'] for score in self.sentiment_scores if score['nltk_compound'] is not None]
        roberta_scores = [score['roberta_score'] for score in self.sentiment_scores if score['roberta_score'] is not None]

        bins = np.arange(-1.0, 1.1, 0.1)
        nltk_hist, _ = np.histogram(nltk_scores, bins=bins)
        roberta_hist, _ = np.histogram(roberta_scores, bins=bins)

        return {
            'nltk': nltk_hist.tolist(),
            'roberta': roberta_hist.tolist()
        }

    def get_sentiment_stats(self) -> Dict[str, Dict[str, float]]:
        nltk_scores = [score['nltk_compound'] for score in self.sentiment_scores if score['nltk_compound'] is not None]
        roberta_scores = [score['roberta_score'] for score in self.sentiment_scores if score['roberta_score'] is not None]

        return {
            'nltk': {
                'mean': np.mean(nltk_scores),
                'median': np.median(nltk_scores),
                'std': np.std(nltk_scores)
            },
            'roberta': {
                'mean': np.mean(roberta_scores),
                'median': np.median(roberta_scores),
                'std': np.std(roberta_scores)
            }
        }