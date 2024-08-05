from tqdm import tqdm
from typing import List
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import tensorflow as tf
from src.types.conversation import Thread
from src.utils.logger import log
from config import get_config


def jaccard_similarity(str1: str, str2: str) -> float:
    """Compute Jaccard similarity between two strings."""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def sequence_similarity(str1: str, str2: str) -> float:
    """Compute sequence similarity between two strings."""
    return SequenceMatcher(None, str1, str2).ratio()

class BERTSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def __call__(self, str1: str, str2: str) -> float:
        embeddings = self.model.encode([str1, str2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

class USESimilarity:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def __call__(self, str1: str, str2: str) -> float:
        with tf.device('/CPU:0'):  # Force CPU execution
            embeddings = self.model([str1, str2])
        return cosine_similarity([embeddings[0].numpy()], [embeddings[1].numpy()])[0][0]

class TFIDFSimilarity:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def __call__(self, str1: str, str2: str) -> float:
        try:
            tfidf_matrix = self.vectorizer.fit_transform([str1, str2])
            return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        except ValueError as e:
            if "empty vocabulary" in str(e):
                return 0.0  # Return 0 similarity for empty vocabulary
            raise  # Re-raise the exception if it's not the empty vocabulary error
        except NotFittedError:
            return 0.0


def compute_similarity_windows(threads: List[Thread], similarity_function) -> List[List[float]]:
    """
    Compute similarity windows for all threads.

    Args:
    threads (List[Thread]): List of Thread objects
    similarity_function: Function to compute similarity between two strings

    Returns:
    List[List[float]]: Similarity windows for all threads
    """
    config = get_config()
    log.info(f"Computing similarity windows for {len(threads)} threads")
    similarity_windows = []

    for thread in tqdm(threads, desc="Processing threads"):
        thread_window = []
        for i, current_chat in enumerate(thread.chats):
            window = []
            if current_chat.role != "assistant":
                thread_window.append(window)
                continue

            comparisons = 0
            for j in range(i+1, len(thread.chats)):
                if comparisons >= config.similarity.window_size:
                    break
                chat = thread.chats[j]
                if chat.role != "assistant":
                    window.append(0.0)
                else:
                    window.append(similarity_function(current_chat.message, chat.message))
                    comparisons += 1
            thread_window.append(window)
        similarity_windows.append(thread_window)

    log.info("Similarity window computation completed")
    return similarity_windows