from typing import List, Tuple
from src.types.datasets import KTOSample, KTOSampleStats
from src.types.conversation import Thread
from src.utils.sentiment import SentimentAnalyzer
from src.utils.logger import log
from tqdm import tqdm

class KTOTrainer:
    def __init__(self, config):
        self.config = config

class KTOGenerator:
    def __init__(self, threads: List[Thread]):
        self.threads = threads
        self.sentiment_analyzer = SentimentAnalyzer()

    def generate_dataset(self) -> Tuple[List[KTOSample], List[KTOSampleStats]]:
        log.info("Starting KTO dataset generation pipeline")

        kto_sample_stats = self._generate_kto_sample_stats()
        log.info(f"Generated {len(kto_sample_stats)} KTO sample stats")

        kto_samples = self._convert_to_kto_samples(kto_sample_stats)
        log.info(f"Generated {len(kto_samples)} final KTO samples")

        return kto_samples, kto_sample_stats

    def _generate_kto_sample_stats(self) -> List[KTOSampleStats]:
        kto_sample_stats = []
        for thread in tqdm(self.threads, desc="Generating KTO samples", unit="thread"):
            partial_context = []
            for i, chat in enumerate(thread.chats[:-1]):  # Exclude last chat
                if chat.role == "assistant":
                    if i + 1 < len(thread.chats) and thread.chats[i + 1].role == "user":
                        final_context = thread.chats[i + 1].message
                        sentiment = self.sentiment_analyzer.score(final_context)
                        kto_sample_stats.append(KTOSampleStats(
                            partial_context=partial_context.copy(),
                            chat=chat.message,
                            final_context=final_context,
                            sentiment=sentiment
                        ))
                elif chat.role == "user":
                    partial_context.append(chat.message)
        return kto_sample_stats

    def _convert_to_kto_samples(self, kto_sample_stats: List[KTOSampleStats]) -> List[KTOSample]:
        return [
            KTOSample(
                prompt="\n".join(stats.partial_context),
                chat=stats.chat,
                sentiment=stats.sentiment
            )
            for stats in kto_sample_stats
        ]