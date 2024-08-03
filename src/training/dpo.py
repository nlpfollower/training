import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
from src.types.conversation import Thread, Chat
from src.types.datasets import DPOBatch, DPOSample, DPOSampleStats
from src.training.mock import Trainer
from src.utils.similarity import jaccard_similarity, sequence_similarity, BERTSimilarity, USESimilarity, TFIDFSimilarity, compute_similarity_windows
from src.utils.sentiment import SentimentAnalyzer
from src.utils.llm_judge import GPTJudge
from src.utils.logger import log
from src.model.llama_model import LlamaModel
from config.config import Config, get_config

def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

class DPOTrainer(Trainer):
    def __init__(self, config):
        self.config = config
        self.beta = config.training.beta

    def move_batch_to_device(self, batch, device):
        return DPOBatch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            labels=batch.labels.to(device),
            chosen_length=batch.chosen_length
        )

    def compute_loss(self, policy_model, reference_model, batch: DPOBatch) -> torch.FloatTensor:
        outputs = policy_model(batch.input_ids, attention_mask=batch.attention_mask)
        policy_logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']

        with torch.no_grad():
            ref_outputs = reference_model(batch.input_ids, attention_mask=batch.attention_mask)
            reference_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs['logits']

        chosen_length = batch.chosen_length
        policy_chosen_logps = self._get_logps(policy_logits[:chosen_length], batch.labels[:chosen_length])
        policy_rejected_logps = self._get_logps(policy_logits[chosen_length:], batch.labels[chosen_length:])
        reference_chosen_logps = self._get_logps(reference_logits[:chosen_length], batch.labels[:chosen_length])
        reference_rejected_logps = self._get_logps(reference_logits[chosen_length:], batch.labels[chosen_length:])

        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            self.beta
        )

        return losses.mean()

    def _get_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = labels[:, 1:].clone()
        targets[targets == -100] = 0
        return torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    def split_batch(self, batch, start_idx, end_idx):
        return DPOBatch(
            input_ids=batch.input_ids[start_idx:end_idx],
            attention_mask=batch.attention_mask[start_idx:end_idx],
            labels=batch.labels[start_idx:end_idx],
            chosen_length=batch.chosen_length // (end_idx - start_idx)
        )

    def get_batch_size(self, batch):
        return len(batch.input_ids)

class DPOGenerator:
    def __init__(self, threads: List[Thread]):
        config = get_config()
        self.threads = threads
        self.similarity_functions = {
            "Jaccard": jaccard_similarity,
            "Sequence": sequence_similarity,
            "BERT": BERTSimilarity(),
            "USE": USESimilarity(),
            "TF-IDF": TFIDFSimilarity()
        }
        self.sentiment_analyzer = SentimentAnalyzer()
        self.gpt_judge = GPTJudge(model=config.api.gpt_model)
        self.config = config

    def generate_dataset(self) -> Tuple[List[DPOSample], List[DPOSampleStats]]:
        log.info("Starting DPO dataset generation pipeline")

        # Step 1: Compute similarities and select pairs above threshold
        similar_pairs = self._compute_similarities_and_select_pairs()
        log.info(f"Selected {len(similar_pairs)} pairs based on similarity")

        # Step 2: Compute all similarity scores for selected pairs
        self._compute_all_similarities(similar_pairs)

        # Step 3: Filter pairs based on sentiment analysis
        sentiment_filtered_pairs = self._filter_by_sentiment(similar_pairs)
        log.info(f"Filtered to {len(sentiment_filtered_pairs)} pairs based on sentiment")

        # Step 4: Filter pairs based on LLM judgment
        llm_filtered_pairs = self._filter_by_llm_judgment(sentiment_filtered_pairs)
        log.info(f"Filtered to {len(llm_filtered_pairs)} pairs after LLM judgment")

        # Step 5: Convert to simplified DPOSamples
        final_samples = self._convert_to_dpo_samples(llm_filtered_pairs)
        log.info(f"Generated {len(final_samples)} final DPO samples")

        return final_samples, llm_filtered_pairs

    def _compute_all_similarities(self, pairs: List[DPOSampleStats]):
        for pair in tqdm(pairs, desc="Computing all similarities", unit="pair"):
            for similarity_name, similarity_function in self.similarity_functions.items():
                if similarity_name not in pair.similarities:
                    pair.similarities[similarity_name] = similarity_function(pair.chat1, pair.chat2)

    def _compute_similarities_and_select_pairs(self) -> List[DPOSampleStats]:
        similar_pairs = []
        for similarity_name, similarity_function in self.similarity_functions.items():
            log.info(f"Computing similarity windows using {similarity_name} similarity")
            similarity_windows = compute_similarity_windows(self.threads, similarity_function)
            threshold = self.config.similarity.thresholds[similarity_name]
            similar_pairs.extend(self._select_pairs_above_threshold(similarity_windows, threshold, similarity_name))
        return similar_pairs

    def _select_pairs_above_threshold(self, similarity_windows: List[List[float]], threshold: float,
                                      similarity_name: str) -> List[DPOSampleStats]:
        selected_pairs = []
        for thread_id, thread_window in enumerate(similarity_windows):
            thread = self.threads[thread_id]
            for i, window in enumerate(thread_window):
                for j, similarity in enumerate(window):
                    if similarity > threshold:
                        pair = self._create_dpo_sample_stats(thread, thread.chats, i, i + j + 1, similarity, similarity_name)
                        selected_pairs.append(pair)
        return selected_pairs

    def _create_dpo_sample_stats(self, thread: Thread, chats: List[Chat], index1: int, index2: int, similarity: float,
                                 similarity_name: str) -> DPOSampleStats:
        partial_context = [c.message for c in chats[:index1] if c.role == "user"]
        extra_context = [c.message for c in chats[index1 + 1:index2] if c.role == "user"]
        final_context = [chats[index2 + 1].message] if index2 + 1 < len(chats) and chats[
            index2 + 1].role == "user" else None

        return DPOSampleStats(
            similarities={similarity_name: similarity},
            partial_context=partial_context,
            chat1=chats[index1].message,
            chat2=chats[index2].message,
            extra_context=extra_context,
            final_context=final_context,
            sentiment={},  # Will be filled in _filter_by_sentiment
            llm_judgment={}  # Will be filled in _filter_by_llm_judgment
        )

    def _filter_by_sentiment(self, pairs: List[DPOSampleStats]) -> List[DPOSampleStats]:
        filtered_pairs = []
        for pair in tqdm(pairs, desc="Analyzing sentiment", unit="pair"):
            if not pair.final_context:
                pair.sentiment = {}  # Empty sentiment for pairs without final_context
                filtered_pairs.append(pair)
                continue
            text_to_analyze = pair.final_context[-1]
            sentiment_score = self.sentiment_analyzer.score(text_to_analyze)
            pair.sentiment = sentiment_score  # This already contains all the necessary fields
            if self._is_sentiment_acceptable(sentiment_score):
                filtered_pairs.append(pair)
        return filtered_pairs

    def _is_sentiment_acceptable(self, sentiment_score: Dict[str, Union[float, str]]) -> bool:
        nltk_compound = sentiment_score['nltk_compound']
        roberta_score = sentiment_score['roberta_score']

        return (nltk_compound >= self.config.sentiment.nltk_threshold and
                roberta_score >= self.config.sentiment.roberta_threshold)

    def _filter_by_llm_judgment(self, pairs: List[DPOSampleStats]) -> List[DPOSampleStats]:
        filtered_pairs = []
        for pair in tqdm(pairs, desc="LLM Judgment", unit="pair"):
            llm_verdict, rephrased_prompt = self.gpt_judge.validate_dpo_sample(
                pair.partial_context,
                pair.chat1,
                pair.chat2,
                pair.extra_context,
                pair.final_context or []
            )
            pair.llm_judgment = {"accepted": llm_verdict, "rephrased_prompt": rephrased_prompt}

            if llm_verdict:
                filtered_pairs.append(pair)
        return filtered_pairs

    def _convert_to_dpo_samples(self, pairs: List[DPOSampleStats]) -> List[DPOSample]:
        return [
            DPOSample(
                generated_prompt=pair.llm_judgment["rephrased_prompt"] or " ".join(pair.partial_context),
                chat1=pair.chat1,
                chat2=pair.chat2
            )
            for pair in pairs if pair.llm_judgment["accepted"]
        ]