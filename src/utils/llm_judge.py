import openai
from typing import List, Dict, Any, Tuple
from src.utils.logger import log
from config import get_config


class GPTJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        config = get_config()
        self.model = model
        self.client = openai.Client(api_key=config.api.gpt_api_key,
                                    organization=config.api.gpt_org_id,
                                    project=config.api.gpt_project_id)

    def validate_dpo_sample(self, partial_context: List[str], chat1: str, chat2: str, extra_context: List[str],
                            final_context: List[str]) -> Tuple[bool, str]:
        messages = self._create_initial_prompt(partial_context, chat1, chat2, extra_context, final_context)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            response = completion.choices[0].message.content.strip().lower()
            is_valid = "yes" in response

            if is_valid:
                unified_prompt = self._generate_unified_prompt(messages)
            else:
                unified_prompt = None

            return is_valid, unified_prompt

        except Exception as e:
            log.error(f"Error in GPT validation: {e}")
            return False, None

    def _create_initial_prompt(self, partial_context: List[str], chat1: str, chat2: str, extra_context: List[str],
                               final_context: List[str]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "Only answer with Yes/No."},
            {"role": "user", "content": (
                "You will receive a list of prompts, called partial_context, written by a human in the course of a conversation "
                "the human had with a GPT model. The model's responses in the conversation are omitted, except for chat1, which is "
                "the model's response to the last message in partial_context. After chat1, the human continued the conversation "
                "with the model, producing another list of prompts, called extra_context. The model replied to the last prompt in "
                "extra_context with a response referred to as chat2. Finally, there's a final_context, which is the user's message "
                "that came after chat2. This conversation was selected because it was determined that chat1 and chat2 look similar. "
                "Your task will be to assess if chat2 is a valid continuation of partial_context. Your answer will be a 'yes' if chat2 could "
                "be a valid continuation of partial_context, and a 'no' otherwise. Use extra_context and final_context for this "
                "assessment about the validity of the pair (partial_context, chat2). You have to be cautious, the messages might "
                "look very similar at first glance, but they may actually be responses to completely different prompts. You should "
                "say 'no' to such cases. You're provided with extra_context and final_context, user's reaction to chat2, to help "
                "you figure out if (partial_context, chat2) is really logically valid."
            )},
            {"role": "user", "content": (
                f"partial_context: {partial_context}\n\n"
                f"chat1: {chat1}\n\n"
                f"extra_context: {extra_context}\n\n"
                f"chat2: {chat2}\n\n"
                f"final_context: {final_context}"
            )}
        ]

    def _generate_unified_prompt(self, initial_messages: List[Dict[str, str]]) -> str:
        unified_prompt_messages = [
            {"role": "system",
             "content": "Now, write a single user prompt that captures the information in the partial_context such that both chat1 "
                        "and chat2 could be valid continuations. Ignore extra_context and final_context for this task."},
        ]

        messages = initial_messages + unified_prompt_messages

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Error in generating unified prompt: {e}")
            return None