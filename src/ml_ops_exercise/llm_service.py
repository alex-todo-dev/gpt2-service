import os

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

from .config import MODEL_NAME, PAD_TOKEN_ID


class LLMService:
    def __init__(self, model: str = MODEL_NAME) -> None:
        cache_dir = os.getenv('HF_HOME', None)
        # Load tokenizer and model explicitly with offline mode
        load_kwargs = {'local_files_only': True}
        if cache_dir:
            load_kwargs['cache_dir'] = cache_dir
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model, **load_kwargs)
        model_obj = GPT2LMHeadModel.from_pretrained(model, **load_kwargs)
        
        # Detect and use GPU if available
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == 0:
            model_obj = model_obj.to('cuda')
        
        # Pass pre-loaded model and tokenizer to pipeline to ensure offline mode
        # device=0 for GPU, device=-1 for CPU
        self.generator = pipeline(
            'text-generation',
            model=model_obj,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, add_special_tokens=False)

    def generate(
        self,
        text: str,
        temp: float,
        top_p: float,
        max_new_tokens: int,
        num_return_sequences: int
    ) -> list[dict[str, str]]:
        return self.generator(
            text,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            truncation=True,
            do_sample=True,
            pad_token_id=PAD_TOKEN_ID,
        )


llm_service = LLMService(model=MODEL_NAME)
