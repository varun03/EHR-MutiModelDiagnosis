import torch
import torch.nn as nn


class T5EHRKnowGen(nn.Module):
    """Optional T5-small generation wrapper.

    This is a practical reproduction wrapper rather than a byte-for-byte copy
    of the authors' private/custom training environment.

    The wrapper fuses:
      - clinical/lab/event text
      - CCS knowledge text
    through modality prompts before passing embeddings to T5.
    """

    def __init__(
        self,
        model_name="t5-small",
        num_prompt_tokens=4,
    ):
        super().__init__()

        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name
        )

        hidden = self.model.config.d_model

        self.text_prompt = nn.Parameter(
            torch.randn(
                1, num_prompt_tokens, hidden
            ) * 0.02
        )
        self.css_prompt = nn.Parameter(
            torch.randn(
                1, num_prompt_tokens, hidden
            ) * 0.02
        )

    def forward(
        self,
        input_texts,
        target_texts,
        css_texts,
        max_length=128,
    ):
        device = next(self.parameters()).device

        tokenizer = self.tokenizer

        inputs = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        targets = tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids.to(device)

        targets[targets == tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=targets,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_texts,
        max_length=128,
    ):
        device = next(self.parameters()).device

        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        ids = self.model.generate(
            **inputs,
            max_length=max_length,
        )

        return self.tokenizer.batch_decode(
            ids,
            skip_special_tokens=True
        )
