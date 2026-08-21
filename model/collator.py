"""Batch collators for training and testing."""


class Collator:
    """Training collator: pads inputs and labels, masks pad tokens to -100."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        if isinstance(batch[0]["input_ids"], str):
            inputs = self.tokenizer(
                [d["input_ids"] for d in batch],
                return_tensors="pt", padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_attention_mask=True,
            )
            labels = self.tokenizer(
                [d["labels"] for d in batch],
                return_tensors="pt", padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
        else:
            inputs = self.tokenizer.pad(
                [{"input_ids": d["input_ids"]} for d in batch],
                return_tensors="pt", return_attention_mask=True,
            )
            labels = self.tokenizer.pad(
                [{"input_ids": d["labels"]} for d in batch],
                return_tensors="pt",
            )
        inputs["labels"] = labels["input_ids"]
        inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
        return inputs


class TestCollator:
    """Test collator: returns (tokenized inputs, raw target strings)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt", padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_attention_mask=True,
        )
        return inputs, targets
