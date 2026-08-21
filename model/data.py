"""Sequential recommendation dataset for T5.

Loads inter.json + index JSON, remaps items to SID strings, and generates
(history → target) pairs for train/valid/test splits.

Split convention: last item = test, second-to-last = valid, rest = train.
"""

import json
import os

import numpy as np
from torch.utils.data import Dataset


class SeqRecDataset(Dataset):

    def __init__(self, args, mode="train", sample_num=-1):
        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.max_his_len = args.max_his_len
        self.index_file = args.index_file
        self.mode = mode
        self.sample_num = sample_num

        self.new_tokens = None
        self.all_items = None
        self._pre_tokenized = False

        self._load_data()
        self._remap_items()
        self.inter_data = self._process_data(mode)

    def _load_data(self):
        inter_path = os.path.join(self.data_path, f"{self.dataset}.inter.json")
        index_path = os.path.join(self.data_path, self.index_file)
        with open(inter_path) as f:
            self.inters = json.load(f)
        with open(index_path) as f:
            self.indices = json.load(f)

    def _remap_items(self):
        """Map item IDs to concatenated SID strings."""
        self.remapped_inters = {}
        for uid, items in self.inters.items():
            self.remapped_inters[uid] = [
                "".join(self.indices[str(i)]) for i in items
            ]

    def get_new_tokens(self):
        """Return sorted list of unique SID tokens (e.g., '<a_0>', '<b_1>')."""
        if self.new_tokens is None:
            self.new_tokens = sorted({
                token for index in self.indices.values() for token in index
            })
        return self.new_tokens

    def get_all_items(self):
        """Return set of all valid SID strings."""
        if self.all_items is None:
            self.all_items = {"".join(idx) for idx in self.indices.values()}
        return self.all_items

    def _process_data(self, mode: str) -> list:
        inter_data = []
        for uid, items in self.remapped_inters.items():
            if mode == "train":
                seq = items[:-2]
                for i in range(1, len(seq)):
                    history = seq[:i]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    inter_data.append({
                        "item": seq[i],
                        "inters": "".join(history),
                    })
            else:
                target = items[-2] if mode == "valid" else items[-1]
                history = items[:-2] if mode == "valid" else items[:-1]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                inter_data.append({
                    "item": target,
                    "inters": "".join(history),
                })

        if mode == "test" and self.sample_num > 0:
            idx = np.random.choice(len(inter_data), self.sample_num, replace=False)
            inter_data = np.array(inter_data)[idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def pre_tokenize(self, tokenizer):
        """Cache tokenized inputs/labels to avoid per-epoch overhead."""
        self._tok_inputs = []
        self._tok_labels = []
        for d in self.inter_data:
            self._tok_inputs.append(tokenizer(
                d["inters"], max_length=tokenizer.model_max_length,
                truncation=True,
            )["input_ids"])
            self._tok_labels.append(tokenizer(
                d["item"], max_length=tokenizer.model_max_length,
                truncation=True,
            )["input_ids"])
        self._pre_tokenized = True

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        if self._pre_tokenized:
            return dict(
                input_ids=self._tok_inputs[index],
                labels=self._tok_labels[index],
            )
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])
