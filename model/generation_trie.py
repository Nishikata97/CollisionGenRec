"""Trie for prefix-constrained beam search.

Inserts all valid SID token sequences into a trie, then provides a callback
for HuggingFace model.generate() that restricts beam expansion to valid prefixes.
"""

from typing import Dict, List


class Trie:

    def __init__(self, sequences: List[List[int]] = None):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for seq in sequences:
                self._add_to_trie(seq, self.trie_dict)
                self.len += 1
        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        self._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return self._get_from_trie(
            prefix_sequence, self.trie_dict,
            self.append_trie, self.bos_token_id,
        )

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(prefix_sequence: List[int], trie_dict: Dict,
                       append_trie=None, bos_token_id: int = None):
        node = trie_dict
        for i, token in enumerate(prefix_sequence):
            if token in node:
                node = node[token]
            elif append_trie:
                return append_trie.get(prefix_sequence[i:])
            else:
                return []
        output = list(node.keys())
        if append_trie and bos_token_id in output:
            output.remove(bos_token_id)
            output += list(append_trie.trie_dict.keys())
        return output

    def __iter__(self):
        def _traverse(prefix, trie_dict):
            if trie_dict:
                for token in trie_dict:
                    yield from _traverse(prefix + [token], trie_dict[token])
            else:
                yield prefix
        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie):
    """Return callback for HuggingFace model.generate()."""
    def prefix_allowed_tokens(batch_id, sentence):
        return candidate_trie.get(sentence.tolist())
    return prefix_allowed_tokens
