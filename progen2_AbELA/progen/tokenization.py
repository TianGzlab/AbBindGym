import json


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class CharacterTokenizer:
    def __init__(self, vocab):
        self.vocab = dict(vocab)
        self.inv_vocab = {value: key for key, value in self.vocab.items()}

    @classmethod
    def from_file(cls, path):
        with open(path) as handle:
            data = json.load(handle)
        return cls(data["model"]["vocab"])

    def encode(self, text):
        if text in self.vocab:
            return Encoded([self.vocab[text]])

        ids = []
        for char in text:
            if char not in self.vocab:
                raise ValueError(f"token {char!r} is not present in tokenizer vocabulary")
            ids.append(self.vocab[char])
        return Encoded(ids)

    def decode_batch(self, batch):
        return ["".join(self.inv_vocab[int(token)] for token in tokens) for tokens in batch]


def load_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError:
        return CharacterTokenizer.from_file(path)
    return Tokenizer.from_file(path)
