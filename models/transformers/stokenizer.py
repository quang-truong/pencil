from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict


class STokenizer(PreTrainedTokenizer):
    # Class variable to store the tokenizer instance
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def max_num_nodes(self) -> int:
        return len([key for key in self.get_vocab().keys() if key.isdigit()])

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def __init__(self, num_nodes: int = 256):
        # Nodes tokens
        self.num_nodes = num_nodes
        self.vocab = {str(i): i for i in range(0, self.num_nodes)}

        # Special tokens
        self.vocab["<|latent|>"] = num_nodes
        self.vocab["<nodes_end>"] = num_nodes + 1
        self.vocab["<edges_end>"] = num_nodes + 2
        self.vocab["|"] = num_nodes + 3  # edge separator
        self.vocab["[Q]"] = num_nodes + 4
        self.vocab["[R]"] = num_nodes + 5
        self.vocab["[A]"] = num_nodes + 6
        # Add special tokens
        self.vocab["<eos>"] = num_nodes + 7

        # Create inverse vocabulary (id to token mapping)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        # Set special token attributes
        self.pad_token = "<eos>"
        self.eos_token = "<eos>"
        self.bos_token = "<eos>"
        # self.unk_token = '<unk>'

        # Initialize parent class first
        super().__init__(
            pad_token=self.pad_token, eos_token=self.eos_token, bos_token=self.bos_token
        )

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dict"""
        return self.vocab.copy()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        # Split on whitespace and validate each token is a number in range
        tokens = []
        for token in text.replace("\n", " ").strip().split():
            if token in self.vocab:
                tokens.append(token)
            else:
                raise ValueError(f"Token {token} not in vocabulary")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        # Convert token to id, return unk_token_id if token not in vocab
        return self.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        # Convert id back to token
        return self.ids_to_tokens[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Join tokens with spaces
        return " ".join(tokens)
