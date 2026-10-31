import logging

from words import create_vocab, read_file, tokens_from_text

logger = logging.getLogger("tokenizer")
logging.basicConfig(level=logging.INFO)


class Tokenizer: 

    def __init__(self, **kwargs):
        if "vocab" in kwargs:
            self.vocab = kwargs["vocab"]
        elif "text" in kwargs:
            self.vocab = create_vocab(kwargs["text"])
        elif "filepath" in kwargs:
            text = read_file(kwargs["filepath"])
            self.vocab = create_vocab(text)
        else:
            self.vocab = {}

        self.unk = "<|unk|>"
        self.endoftext = "<|endoftext|>"

        self.vocab[self.unk] = len(self.vocab)
        self.vocab[self.endoftext] = len(self.vocab)
        
        self.inv_vocab = {v:k for k,v in self.vocab.items()}


    def tokenize(self, text):
        return tokens_from_text(text)

    def encode(self, text):
        tokens = self.tokenize(text)
        filtered = [token if token in self.vocab else self.unk for token in tokens]
        encoded = [self.vocab[token] for token in filtered]
        return encoded

    def decode(self, encoded_tokens):
        return "".join([self.inv_vocab[token] for token in encoded_tokens])

if __name__ == "__main__":
    t = Tokenizer(**{"filepath": "the-verdict.txt"})
    logger.info(t.encode("hello"))
    logger.info(t.decode(t.encode("hello")))
