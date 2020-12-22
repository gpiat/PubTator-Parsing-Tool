import pickle
import string

from abc import abstractmethod
from enum import Enum
from nltk.tokenize import TweetTokenizer
from transformers import BertTokenizer


class TokenType(Enum):
    WORD = 0
    CHARACTER = 1
    CHAR = 1
    WORDPIECE = 2
    WP = 2

    @classmethod
    def from_str(cls, s):
        s = s.lower()
        try:
            return cls(int(s))
        except ValueError:
            pass
        if s in ['word']:
            return cls.WORD
        elif s in ['char', 'character']:
            return cls.CHAR
        elif s in ['wordpiece', 'wp']:
            return cls.WP
        else:
            raise ValueError("{} not recognized as a token type."
                             " Available token types are 'word',"
                             " 'char' and 'wordpiece'.".format(s))


class BaseTokenizer:
    """ This is not a class that should be instantiated. Its encode() method
        calls the tokenize() method which is not declared, but should be
        implemented in subclasses.
    """

    def __init__(self, vocab_fname=None):
        """ Args:
                vocab_fname (str): optional. If given, the vocabulary will be
                loaded from the file, which should be formatted with one token
                per line.
        """
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.pad_token_id = 0
        self.unk_token_id = 1
        if vocab_fname is not None:
            with open(vocab_fname, 'rb') as f:
                vocab = pickle.load(f)
            self.init_vocab(vocab)

    def init_vocab(self, vocab):
        self.vocab = {word: (number + 2)
                      for number, word in enumerate(vocab)}
        self.vocab[self.unk_token] = self.unk_token_id
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab_size = len(self.vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    @abstractmethod
    def tokenize(self):
        raise NotImplementedError

    @abstractmethod
    def detokenize(self):
        raise NotImplementedError

    def encode(self, text):
        if type(text) == str:
            text = self.tokenize(text)
        try:
            encoded = self._encode_decode(text,
                                          converter=self.vocab,
                                          default_value=self.unk_token_id)
        except AttributeError as e1:
            e2 = AttributeError("It seems vocabulary was not initialized and"
                                " therefore encoding could not be performed.")
            e2.__traceback__ = e1.__traceback__
            raise e2
        return encoded

    def decode(self, encoded_text):
        # Unlike the use of the _encode_decode() in encode(), this does not
        # need to be in a try / except block because if this function is
        # called, encode() must have been called beforehand.
        decoded = self._encode_decode(encoded_text,
                                      converter=self.ids_to_tokens,
                                      default_value=self.unk_token)
        decoded = self.detokenize(decoded)
        return decoded

    def _encode_decode(self, input_txt, converter, default_value):
        output_text = []
        for token in input_txt:
            try:
                output_text.append(converter[token])
            except KeyError:
                output_text.append(default_value)
        return output_text


class CharTokenizer(BaseTokenizer):
    """ Character-level tokenizer.
    """

    def __init__(self, vocab_fname):
        super().__init__(vocab_fname)
        self.tokenization = TokenType.CHAR

    def tokenize(self, text):
        return list(text)

    def detokenize(self, tokenized_txt):
        return ''.join(tokenized_txt)


class WordTokenizer(BaseTokenizer):
    """ Word-level tokenizer, uses NLTK's tweet tokenizer to separate words.
    """

    def __init__(self, vocab_fname):
        super().__init__(vocab_fname)
        self.tokenization = TokenType.WORD
        self.tknzr = TweetTokenizer()

    def tokenize(self, text):
        return self.tknzr.tokenize(text)

    def detokenize(self, tokenized_txt):
        detok_txt = ''
        for tok in tokenized_txt:
            detok_txt += tok
            if tok not in string.punctuation:
                detok_txt += ' '
        return detok_txt


def get_tokenizer(tokenization, vocab):
    if type(tokenization) == str:
        tokenization = TokenType.from_str(tokenization)
    if tokenization == TokenType.CHAR:
        return CharTokenizer(vocab)
    elif tokenization == TokenType.WP:
        tok = BertTokenizer.from_pretrained(vocab)
        tok.tokenization = TokenType.WP
        return tok
    else:
        return WordTokenizer(vocab)
