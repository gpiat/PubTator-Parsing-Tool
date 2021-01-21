# The PubTator Parsing Tool
A Python package for loading and manipulating PubTator files as Python objects.

## Usage
For basic word tokenization and simple operations
```python
from pubtatortool import PubTatorCorpus
train_corpus = PubTatorCorpus(['train_corpus_part_1.txt',
                               'train_corpus_part_2.txt'])
dev_corpus = PubTatorCorpus(['dev_corpus.txt'])
test_corpus = PubTatorCorpus(['test_corpus.txt'])
```

For wordpiece tokenization and full ability to encode and decode text for use with machine learning models
```python
from pubtatortool import PubTatorCorpus
from pubtatortool.tokenization import get_tokenizer
tokenizer = get_tokenizer(tokenization='wordpiece', vocab='bert-base-cased')
train_corpus = PubTatorCorpus(['train_corpus_part_1.txt',
                               'train_corpus_part_2.txt'], tokenizer)
dev_corpus = PubTatorCorpus(['--dev_corpus.txt'], tokenizer)
test_corpus = PubTatorCorpus(['--test_corpus.txt'], tokenizer)
```

You can then serialize a corpus using Pickle, iterate over documents using `corpus.document_list`, and perform various operations on documents regardless of tokenization policy, even if it is lossy, without worrying about mention and text decoupling.

For example, you can create a TSV-formatted file from a PubTator file in 10 lines of code:
```python
from pubtatortool import PubTatorCorpus
from pubtatortool.tokenization import get_tokenizer
tokenizer = get_tokenizer(tokenization='wordpiece', vocab='bert-base-cased')
corpus = PubTatorCorpus(['mycorpus.txt'], tokenizer)
with open('outfile.txt', 'w') as outfile:
    for doc in corpus.document_list:
        for sentence, targets in zip(doc.sentences, doc.sentence_targets()):
            for token, label in zip(sentence, targets):
                print("{tok}\t{lab}".format(tok=token, lab=label),
                      file=outfile)
            print('', file=outfile)
```

