from .pubtator_document import PubTatorDocument
from .tokenization import WordTokenizer


def peek(file, n=1):
    """ Reads n lines ahead in a file without changing the file
        object's current position in the file
        Args:
            - file: a text file
            - n (int): how many lines ahead should be read, defaults to 1
        Return:
            - line: last line read
    """
    pos = file.tell()
    lines = [file.readline() for i in range(n)]
    file.seek(pos)
    return lines[-1]


class PubTatorCorpus:
    """ This class instantiates a PubTator corpus using one or more
        PubTator-formatted files. Its main purpose is to iterate over
        documents with the documents() generator function.
    """

    def __init__(self, fnames, tokenizer=None):
        """ Args:
                - fnames (list<str>): list of filenames in the corpus
                - auto_looping (bool): whether retrieving documents should
                    automatically loop or not
                - tokenizer: Optional (defaults to
                    `tokenization_tools.WordTokenizer`). A tokenizer that
                    implements a `tokenize()` method and has a `tokenization`
                    attribute of type `tokenization_tools.TokenType`.
        """

        self._filenames = fnames
        self._currentfile = 0

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = WordTokenizer(None)
        self.tokenization = tokenizer.tokenization
        self._init_documents()
        self.n_documents = len(self.document_list)

        self.cuids, self.stids, self.vocab = self._get_cuids_and_vocab()
        self.nconcepts = len(self.cuids)

    def _get_cuids_and_vocab(self):
        """ Collects the CUIDs, STIDs and vocabulary of the corpus.
            Should not be used outside of the constructor, because
            it relies on the document counter being at the start.
            If you need CUIDs or vocab, use the appropriate attributes.
        """
        cuids = {}
        stids = {}
        vocab = set()
        for document in self.document_list:
            for entity in document.umls_entities:
                if entity.concept_ID in cuids:
                    cuids[entity.concept_ID] += 1
                else:
                    cuids[entity.concept_ID] = 1
                if entity.semantic_type_ID in stids:
                    stids[entity.semantic_type_ID] += 1
                else:
                    stids[entity.semantic_type_ID] = 1
            vocab = vocab.union(document.get_vocab())
        return cuids, stids, vocab

    def _init_documents(self):
        self.document_list = []
        while self._currentfile < len(self._filenames):
            # opening the file -- not using `with` because
            # it causes excessive indentation
            f = open(self._filenames[self._currentfile], 'r')

            next_line = None
            while next_line != '' and peek(f) != '':
                title = f.readline()
                abstract = f.readline()
                next_line = f.readline()

                # after the abstract, each entity mention is written on a
                # separate line. The next document comes after a newline.
                umls_entities = []
                while next_line != '\n' and next_line != '':
                    # [:-1] deletes the trailing newline character
                    umls_entities.append(next_line[:-1])
                    next_line = f.readline()

                self.document_list.append(PubTatorDocument(title, abstract,
                                                           umls_entities,
                                                           self.tokenizer))
            f.close()

            self._currentfile += 1

    def documents(self):
        """ Yields a document containing:
                - pmid (str): the next document's PMID
                - title (str): the next document's title
                - abstract (str): the next document's abstract
                - umls_entities (list<str>): list of UMLS entities
                    for the next document
        """
        for document in self.document_list:
            yield document
