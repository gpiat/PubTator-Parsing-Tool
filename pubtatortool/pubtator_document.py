import itertools
import nltk
from diff_match_patch import diff_match_patch


def text_preprocess(text):
    """ Takes the raw title and abstract as written in the file
        and keeps only the relevant text and PMID.
        Args:
            text (str): raw text of the form "PMID|?|Text\n"
                with '?' being 't' or 'a' depending on whether
                the text is a title or abstract.
        return:
            pmid (str): PMID of the article, of 1 to 8 digits
            text (str): cleaned text
    """
    # separating PMID and letter from the text.
    # *text captures the rest of the list in case
    # there happens to be a | in the text.
    pmid, _, *text = text.split("|")
    # joining the various fields of the title in case there
    # happens to be a | in the title text; removing \n
    text = "|".join(text)[:-1]

    return pmid, text


class UMLS_Entity:
    """ This data structure allows easy access to the various fields
        which describe a mention of a UMLS entity in a text.
    """

    def __init__(self, text):
        self._orig_text = text
        # The first field that is ignored is the PMID of the document.
        (_, self.start_idx, self.stop_idx, self.mention_text,
         self.semantic_type_ID, self.cui) = text.split('\t')

        # The CUI may be formatted as "C0123456" or "UMLS:C0123456".
        # We want to chop off the "UMLS:" if applicable.
        self.cui = self.cui.split(':')[-1]

        self.start_idx = int(self.start_idx)
        self.stop_idx = int(self.stop_idx)

    def __str__(self):
        return self._orig_text


class PubTatorDocument:
    """ This class instantiates a document from the PubTator corpus
        using the information provided in the PubTator format.
        Attr:
            abstract (str): Abstract of the article
            char_level_targets
            pmid (str): PMID of the document
            raw_text (str): simple concatenation of title and abstract. The
                indexing of characters in raw_text matches the one used in
                PubTator entity mention annotations.
            sentences
            start_end_indices
            targets
            text (list<str>): title fused with abstract but as a list of tokens
            title (str): Title of the article
            token_to_char_lookup
            tokenization
            tokenizer
            umls_entities (list<UMLS_Entity>): list of UMLS entity
                mentions in the text
            vocab
    """

    def __init__(self, title, abstract, mentions, tokenizer):
        """ Args:
                - (str) title: raw title line of text
                - (str) abstract: raw abstract line of text
                - (list<str>) mentions: list of raw lines
                    of text containing the UMLS entity mentions.
                - tokenizer: TODO.
        """
        self.pmid, self.title = text_preprocess(title)
        _, self.abstract = text_preprocess(abstract)
        # no space is insterted between title and abstract to match up
        # with PubTator PubTator format.
        self.raw_text = self.title + '\n' + self.abstract
        self.umls_entities = [UMLS_Entity(entity) for entity in mentions]

        try:
            sentence_tok = nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download('punkt')
            sentence_tok = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = tokenizer
        self.sentences = self.raw_text.split('\n')
        # self.sentences is now of the form ['Title', 'Abstract. Stuff.']
        self.sentences.extend(sentence_tok.tokenize(self.sentences[1]))
        # self.sentences is now of the form
        # ['Title', 'Abstract. Stuff.', 'Abstract.', 'Stuff.']
        del self.sentences[1]
        # self.sentences is now of the form ['Title', 'Abstract.', 'Stuff.']
        self.sentences = [self.tokenizer.tokenize(sentence)
                          for sentence in self.sentences]
        # for example, if tokenization is wordpiece, self.sentences is now
        # of the form [['Title'], ['Abs', '##tract', '.'], ['Stuff', '.']]

        self.text = self.tokenizer.tokenize(self.raw_text)
        self.tokenization = tokenizer.tokenization
        self._initialize_targets()

        # list of all start and end indices of all entities
        # originally the stop index is exclusive, but we need it
        # to be inclusive and vice-versa for the start index.
        self.start_end_indices = list(itertools.chain(
            [(e.start_idx - 1, e.stop_idx - 1) for e in self.umls_entities]))

    def _initialize_targets(self):
        char_level_targets = [None] * len(self.raw_text)
        for i in range(len(char_level_targets)):
            for e in self.umls_entities:
                if i >= e.start_idx and i < e.stop_idx:
                    char_level_targets[i] = e.cui, e.semantic_type_ID
                    continue
                elif i > e.stop_idx:
                    continue

        token = iter(self.text)
        concat_tokens = ''.join(self.text)
        dmp = diff_match_patch()
        diff = dmp.diff_main(self.raw_text, concat_tokens)
        # this diff library creates diffs of the form:
        #   [(flag, substring), (flag, substring), ...]
        # where "flag" can be 1, -1 or 0 depending on whether
        # the substring is in concat_tokens but not raw_text,
        # vice-versa, or in both repectively.
        # example:
        #   raw_text = 'Nonylphenol'
        #   concat_tokens = 'Non##yl##phe##no##l'
        #   diff = [(0, 'Non'), (1, '##'), (0, 'yl'), (1, '##'),
        #           (0, 'phe'), (1, '##'), (0, 'no'), (1, '##'), (0, 'l')]
        # However, it is much easier to handle a character-level diff, e.g. :
        #   [(0, 'N'), (0, 'o'), (0, 'n'), (1, '#'), (1, '#'), (0, 'y'), ...]
        # This is what we set out to do with the following list
        # comprehension, which may seem obscure, but it is 30%
        # faster than the equivalent loop, which can be written as:
        # new_diff = []
        # for flag, sub_str in diff:
        #     new_diff += list(zip([flag] * len(sub_str), sub_str))
        # diff = new_diff
        #
        # diff = list(itertools.chain(*[zip([a] * len(b), b)
        #                               for a, b in diff]))
        #
        # In actuality, once we have a sequence of numbers that tell us
        # whether a character comes from one string, the other or both, the
        # character itself is redundant information. Removing it from
        # consideration further divides execution time by 2.
        diff = list(itertools.chain(*[[flag] * len(sub_str)
                                      for flag, sub_str in diff]))

        # initializing the targets for each token as None
        token_targets = [None] * len(self.text)
        # This helps us keep track of where we are within the current token
        # so that we know when to move on to the next token
        chars_left_in_current_token = len(next(token))
        # Index of the currently tracked character in the raw_text, which
        # always refers (if possible) to the same character as the character
        # defined by chars_left_in_current_token. I'm not completely certain
        # why it needs to start at -1 rather than 0, it just works.
        current_char_index = -1
        current_token_index = 0
        # we maintain a lookup table of tokens to characters for the purpose
        # of versatility.
        self.token_to_char_lookup = {current_token_index: []}
        for flag in diff:
            # hypothetically, we can choose any time to assign the
            # token's label but we choose to do so when we're on the
            # last character of the current token.
            if chars_left_in_current_token == 0:
                token_targets[current_token_index] =\
                    char_level_targets[current_char_index]
                current_token_index += 1
                # We also add the current token to the lookup table
                self.token_to_char_lookup[current_token_index] = []
                chars_left_in_current_token = len(next(token))
            # we're pointing to the same character in both the
            # raw_text and the tokenized text, so update both
            # pointers for the next iteration
            if flag == 0:
                current_char_index += 1
                # we add the current character index to the lookup table
                self.token_to_char_lookup[current_token_index].append(
                    current_char_index)
                chars_left_in_current_token -= 1
            # we're on a character missing in the tokenized text,
            # so we skip it
            elif flag == -1:
                current_char_index += 1
                # we do not add the current character index to the lookup
                # table, since it is not in the tokens
            # we're on a character missing in the raw_text, so we skip it
            else:
                chars_left_in_current_token -= 1
        self.char_level_targets = char_level_targets
        self.targets = token_targets

    def get_vocab(self):
        try:
            return self.vocab
        except AttributeError:
            self.vocab = list(set(self.text))
        return self.vocab

    def write_to(self, filename):
        with open(filename, 'a') as f:
            f.write('|'.join([self.pmid, 't', self.title]) + '\n')
            f.write('|'.join([self.pmid, 'a', self.abstract]) + '\n')
            for entity in self.umls_entities:
                f.write(str(entity) + '\n')
            f.write('\n')

    def sentence_targets(self):
        """ The targets are a single list with one element per (sub)token.
            In case the targets are needed and must be formatted the same
            as sentences, this generator will extract the relevant sublists
            on-the-fly.
            This assumes that the sentence parser doesn't split two sentences
            in the middle of a (sub)token.
        """
        start = 0
        for sentence in self.sentences:
            end = start + len(sentence)
            yield self.targets[start:end]
            start = end
