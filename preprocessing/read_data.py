from .language import Lang
from config.cfg import MAX_LENGTH
from config.utils import preproc_single_sentence


class ReadLanguages:
    def __init__(self, lang1, lang2, max_length=MAX_LENGTH, reverse=False):
        self.lang1 = lang1
        self.lang2 = lang2
        self.max_length = max_length
        self.reverse = reverse

    def _normalizeString(self, s):
        s = preproc_single_sentence(s)
        return s

    def _preprocess(self):
        lines = open('./data/%s-%s.txt' % (self.lang1, self.lang2),
                     encoding='utf-8').read().strip().split('\n')

        pairs = [
            line.split('\t')[:2]
            if not self.reverse else line.split('\t')[:2].reverse()
            for line in lines
        ]

        normalized_pairs = [[self._normalizeString(s) for s in pair]
                            for pair in pairs]

        return normalized_pairs

    def _filterPair(self, p):
        return len(p[0].split(' ')) < self.max_length and \
            len(p[1].split(' ')) < self.max_length

    def prepare(self):
        input_lang = Lang(self.lang1)
        output_lang = Lang(self.lang2)
        normalized_pairs = self._preprocess()
        filtered_pairs = list(filter(self._filterPair, normalized_pairs))
        for pair in filtered_pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print(
            f'Input language : {input_lang.name}, number of words : {input_lang.n_words}'
        )
        print(
            f'Output language : {output_lang.name}, number of words : {output_lang.n_words}',
            end='\n\n')
        return input_lang, output_lang, filtered_pairs
