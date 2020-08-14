import torch
import random
from config.cfg import SOS, EOS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainingData:
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.SOS_token = SOS
        self.EOS_token = EOS

    def _indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def _tensorFromSentence(self, lang, sentence):
        indexes = self._indexesFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long,
                            device=device).view(-1, 1)

    def _tensorsFromPair(self, pair):
        input_tensor = self._tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self._tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    def get_random_pair(self, return_original=False):
        random_pair = random.choice(self.pairs)
        random_pair_tensor = self._tensorsFromPair(random_pair)
        if return_original:
            return (random_pair_tensor, random_pair)
        return random_pair_tensor
