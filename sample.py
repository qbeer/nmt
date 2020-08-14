from preprocessing.read_data import ReadLanguages
from preprocessing.training_data import TrainingData
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from models.attention_decoder import AttnDecoderRNN
from train.training import Seq2SeqTrainer
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(args):
    attention = args.attention
    input_lang = args.input_lang
    output_lang = args.output_lang
    hidden_size = args.hidden_size

    reader = ReadLanguages(input_lang, output_lang)

    in_lang, out_lang, pairs = reader.prepare()

    data = TrainingData(pairs=pairs, input_lang=in_lang, output_lang=out_lang)

    encoder = EncoderRNN(input_size=data.input_lang.n_words,
                         hidden_size=hidden_size).to(device)

    if attention:
        decoder = AttnDecoderRNN(
            hidden_size=hidden_size,
            output_size=data.output_lang.n_words).to(device)
    else:
        decoder = DecoderRNN(hidden_size=hidden_size,
                             output_size=data.output_lang.n_words).to(device)

    encoder_path = './weights/encoder'
    decoder_path = './weights/decoder'
    if attention:
        encoder_path += "_attention"
        decoder_path += "_attention"

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    seq2seq = Seq2SeqTrainer(encoder=encoder,
                             decoder=decoder,
                             dataset=data,
                             attention=attention)

    while True:
        console_input = input('Enter %s sentence to translate with NMT:\n' %
                              input_lang)
        output, _ = seq2seq.evaluate(input_text=console_input)
        print('Neural machine translation : ' + output, end='\n\n')


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', required=False, default=200, type=int)
parser.add_argument('--input_lang', required=False, default='eng', type=str)
parser.add_argument('--output_lang', required=False, default='hun', type=str)
parser.add_argument('--attention',
                    required=False,
                    default=False,
                    action='store_true')

args = parser.parse_args()

run(args)