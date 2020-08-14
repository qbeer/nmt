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
    teacher_forcing = args.teacher_forcing
    n_iters = args.n_iters
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

    try:
        encoder.load_state_dict(
            torch.load('./weights/encoder' +
                       '_attention' if attention else ''))
        decoder.load_state_dict(
            torch.load('./weights/decoder' +
                       '_attention' if attention else ''))
    except Exception:
        print('Could not load pre-trained weights, traning from scartch',
              end='\n\n')
        pass

    seq2seq = Seq2SeqTrainer(encoder=encoder,
                             decoder=decoder,
                             dataset=data,
                             teacher_forcing=teacher_forcing,
                             attention=attention)

    seq2seq.train(n_iters=n_iters)

    torch.save(encoder.state_dict(),
               './weights/encoder' + '_attention' if attention else '')
    torch.save(decoder.state_dict(),
               './weights/decoder' + '_attention' if attention else '')


parser = argparse.ArgumentParser()
parser.add_argument('--n_iters', required=False, default=200_000, type=int)
parser.add_argument('--hidden_size', required=False, default=200, type=int)
parser.add_argument('--input_lang', required=False, default='eng', type=str)
parser.add_argument('--output_lang', required=False, default='hun', type=str)
parser.add_argument('--teacher_forcing',
                    required=False,
                    default=True,
                    action='store_true')
parser.add_argument('--attention',
                    required=False,
                    default=True,
                    action='store_true')

args = parser.parse_args()

run(args)