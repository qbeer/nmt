from preprocessing.read_data import ReadLanguages
from preprocessing.training_data import TrainingData
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from train.training import Seq2SeqTrainer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reader = ReadLanguages('eng', 'hun')
in_lang, out_lang, pairs = reader.prepare()

data = TrainingData(pairs=pairs, input_lang=in_lang, output_lang=out_lang)

encoder = EncoderRNN(input_size=data.input_lang.n_words,
                     hidden_size=200).to(device)
decoder = DecoderRNN(hidden_size=200,
                     output_size=data.output_lang.n_words).to(device)

encoder.load_state_dict(torch.load('./weights/encoder'))
decoder.load_state_dict(torch.load('./weights/decoder'))

seq2seq = Seq2SeqTrainer(encoder=encoder,
                         decoder=decoder,
                         dataset=data,
                         teacher_forcing=True)

seq2seq.train(n_iters=200_000)

torch.save(encoder.state_dict(), './weights/encoder')
torch.save(decoder.state_dict(), './weights/decoder')