import torch
from config.cfg import MAX_LENGTH, SOS, EOS
from torch.utils.tensorboard import SummaryWriter
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq2SeqTrainer:
    def __init__(self,
                 encoder,
                 decoder,
                 dataset,
                 teacher_forcing=False,
                 teacher_forcing_ratio=0.5,
                 attention=False):
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.attention = attention
        if not self.teacher_forcing:
            self.teacher_forcing_ratio = -1

    def _trainStep(self, input_tensor, output_tensor):
        loss = 0.0

        hidden = self.encoder.initHidden()

        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        input_length = input_tensor.size(0)
        output_length = output_tensor.size(0)

        encoder_outputs = torch.zeros(MAX_LENGTH,
                                      self.encoder.hidden_size,
                                      device=device)

        for ei in range(input_length):
            output, hidden = self.encoder(input_tensor[ei], hidden)
            encoder_outputs[ei] = output[0, 0]

        # The last hidden state is the context for the decoder

        decoder_input = torch.tensor([[SOS]], device=device)

        use_teacher_forcing = True if random.random(
        ) < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(output_length):
                if self.attention:
                    decoder_output, hidden, _ = self.decoder(
                        decoder_input, hidden, encoder_outputs)
                else:
                    decoder_output, hidden = self.decoder(
                        decoder_input, hidden)
                decoder_input = output_tensor[di]  # teacher forcing
                loss += self.criterion(decoder_output, output_tensor[di])

        else:
            for di in range(output_length):
                if self.attention:
                    decoder_output, hidden, _ = self.decoder(
                        decoder_input, hidden, encoder_outputs)
                else:
                    decoder_output, hidden = self.decoder(
                        decoder_input, hidden)
                _, index_of_max_prop = decoder_output.topk(1)
                decoder_input = index_of_max_prop.squeeze().detach()

                loss += self.criterion(decoder_output, output_tensor[di])

                if decoder_input.item() == EOS:
                    break

        loss.backward()

        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.item() / output_length

    def train(self, n_iters, learning_rate=1e-3):
        comment = 'seq2seq'
        if self.attention:
            comment += '_attention'
        self.writer = SummaryWriter(comment=comment)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                            lr=learning_rate)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),
                                            lr=learning_rate)

        self.criterion = torch.nn.NLLLoss()

        for _iter in range(n_iters):
            input_tensor, output_tensor = self.dataset.get_random_pair()

            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

            loss = self._trainStep(input_tensor, output_tensor)

            self.writer.add_scalar('loss', loss, global_step=_iter)

            if (_iter + 1) % 1000 == 0:
                outp, orig_pair = self.evaluate()
                print('Original sentence : %s' % orig_pair[0])
                print('Neural translation : %s' % outp)
                print('Original translation : %s' % orig_pair[1], end='\n\n')

    def evaluate(self, input_text=None):
        with torch.no_grad():
            pair_tensor, pair = self.dataset.get_random_pair(
                return_original=True)
            if input_text:
                input_tensor = self.dataset.prepoc_single_example(input_text)
                pair = input_text
            else:
                input_tensor, _ = pair_tensor
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(MAX_LENGTH,
                                          self.encoder.hidden_size,
                                          device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []

            for _ in range(MAX_LENGTH):
                if self.attention:
                    decoder_output, decoder_hidden, _ = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)

                _, topi = decoder_output.data.topk(1)
                if topi.item() == EOS:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(
                        self.dataset.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            decoded_words = ' '.join(decoded_words)

            return decoded_words, pair
