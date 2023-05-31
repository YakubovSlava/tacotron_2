from utils.utils import TacotronPreprocessor, TTSDataset, collate_fn, reconstruct_audio, plot_alignment_to_numpy
import pandas as pd
import numpy as np
import re
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
from torchaudio.functional import preemphasis
import hyperparams as hps
from torch.autograd import Variable
from IPython import display
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch import autocast
from torch import nn



class EncoderConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, bias=False, padding=2, dilation=1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        return self.module(x)
    
class Encoder(nn.Module):
    def __init__(self, characters_num, embedding_size, lstm_hidden_size) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(characters_num, embedding_size)
        self.conv_layers = nn.Sequential(
            EncoderConvLayer(embedding_size, embedding_size, 5),
            EncoderConvLayer(embedding_size, embedding_size, 5),
            EncoderConvLayer(embedding_size, embedding_size, 5),
        )
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=lstm_hidden_size,
                           bidirectional=True, batch_first=True)
        self.rnn_dropout = nn.Dropout(0.1)

    
    def forward(self, x: torch.tensor, mask_idx=None):
        """
        На вход подается последовательность символов. Размерность [BATCH_SIZE, NUM_CHARACTERS]
        """
        x = self.char_embedding(x)  #[BATCH_SIZE, NUM_CHARACTERS, EMB_SIZE]
        x = x.transpose(1,2) #[BATCH_SIZE, EMB_SIZE, NUM_CHARACTERS]
        x = self.conv_layers(x)
        x = x.transpose(1,2) #[BATCH_SIZE, NUM_CHARACTERS, CONV_EMB]

        if mask_idx is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, mask_idx, batch_first=True, enforce_sorted=False)   
        x = self.rnn(x)[0]
        if mask_idx is not None:   
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.rnn_dropout(x)
        return x
    
class PreNet(nn.Module):
    def __init__(self, num_mels, prenet_hidden_dim) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(num_mels, prenet_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(prenet_hidden_dim, prenet_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        x = self.module(x)
        return x
        

### Самая сложная часть модели
class Tacotron2Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.location = nn.Conv1d(in_channels=1, out_channels=hps.ATTENTION_LOCATION_FILTERS, 
                                  kernel_size=hps.ATTENTION_LOCATION_KERNEL_SIZE, dilation=1, padding=((hps.ATTENTION_LOCATION_KERNEL_SIZE-1)//2), bias=False)
        self.location_linear = nn.Linear(hps.ATTENTION_LOCATION_FILTERS, hps.ATTENTION_DIM, bias=False)
        self.rnn_hs_linear = nn.Linear(hps.DECODER_RNN_HIDDEN_DIM * 2, hps.ATTENTION_DIM, bias=False)

        self.alignments_linear = nn.Linear(hps.ATTENTION_DIM, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, rnn_hs, encoder_output, processed_encoder_output, location_attention, mask=None):
        """
         На вход подсчета аттеншна передаются:
            - Накопленная информация из РНН 
            - Выход энкодера (BS, seq_len, 2*encoder_lstm_dim)
            - Прогнанный через линейный слой энкодер (чтобы не дублить операцию)
            - commulative-attention (BS, 1, seq_len)
        """

        ### Location attention
        #   Логика:
        #   Входящий вектор на каждую букву прогоняем через свертку таким образом, чтобы у нас на каждую букву было 32 значения, которые будут говорить о том, что данная буква уже встретилась в выданном аудио
        location_attention = location_attention.transpose(1,2) # [BS, seq_len, 1]
        location_attention = self.location(location_attention) # [BS, seq_len, Location_attention_filters]
        location_attention = self.dropout(location_attention)
        location_attention = location_attention.transpose(1,2) # [BS, seq_len, Location_attention_filters]
        location_attention = self.location_linear(location_attention) # [BS, seq_len, attention_dim]

        rnn_hs = torch.cat((rnn_hs[0], rnn_hs[1]), 1) # [BS, 2 * DECODER_RNN_HIDDEN_DIM]
        rnn_hs = rnn_hs.unsqueeze(1) # [BS, 1, 2 * DECODER_RNN_HIDDEN_DIM]
        rnn_hs = self.rnn_hs_linear(rnn_hs) # [BS, 1, Attention_dim]
        alignments = nn.functional.tanh(rnn_hs + location_attention + processed_encoder_output) # (BS, seq_len, Attention_dim)
        alignments = self.alignments_linear(alignments) # (BS, seq_len, 1)


        if mask is not None:
            alignments.data.masked_fill_(~mask.unsqueeze(-1), -torch.inf)
        alignments = self.softmax(alignments) # (BS, seq_len, 1)

        attention_score = (encoder_output.transpose(1, 2) @ alignments).transpose(1, 2)

        

        
        return  attention_score, alignments



class DecoderPostNetConv(nn.Module):
    def __init__(self, in_kernels, out_kernels, last_layer=False) -> None:
        super().__init__()
        self.last_layer = last_layer
        self.post_net = nn.Sequential(
            nn.Conv1d(in_kernels, out_kernels, hps.POSTNET_KERNEL_SIZE, padding=2, bias=False, dilation=1),
            nn.BatchNorm1d(hps.POSTNET_NUM_FILTERS),
            nn.Identity() if last_layer else nn.Tanh(),
            nn.Dropout(0.5),
        )
    def forward(self, x):
        return self.post_net(x)
    

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prenet = PreNet(hps.N_MEL_FILTERBANKS, hps.PRENET_HIDDEN_SIZE)
        self.encoder_linear = nn.Linear(hps.LSTM_HIDDEN_SIZE * 2, hps.ATTENTION_DIM)
        self.decoder_rnn = nn.LSTM(input_size=hps.PRENET_HIDDEN_SIZE + hps.CHARACTER_EMB_SIZE, 
                                   hidden_size=hps.DECODER_RNN_HIDDEN_DIM, batch_first=True, num_layers=2)
        self.attention = Tacotron2Attention()
        self.linear_projection = nn.Linear(hps.DECODER_RNN_HIDDEN_DIM + hps.CHARACTER_EMB_SIZE, hps.N_MEL_FILTERBANKS)
        self.stop_projection = nn.Linear(hps.DECODER_RNN_HIDDEN_DIM + hps.CHARACTER_EMB_SIZE, 1)

        self.post_net = nn.Sequential(
            DecoderPostNetConv(hps.N_MEL_FILTERBANKS, hps.POSTNET_NUM_FILTERS, hps.POSTNET_NUM_FILTERS),
            DecoderPostNetConv(hps.POSTNET_NUM_FILTERS, hps.POSTNET_NUM_FILTERS),
            DecoderPostNetConv(hps.POSTNET_NUM_FILTERS, hps.POSTNET_NUM_FILTERS),
            DecoderPostNetConv(hps.POSTNET_NUM_FILTERS, hps.POSTNET_NUM_FILTERS),
            DecoderPostNetConv(hps.POSTNET_NUM_FILTERS, hps.POSTNET_NUM_FILTERS, last_layer=True)
        )
        self.post_linear = nn.Linear(hps.POSTNET_NUM_FILTERS, hps.N_MEL_FILTERBANKS)
        self.dropout = nn.Dropout(0.1)

        
    
    def forward(self, mels, encoder_output, mask):
        mels = mels.transpose(1,2)
        mels = self.prenet(mels)
        processed_encoder = self.encoder_linear(encoder_output)


        next_h = torch.zeros(2, mels.shape[0], hps.DECODER_RNN_HIDDEN_DIM, device=mels.device)
        next_c = torch.zeros(2, mels.shape[0], hps.DECODER_RNN_HIDDEN_DIM, device=mels.device)

        mel_predictions = []
        stop_tokens = []
        curr_attention_context = torch.zeros(encoder_output.shape[0], 1, hps.CHARACTER_EMB_SIZE, device=mels.device)
        cummulated_attention = torch.zeros(encoder_output.shape[0], encoder_output.shape[1], 1, device=mels.device)

        next_h.requires_grad = True
        next_c.requires_grad = True
        curr_attention_context.requires_grad = True
        cummulated_attention.requires_grad = True




        for i in range(mels.shape[1]):
            curr_mel = mels[:, i, :].unsqueeze(1)
            curr_rnn_input = torch.cat((curr_mel, curr_attention_context), dim=-1)
            next_mel, (next_h, next_c) = self.decoder_rnn(curr_rnn_input, (next_h, next_c))
            next_mel = self.dropout(next_mel)
            curr_attention_context, alignments = self.attention(next_h, encoder_output, processed_encoder, cummulated_attention, mask)
            cummulated_attention = cummulated_attention + alignments
            next_mel_inp = torch.cat((next_mel, curr_attention_context), dim=2)
            next_mel = self.linear_projection(next_mel_inp)
            next_stop = self.stop_projection(next_mel_inp).squeeze(1)
            mel_predictions.append(next_mel)
            stop_tokens.append(next_stop)

        result_mel = torch.cat(mel_predictions, dim=1)
        result_stops = torch.cat(stop_tokens, dim=1)
        result_mel = result_mel.transpose(1, 2)
        result_mel_post = self.post_net(result_mel)
        result_mel_post = result_mel_post.transpose(1,2)
        result_mel_post = self.post_linear(result_mel_post)
        result_mel_post = result_mel_post.transpose(1,2)
        result_mel_post = result_mel + result_mel_post
        
        
        return result_mel, result_mel_post, result_stops
    
    def predict(self, encoder_output):
        mels = torch.log(torch.clamp(torch.zeros(encoder_output.shape[0], 1, hps.N_MEL_FILTERBANKS, device=encoder_output.device), hps.CLIPMIN))
        mels = self.prenet(mels)
        processed_encoder = self.encoder_linear(encoder_output)


        next_h = torch.zeros(2, mels.shape[0], hps.DECODER_RNN_HIDDEN_DIM, device=mels.device)
        next_c = torch.zeros(2, mels.shape[0], hps.DECODER_RNN_HIDDEN_DIM, device=mels.device)

        mel_predictions = []
        mel_predictions_post = []
        stop_tokens = []
        attentions = []

        curr_attention_context = torch.zeros(encoder_output.shape[0], 1, hps.CHARACTER_EMB_SIZE, device=mels.device)
        cummulated_attention = torch.zeros(encoder_output.shape[0], encoder_output.shape[1], 1, device=mels.device)

        next_h.requires_grad = True
        next_c.requires_grad = True
        curr_attention_context.requires_grad = True
        cummulated_attention.requires_grad = True


        for i in range(1500):
            curr_mel = mels
            curr_rnn_input = torch.cat((curr_mel, curr_attention_context), dim=-1)
            next_mel, (next_h, next_c) = self.decoder_rnn(curr_rnn_input, (next_h, next_c))
            curr_attention_context, alignments = self.attention(next_h, encoder_output, processed_encoder, cummulated_attention)
            cummulated_attention = cummulated_attention + alignments
            next_mel_inp = torch.cat((next_mel, curr_attention_context), dim=2)
            next_mel = self.linear_projection(next_mel_inp)

            next_mel_post = self.post_net(next_mel.transpose(1,2)).transpose(1,2)
            next_mel_post = self.post_linear(next_mel_post)
            next_mel_post = next_mel_post + next_mel
            mels = self.prenet(next_mel_post)

            next_stop = self.stop_projection(next_mel_inp).squeeze(1)
            mel_predictions.append(next_mel)
            mel_predictions_post.append(next_mel_post)
            stop_tokens.append(next_stop)
            attentions.append(alignments)

        
        attention = torch.cat(attentions, dim=1)
        result_mel = torch.cat(mel_predictions, dim=1)
        result_mel_post = torch.cat(mel_predictions_post, dim=1)
        result_stops = torch.cat(stop_tokens, dim=1)

        return result_mel_post.transpose(1,2), result_stops, attention



class Tacotron2(nn.Module):
    def __init__(self, characters_num: int = 0) -> None:
        super().__init__()
        self.characters_num = characters_num
        self.encoder = Encoder(characters_num, hps.CHARACTER_EMB_SIZE, hps.LSTM_HIDDEN_SIZE)
        self.decoder = Decoder()

    def get_mask(self, mask_idx):
        mask = torch.zeros(mask_idx.shape[0], max(mask_idx))
        mask = ((torch.arange(0, max(mask_idx)).unsqueeze(1)<torch.tensor(mask_idx))).transpose(0,1)
        return mask

    def forward(self, text, mels, mask_idx):
        encoder_output = self.encoder(text, torch.tensor(mask_idx))
        mask = self.get_mask(torch.tensor(mask_idx))
        mask = mask.to(encoder_output.device)
        decoder_output = self.decoder(mels, encoder_output, mask)
        return decoder_output
    
    def predict(self, text):
        encoder_output = self.encoder(text)
        decoder_output = self.decoder.predict(encoder_output)
        return decoder_output



