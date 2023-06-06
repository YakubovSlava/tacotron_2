import pandas as pd
import numpy as np
import re
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset
from torchaudio import transforms
from torchaudio.functional import preemphasis
import hyperparams as hps
import matplotlib.pyplot as plt


class TacotronPreprocessor:
    def __init__(
            self,
            save_intonation: bool = True,
            replace_symbol: bool = True
    ):
        self.save_intonation = save_intonation
        self.replace_symbol = replace_symbol
        self.vocab = None
        self.original_texts = None
        self.normalized_texts = None

        pass

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub('[^А-я \!\?\.\,\-]+', ' ', text)
        text = text.strip()
        return text

    def create_vocabulary(self):
        all = []
        for i in self.normalized_texts:
            all += i
        self.vocab = np.unique(all)

    def fit(self, texts: list):
        self.normalized_texts = [self.normalize_text(x) for x in texts]
        self.create_vocabulary()
        del self.normalized_texts

    def transform_single_text(self, text):
        text = text.lower()
        text = re.sub('[^А-я \!\?\.\,\-]+', '', text)
        temp_res = []
        for letter in text:
            index = np.where(self.vocab == letter)[0][0] + 1
            temp_res.append(index)
        return temp_res

    def transform_all_texts(self, texts):
        res = []
        for txt in texts:
            res.append(self.transform_single_text(txt))

        res = np.asarray(res)
        return res
    

class TTSDataset(Dataset):
    def __init__(self, data_path='data/RUSLAN_text/metadata_RUSLAN_16269.csv', num_elements=None):
        super().__init__()
        self.data_path = data_path
        self.dataset = pd.read_csv(data_path, sep='|', header=None)
        if num_elements:
            self.dataset = pd.read_csv(data_path, sep='|', header=None).iloc[:num_elements]
        self.dataset.columns = ['path', 'text']
        self.preprocessor = TacotronPreprocessor()
        self.preprocessor.fit(self.dataset.text.tolist())
        self.transform = transforms.MelSpectrogram(hps.SAMPLE_RATE, n_fft=hps.N_FFT, win_length=hps.WINLEN,
                                       hop_length= hps.HOPLEN, f_min=hps.FMIN, f_max = hps.FMAX,
                                       n_mels=hps.N_MEL_FILTERBANKS, mel_scale='slaney')

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        temp_row = self.dataset.iloc[item]
        path = 'data/RUSLAN/'+temp_row.path+'.wav'
        text = temp_row.text
        text_norm = self.preprocessor.transform_single_text(text)
        text_norm = torch.tensor(text_norm)
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.squeeze(0)
        new_waveform = F.resample(waveform, sample_rate, hps.SAMPLE_RATE)
        mel_spec = self.transform(new_waveform)
        return text_norm, mel_spec
    
def collate_fn(data):
    texts, mels = zip(*data)
    max_text_length = max([x.shape[0] for x in texts])
    text_mask_idx = [x.shape[0] for x in texts]

    max_mel_length = max([x.shape[1] for x in mels]) + 1

    new_texts = torch.zeros(len(texts), max_text_length) 
    for i in range(len(texts)):
        temp_text = texts[i]
        curr_text_length = temp_text.shape[0]
        new_texts[i][:curr_text_length] = temp_text

    new_mels = torch.zeros(len(mels), hps.N_MEL_FILTERBANKS, max_mel_length)
    stops = torch.zeros(len(mels), max_mel_length)
    for i in range(len(mels)):
        temp_mel = mels[i]
        temp_audio_length = temp_mel.shape[1]
        new_mels[i][:, 1:temp_audio_length+1] = temp_mel
        stops[i][temp_audio_length+1:] = 1
    
    new_mels = torch.clip((new_mels), hps.CLIPMIN)
    
    return new_texts.long(), torch.log(new_mels), stops, torch.tensor(text_mask_idx)



# ONLY FOR CUDA USAGE
def reconstruct_audio(mel, stops):
    stops[-1] = 1
    stop = (stops==1).int().argmax()
    mel = (torch.exp(mel[:, :stop]))
    print('entered')
    inverse_melscale_transform = transforms.InverseMelScale(n_stft=hps.N_FFT // 2 + 1, n_mels = hps.N_MEL_FILTERBANKS, 
                                                            sample_rate=hps.SAMPLE_RATE, f_min=hps.FMIN, f_max=hps.FMAX, mel_scale='slaney').cuda()
    spectrogram = inverse_melscale_transform(mel.cuda())
    print('inversed')
    transform = transforms.GriffinLim(n_fft=hps.N_FFT, win_length=hps.WINLEN, hop_length=hps.HOPLEN, n_iter=30).cuda()
    waveform = transform(spectrogram).detach().cpu() * hps.WAV_MAX
    print('transformed')
    return waveform


