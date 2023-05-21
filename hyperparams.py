# AUDIO
SAMPLE_RATE = 22050
FRAME_SIZE_MS = 50
FRAME_HOP_MS = 12.5
WINLEN = 1024
HOPLEN = 256
CLIPMIN = 0.01


FRAME_SIZE = FRAME_SIZE_MS / 1000
FRAME_HOP = FRAME_HOP_MS / 1000

N_MEL_FILTERBANKS = 80
FMIN = 0
FMAX = 8000
POWER = 1.5
N_FFT = 1024



FMIN_paper = 125
FMAX_paper = 7600



# MODEL

### Encoder
CHARACTER_EMB_SIZE = 512
ENCODER_KERNEL_SIZE = 5
LSTM_HIDDEN_SIZE = 256


### Decoder
### Prenet
PRENET_HIDDEN_SIZE = 256

### Attention

ATTENTION_RNN_DIM = 256
ATTENTION_DIM = 128

ATTENTION_LOCATION_FILTERS = 32
ATTEMTION_LOCATION_KERNEL_SIZE = 31



