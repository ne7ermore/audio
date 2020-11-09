import numpy as np
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
import librosa

def butter_highpass(cutoff, sr=24000, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)     

def get_mel_bias_transpose(sr=24000):
    return mel(sr, 1024, fmin=90, fmax=7600, n_mels=80).T

def amp_to_db(x, min_level_db=-100):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def normalize(S, bias=-16, min_level_db=-100):
    return np.clip((S + bias - min_level_db) / -min_level_db, 0, 1)    

def denormalize(S, bias=-16, min_level_db=-100):
    return ((np.clip(S, 0, 1) * -min_level_db) + min_level_db - bias) / 20

if __name__ == "__main__":
    sample_rate = 24000
    wav = "../wavs/1.wav"
    b, a = butter_highpass(30, sample_rate)
    mel_bias_t = get_mel_bias_transpose(sample_rate)
    x, _ = librosa.load(wav, sr=sample_rate, mono=True)
    x = signal.filtfilt(b, a, x)
    D = pySTFT(x).T
    D_mel = np.dot(D, mel_bias_t)
    S = normalize(amp_to_db(D_mel))
    print(S)