from email.mime import audio
import numpy as np
from scipy.signal import hilbert
import scipy.fftpack as fft
import numpy as np
from scipy.signal import resample
from scipy.interpolate import PchipInterpolator
import soundfile as sf
from scipy.signal import resample_poly
import math

audio_sr, dwnsmp_sr = 4000,100

wav,sr = sf.read('/scratch/snormanh_lab/shared/Sigurd/spectrotemporal-synthesis-v2/speech.wav')

def freq2erb(freq_Hz):
    """Converts Hz to ERBs, using the formula of Glasberg and Moore."""
    n_erb = 9.265 * np.log(1 + freq_Hz / (24.7 * 9.265))
    return n_erb

def erb2freq(n_erb):
    """Converts ERBs to Hz, using the formula of Glasberg and Moore."""
    freq_Hz = 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)
    return freq_Hz

def freq2erb_ferret(freq_Hz):
    """
    ERB bandwidths based on Sumner & Palmer:
    ERB = 0.31 * freq_kHz^0.533
    N_ERB derived by integrating the reciprocal of the equation above.
    """
    freq_kHz = freq_Hz / 1000
    n_erb = (1 / 0.31) * (1 / (1 - 0.533)) * freq_kHz ** (1 - 0.533)
    return n_erb

def erb2freq_ferret(n_erb):
    """Inverse of freq2erb_ferret"""
    freq_kHz = (n_erb * 0.31 * (1 - 0.533)) ** (1 / (1 - 0.533))
    freq_Hz = freq_kHz * 1000
    return freq_Hz


def make_erb_cos_filters(signal_length, sr, N, low_lim, hi_lim, animal='human'):
    nfreqs = signal_length // 2  # does not include DC
    max_freq = sr / 2
    freqs = np.linspace(0, max_freq, nfreqs + 1)  # go all the way to nyquist

    if hi_lim > sr / 2:
        hi_lim = max_freq
    # make cutoffs evenly spaced on an erb scale
    cutoffs = erb2freq(np.linspace(freq2erb(low_lim), freq2erb(hi_lim), N + 2))

    cos_filts = np.zeros((nfreqs + 1, N))
    for k in range(N):
        l = cutoffs[k]
        h = cutoffs[k + 2]  # adjacent filters overlap by 50%
        l_ind = np.min(np.where(freqs > l))
        h_ind = np.max(np.where(freqs < h))
        avg = (freq2erb(l) + freq2erb(h)) / 2
        rnge = (freq2erb(h) - freq2erb(l))
        cos_filts[l_ind:h_ind, k] = np.cos((freq2erb(freqs[l_ind:h_ind]) - avg) / rnge * np.pi)

    # add lowpass and highpass to get perfect reconstruction
    filts = np.zeros((nfreqs + 1, N + 2))
    filts[:, 1:N + 1] = cos_filts
    h_ind = np.max(np.where(freqs < cutoffs[1]))  # lowpass filter goes up to peak of first cos filter
    filts[0:h_ind, 0] = np.sqrt(1 - filts[0:h_ind, 1] ** 2)
    l_ind = np.min(np.where(freqs > cutoffs[N]))  # highpass filter goes down to peak of last cos filter
    filts[l_ind:nfreqs + 1, N + 1] = np.sqrt(1 - filts[l_ind:nfreqs + 1, N] ** 2)

    Hz_cutoffs = cutoffs

    return filts, Hz_cutoffs, freqs


def format_wav(wav, wav_sr, max_duration_sec,audio_sr):
    """
    Format a waveform according to the parameters in dict P
    wav: input waveform
    wav_sr: input sampling rate
    P: dictionary of parameters with keys 'max_duration_sec' and 'audio_sr'
    """

    # convert to mono if stereo
    if len(wav.shape) > 1 and wav.shape[1] == 2:
        wav = np.mean(wav, axis=1)

    # shorten
    duration_sec = len(wav) / wav_sr
    if duration_sec > max_duration_sec:
        wav = wav[:int(wav_sr * max_duration_sec)]

    # resample to desired audio rate
    gcd = math.gcd(wav_sr, audio_sr)

    # 计算上采样和下采样率
    up = audio_sr // gcd
    down = wav_sr // gcd

    wav = resample_poly(wav, up, down, axis=0)

    # wav = resample(wav, int(len(wav) * audio_sr / wav_sr))

    # set RMS to 0.01
    wav = 0.01 * wav / np.sqrt(np.mean(np.square(wav)))

    return wav

def generate_subbands(signal, filts):
    if signal.ndim == 1:  # turn into column vector
        signal = np.expand_dims(signal, axis=1)

    N = filts.shape[1] - 2
    signal_length = len(signal)
    filt_length = filts.shape[0]

    fft_sample = fft.fft(signal.flatten())  # FFT works on 1D arrays in Python
    
    if signal_length % 2 == 0:  # even length
        fft_filts = np.concatenate((filts, np.flip(filts[1:-1], axis=0)), axis=0)
    else:  # odd length
        fft_filts = np.concatenate((filts, np.flip(filts[1:], axis=0)), axis=0)

    fft_subbands = fft_filts * np.stack([fft_sample] * (N+2), axis=1)
    subbands = np.real(fft.ifft(fft_subbands, axis=0))  # ifft works on columns; imag part is small, probably discretization error?

    return subbands



def resample_time_and_freq(X, t1_sr, t2_sr, f1, f2):
    """
    Resamples time axis and interpolates frequency axis of cochleogram X.
    
    X: input matrix
    t1_sr: input temporal sampling rate
    t2_sr: output temporal sampling rate
    f1: input frequencies
    f2: output frequencies
    """

    # resample time axis
    Yt = resample(X, int(X.shape[0]*t2_sr/t1_sr))

    # interpolate frequency axis
    n_t = Yt.shape[0]
    Ytf = np.empty((n_t, len(f2)))
    for i in range(n_t):
        interpolator = PchipInterpolator(np.log2(f1), Yt[i,:], extrapolate=True)
        Ytf[i,:] = interpolator(np.log2(f2))
        
    return Ytf


wav,sr = sf.read('/scratch/snormanh_lab/shared/Sigurd/spectrotemporal-synthesis-v2/speech.wav')
max_duration_sec=1
audio_sr=100
wav = format_wav(wav, sr, max_duration_sec,audio_sr)
        # make_erb_cos_filters(length(wav), P.audio_sr, ...
        # P.n_filts, P.lo_freq_hz, P.audio_sr/2, P.animal);

n_filts = 17
lo_freq_hz = 100



# audio_filts = make_erb_cos_filters(len(wav),audio_sr, n_filts, lo_freq_hz, audio_sr/2, 'human')
ranges = np.arange(len(wav)//2*(n_filts+2))
#reshape
audio_filts = np.reshape(ranges,(len(wav)//2,n_filts+2))


audio_low_cutoff = [200, 8000]
logf_spacing = 0.1
R = {}

logf = np.arange(np.log2(audio_low_cutoff[0]), 
                 np.log2(audio_low_cutoff[-1]) + logf_spacing, logf_spacing)
logf = 2 ** logf


coch_subbands_audio_sr_erbf = generate_subbands(wav, audio_filts)



analytic_subbands_audio_sr_erbf = hilbert(coch_subbands_audio_sr_erbf, axis=0)
coch_envs_audio_sr_erbf = np.abs(analytic_subbands_audio_sr_erbf)


coch_envs_compressed_audio_sr_erbf = coch_envs_audio_sr_erbf ** R['compression_factor']

# coch_envs_compressed_dwnsmp_sr_logf = resample_time_and_freq(...
#     coch_envs_compressed_audio_sr_erbf, ...
#     audio_sr, dwnsmp_sr, audio_low_cutoff, logf);

coch_envs_compressed_dwnsmp_sr_logf = resample_time_and_freq(coch_envs_compressed_audio_sr_erbf, audio_sr, dwnsmp_sr, audio_low_cutoff, logf)