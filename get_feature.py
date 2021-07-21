import math
import numpy as np
import librosa

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def normalize_signal(signal, gain=None):
    """
    Normalize float32 signal to [-1, 1] range
    """
    if gain is None:
        gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain



def get_speech_features(signal, sample_freq, params):
    """
    Get speech features using either librosa (recommended) or
    python_speech_features
    Args:
        signal (np.array): np.array containing raw audio signal
        sample_freq (float): sample rate of the signal
        params (dict): parameters of pre-processing
    Returns:
        np.array: np.array of audio features with shape=[num_time_steps,
        num_features].
        audio_duration (float): duration of the signal in seconds
    """

    backend = params.get('backend', 'psf')
    features_type = params.get('input_type', 'spectrogram')
    num_features = params['num_audio_features']
    window_size = params.get('window_size', 20e-3)
    window_stride = params.get('window_stride', 10e-3)
    augmentation = params.get('augmentation', None)

    if backend == 'librosa':
        window_fn = WINDOWS_FNS[params.get('window', "hanning")]
        dither = params.get('dither', 0.0)
        num_fft = params.get('num_fft', None)
        norm_per_feature = params.get('norm_per_feature', False)
        mel_basis = params.get('mel_basis', None)
        gain = params.get('gain')
        mean = params.get('features_mean')
        std_dev = params.get('features_std_dev')
        features, duration = get_speech_features_librosa(
                signal, sample_freq, num_features, features_type,
                window_size, window_stride, augmentation, window_fn=window_fn,
                dither=dither, norm_per_feature=norm_per_feature, num_fft=num_fft,
                mel_basis=mel_basis, gain=gain, mean=mean, std_dev=std_dev
            )
    else:
        print("error backend")
        return None

    return features, duration



def get_speech_features_librosa(signal, sample_freq, num_features,
                                features_type='spectrogram',
                                window_size=20e-3,
                                window_stride=10e-3,
                                augmentation=None,
                                window_fn=np.hanning,
                                num_fft=None,
                                dither=0.0,
                                norm_per_feature=False,
                                mel_basis=None,
                                gain=None,
                                mean=None,
                                std_dev=None):
    """
    Function to convert raw audio signal to numpy array of features.
    Backend: librosa
    Args:
        signal (np.array): np.array containing raw audio signal.
        sample_freq (float): frames per second.
        num_features (int): number of speech features in frequency domain.
        pad_to (int): if specified, the length will be padded to become divisible
            by ``pad_to`` parameter.
        features_type (string): 'mfcc' or 'spectrogram'.
        window_size (float): size of analysis window in milli-seconds.
        window_stride (float): stride of analysis window in milli-seconds.
        augmentation (dict, optional): dictionary of augmentation parameters. See
            :func:`augment_audio_signal` for specification and example.

    Returns:
        np.array: np.array of audio features with shape=[num_time_steps,
        num_features].
        audio_duration (float): duration of the signal in seconds
    """
    signal = normalize_signal(signal.astype(np.float32), gain)
    if augmentation:
        signal = augment_audio_signal(signal, sample_freq, augmentation)
    audio_duration = len(signal) * 1.0 / sample_freq

    n_window_size = int(sample_freq * window_size)
    n_window_stride = int(sample_freq * window_stride)
    num_fft = num_fft or 2**math.ceil(math.log2(window_size*sample_freq))

    
    if dither > 0:
        signal += dither*np.random.randn(*signal.shape)


    if features_type == 'spectrogram':
        # ignore 1/n_fft multiplier, since there is a post-normalization
        powspec = np.square(np.abs(librosa.core.stft(
            signal, n_fft=n_window_size,
            hop_length=n_window_stride, win_length=n_window_size, center=True,
            window=window_fn)))
        # remove small bins
        powspec[powspec <= 1e-30] = 1e-30
        features = 10 * np.log10(powspec.T)

        assert num_features <= n_window_size // 2 + 1, \
            "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

        # cut high frequency part
        features = features[:, :num_features]

    elif features_type == 'mfcc':
        signal = preemphasis(signal, coeff=0.97)
        S = np.square(
                np.abs(
                    librosa.core.stft(signal, n_fft=num_fft,
                                    hop_length=int(window_stride * sample_freq),
                                    win_length=int(window_size * sample_freq),
                                    center=True, window=window_fn
                    )
                )
            )
        features = librosa.feature.mfcc(
            sr=sample_freq, S=S, n_mfcc=num_features, n_mels=2*num_features).T

    elif features_type == 'logfbank':
        signal = preemphasis(signal,coeff=0.97)
        S = np.abs(librosa.core.stft(signal, n_fft=num_fft,
                                    hop_length=int(window_stride * sample_freq),
                                    win_length=int(window_size * sample_freq),
                                    center=True, window=window_fn))**2.0
        if mel_basis is None:
            # Build a Mel filter
            mel_basis = librosa.filters.mel(sample_freq, num_fft, n_mels=num_features,
                                            fmin=0, fmax=int(sample_freq/2))
            features = np.log(np.dot(mel_basis, S) + 1e-20).T

    else: 
        raise ValueError('Unknown features type: {}'.format(features_type))

    norm_axis = 0 if norm_per_feature else None
    if mean is None:
        mean = np.mean(features, axis=norm_axis)
    if std_dev is None:
        std_dev = np.std(features, axis=norm_axis)

    features = (features - mean) / std_dev

    if augmentation:
        n_freq_mask = augmentation.get('n_freq_mask', 0)
        n_time_mask = augmentation.get('n_time_mask', 0)
        width_freq_mask = augmentation.get('width_freq_mask', 10)
        width_time_mask = augmentation.get('width_time_mask', 50)

        for idx in range(n_freq_mask):
            freq_band = np.random.randint(width_freq_mask + 1)
            freq_base = np.random.randint(0, features.shape[1] - freq_band)
            features[:, freq_base:freq_base+freq_band] = 0
        for idx in range(n_time_mask):
            time_band = np.random.randint(width_time_mask + 1)
            if features.shape[0] - time_band > 0:
                time_base = np.random.randint(features.shape[0] - time_band)
                features[time_base:time_base+time_band, :] = 0

    # now it is safe to pad
    # if pad_to > 0:
    #   if features.shape[0] % pad_to != 0:
    #     pad_size = pad_to - features.shape[0] % pad_to
    #     if pad_size != 0:
    #         features = np.pad(features, ((0,pad_size), (0,0)), mode='constant')
    return features, audio_duration


def test():
    import sys
    import scipy.io.wavfile as wave
    wav_path = sys.argv[1]
    sample_freq, signal = wave.read(wav_path)
    params = {
        "input_type":"logfbank", 
        "backend":"librosa", 
        "num_audio_features": 64, 
        "window_size": 20e-3, 
        "window_stride":10e-3, 
        "window": "hanning", 
        "dither": 1e-5, 
        "norm_per_feature": True
    }
    features, duration = get_speech_features(signal, sample_freq, params)
    print("feature shape: ", features.shape)
    print("features: \n", features)

if __name__ == "__main__":
    test()