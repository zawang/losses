import scipy.io.wavfile
import torch
import numpy as np

from PMSQE_asteroid.enc_dec import Filterbank, _EncDec, Encoder
from PMSQE_asteroid.pmsqe import SingleSrcPMSQE
from PMSQE_asteroid.stft_fb import STFTFB
from PMSQE_asteroid.transforms import take_mag, check_complex


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_PMSQE(reference_file, estimated_file):
    """
        Example use of SingleSrcPMSQE function with 8 or 16 Khz files.
        Assumes that reference_file and estimated_file have the same shape.
        
        :param reference_file:     Reference wav filename
        :param estimated_file:     Estimated wav filename
    """
    
    # Load audio files and convert to tensors
    fs, ref = scipy.io.wavfile.read(reference_file)
    _, est = scipy.io.wavfile.read(estimated_file)
    ref = torch.from_numpy(ref).float().to(device)
    est = torch.from_numpy(est).float().to(device)
    
    # Normalize audio to be between -1.0 and 1.0
    ref = ref / torch.max(torch.abs(ref))
    est = est / torch.max(torch.abs(est))
    
    # If the shape of each of the files is (time,), then that's the common single channel case with a batch size of 1
    # So, add two dimensions of 1 each so the shape becomes (batch, channels, time) = (1, 1, time)
    if (ref.ndim == 1):
        ref = ref.unsqueeze(0).unsqueeze(0)
        est = est.unsqueeze(0).unsqueeze(0)
    
    kernel_size = 0
    if (fs == 16000):
        kernel_size = 512
    elif (fs == 8000):
        kernel_size = 256
    else:
        raise ValueError("Unsupported sample rate {}".format(fs))

    stft = Encoder(STFTFB(device=device, kernel_size=kernel_size, n_filters=kernel_size, stride=kernel_size//2))
    ref_spec = take_mag(stft(ref))
    est_spec = take_mag(stft(est))
    loss_func = SingleSrcPMSQE(device=device, window_name = 'hann', sample_rate=fs)
    loss_value = loss_func(est_spec, ref_spec)
    return loss_value

test_PMSQE('testing_examples/sp04.wav', 'testing_examples/sp04_babble_sn10.wav')
