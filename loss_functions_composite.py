"""
Based off of https://github.com/santi-pdp/segan_pytorch/blob/master/segan/utils.py and https://github.com/mpariente/asteroid/tree/master/asteroid

All metrics are ready to use as loss functions. Before using PMSQE or any composite metric that uses PMSQE, InitializePMSQE must be called first.

All metrics besides PMSQE take in 2D tensor time series where the first axis is the batch size (i.e. the number of ref-deg pairs being evaluated) and the second axis is time (i.e. number of samples). For example, if every audio file is 3 samples long and we are evaluating 5 ref-deg pairs, then both ref_wav and deg_wav should have shape torch.Size([5, 3]).

PMSQE takes in tensors of shape (batch size, channels, time).
"""

import torch
import numpy as np

# For PMSQE
from PMSQE_asteroid.enc_dec import Filterbank, _EncDec, Encoder
from PMSQE_asteroid.pmsqe import SingleSrcPMSQE
from PMSQE_asteroid.stft_fb import STFTFB
from PMSQE_asteroid.transforms import take_mag, check_complex

# For WSS and LLR
alpha = 0.95

# Equalizes the length of both signals
def preprocessing(ref_wav, deg_wav):
    len_ = min(ref_wav.shape[1], deg_wav.shape[1])
    ref_wav = ref_wav[:, :len_]
    deg_wav = deg_wav[:, :len_]
    
    return ref_wav, deg_wav

def SNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    clean_speech = ref_wav
    processed_speech = deg_wav
    
    dif = ref_wav - deg_wav
    overall_snr = 10 * torch.log10(torch.sum(ref_wav ** 2) / (torch.sum(dif ** 2) + 10e-20))
    return -overall_snr

def SSNR(ref_wav, deg_wav, srate=16000, device=torch.device('cpu'), eps=1e-10):
    """
    Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
    This function implements the segmental signal-to-noise ratio
    as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = clean_speech.shape[1]
    processed_length = processed_speech.shape[1]
    
    # Scale both to have same dynamic range.
    clean_speech = clean_speech - torch.mean(clean_speech)
    processed_speech = processed_speech - torch.mean(processed_speech)
    
    # global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    MIN_SNR = -10      # minimum SNR in dB
    MAX_SNR = 35       # maximum SNR in dB
    
    # For each frame, calculate SSNR
    num_frames = int(clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = torch.linspace(1, winlength, winlength, device=device) / (winlength + 1)
    window = 0.5 * (1 - torch.cos(2 * np.pi * time))
    segmental_snr = []
    
    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[:, start:start+winlength]
        processed_frame = processed_speech[:, start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        
        # (2) Compute Segmental SNR
        signal_energy = torch.sum(clean_frame ** 2)
        noise_energy = torch.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * torch.log10(signal_energy / (noise_energy + eps)+ eps))
#        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
#        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)

    return -torch.mean(torch.tensor(segmental_snr, dtype=torch.float32, requires_grad=True, device=device))

def WSS(ref_wav, deg_wav, srate=16000, device=torch.device('cpu')):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = clean_speech.shape[1]
    processed_length = processed_speech.shape[1]
    
    assert clean_length == processed_length, clean_length
    
    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25 # num of critical bands
    
    USE_FFT_SPECTRUM = 1
    n_fft = int(2 ** np.ceil(np.log(2*winlength)/np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1
    
    # Critical band filter definitions (Center frequency and BW in Hz)
    
    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30,
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93,
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                              95.3398, 105.411, 116.256, 127.914, 140.423,
                              153.823, 168.154, 183.457, 199.776, 217.153,
                              235.631, 255.255, 276.072, 298.126, 321.465,
                              346.136]
                 
    bw_min = bandwidth[0] # min critical bandwidth
        
    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.

    min_factor = np.exp(-30. / (2 * 2.303)) # -30 dB point of filter

    crit_filter = torch.zeros((num_crit, n_fftby2), device=device)
    all_f0 = []
    for i in range(num_crit):
        f0 = torch.tensor((cent_freq[i] / max_freq) * (n_fftby2), dtype=torch.float32, device=device)
        all_f0.append(torch.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = torch.tensor(np.log(bw_min) - np.log(bandwidth[i]), dtype=torch.float32, device=device)
        j = torch.tensor(list(range(n_fftby2)), dtype=torch.float32, device=device)
        crit_filter[i, :] = torch.exp(-11 * (((j - torch.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor).float()

    # For each frame of input speech, compute Weighted Spectral Slope Measure

    # num of frames
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0 # starting sample
    time = torch.linspace(1, winlength, winlength, device=device) / (winlength + 1)
    window = 0.5 * (1 - torch.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[:, start:start+winlength]
        processed_frame = processed_speech[:, start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
            
        # (2) Compute Power Spectrum of clean and processed
        clean_frame = clean_frame[:, :n_fft]
        clean_frame_len = clean_frame.shape[1]
            
        if (clean_frame_len < n_fft):
            # Pad with zeros
            clean_frame = torch.cat((clean_frame, torch.zeros(clean_frame.shape[0], n_fft - clean_frame_len, device=device)), 1)
        
        img = torch.zeros(clean_frame.shape[0], n_fft, device=device) # imaginary components used to run torch.fft
        clean_frame = torch.transpose(torch.stack((clean_frame, img)), 0, 2)
        clean_frame_fft = torch.fft(clean_frame, 2)
        clean_spec = torch.transpose(clean_frame_fft[:, :, 0]**2 + clean_frame_fft[:, :, 1]**2, 0, 1)
            
        processed_frame = processed_frame[:, :n_fft]
        processed_frame_len = processed_frame.shape[1]

        if (processed_frame_len < n_fft):
            # Pad with zeros
            processed_frame = torch.cat((processed_frame, torch.zeros(processed_frame.shape[0], n_fft - processed_frame_len, device=device)), 1)
        processed_frame = torch.transpose(torch.stack((processed_frame, img)), 0, 2)
        processed_frame_fft = torch.fft(processed_frame, 2)
        processed_spec = torch.transpose(processed_frame_fft[:, :, 0]**2 + processed_frame_fft[:, :, 1]**2, 0, 1)
            
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit
        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = torch.sum(clean_spec[:, :n_fftby2] * crit_filter[i, :])
            processed_energy[i] = torch.sum(processed_spec[:, :n_fftby2] * crit_filter[i, :])
        clean_energy = torch.tensor(clean_energy, dtype=torch.float32, device=device).view(-1, 1)
        eps = torch.ones((clean_energy.shape[0], 1), device=device) * 1e-10
        clean_energy = torch.cat((clean_energy, eps), axis=1)
        clean_energy = 10 * torch.log10(torch.max(clean_energy, dim=1)[0])
        processed_energy = torch.tensor(processed_energy, dtype=torch.float32, device=device).view(-1, 1)
        processed_energy = torch.cat((processed_energy, eps), axis=1)
        processed_energy = 10 * torch.log10(torch.max(processed_energy, dim=1)[0])
            
        # (4) Compute Spectral Shape (dB[i+1] - dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
        processed_slope = processed_energy[1:num_crit] - \
            processed_energy[:num_crit-1]
        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])
        # (6) Compute the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)
        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = torch.tensor(clean_loc_peak, dtype=torch.float32, device=device)
        processed_loc_peak = torch.tensor(processed_loc_peak, dtype=torch.float32, device=device)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - \
                                   clean_energy[:num_crit-1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - \
                                processed_energy[:num_crit-1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - \
                                       processed_energy[:num_crit-1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(torch.sum(W * (clean_slope[:num_crit - 1] - \
                                     processed_slope[:num_crit - 1]) ** 2))
                                                                            
        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / torch.sum(W)
        start += int(skiprate)
    wss_dist_vec = torch.tensor(distortion, dtype=torch.float32, device=device)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    return torch.mean(torch.tensor(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))], dtype=torch.float32, requires_grad=True, device=device))

def LLR(ref_wav, deg_wav, srate=16000, device=torch.device('cpu')):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = clean_speech.shape[1]
    processed_length = processed_speech.shape[1]
    
    assert clean_length == processed_length, clean_length
    
    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16
    
    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = torch.linspace(1, winlength, winlength, device=device) / (winlength + 1)
    window = 0.5 * (1 - torch.cos(2 * np.pi * time))
    distortion = []
    
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[:, start:start+winlength]
        processed_frame = processed_speech[:, start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P, device)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P, device)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]
        # (3) Compute the LLR measure
        numerator = torch.mm(torch.mm(A_processed, toeplitz(R_clean, device)), torch.transpose(A_processed, 0, 1))
        denominator = torch.mm(torch.mm(A_clean, toeplitz(R_clean, device)), torch.transpose(A_clean, 0, 1))
        log_ = torch.log(numerator / denominator)
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    
    LLR_dist = np.array(distortion)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    return torch.mean(torch.tensor(LLRs[:LLR_len], dtype=torch.float32, device=device))

def lpcoeff(speech_frame, model_order, device=torch.device('cpu')):
    # (1) Compute Autocor lags
    # max?
    winlength = speech_frame.shape[1]
    R = []
    #R = [0] * (model_order + 1)
    for k in range(model_order + 1):
        first = speech_frame[:, :(winlength - k)]
        second = speech_frame[:, k:winlength]
        #raise NotImplementedError
        R.append(torch.sum(first * second))
    #R[k] = np.sum( first * second)
    # (2) Lev-Durbin
    a = torch.ones((model_order,), device=device)
    E = torch.zeros((model_order + 1,), device=device)
    rcoeff = torch.zeros((model_order,), device=device)
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = torch.sum(a_past * torch.tensor(R[i:0:-1], dtype=torch.float32, device=device))
        rcoeff[i] = (R[i+1] - sum_term)/E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * torch.flip(a_past, [0])
        E[i+1] = (1-rcoeff[i]*rcoeff[i])*E[i]
    acorr = torch.tensor(R, dtype=torch.float32, device=device)
    a = a * -1
    lpparams = torch.tensor([1] + list(a), dtype=torch.float32, device=device)
    return acorr, rcoeff, lpparams

# Constructs a toeplitz matrix
def toeplitz(a, device=torch.device('cpu')):
    n = len(a)
    b = torch.zeros((n, n), device=device)
    for i in range(n):
        row_idx = torch.arange(i, n)
        col_idx = torch.arange(0, n - i)
        b[row_idx, col_idx] = b[col_idx, row_idx] = a[i]
    
    return b

kernel_size = None
stft = None
PMSQE = None

def InitializePMSQE(srate, device=torch.device('cpu')):
    global kernel_size
    global stft
    global PMSQE
    
    if (srate == 16000):
        kernel_size = 512
    elif (srate == 8000):
        kernel_size = 256
    else:
        raise ValueError("Unsupported sample rate {}".format(fs))
        
    stft = Encoder(STFTFB(device=device, kernel_size=kernel_size, n_filters=kernel_size, stride=kernel_size//2))
    PMSQE = SingleSrcPMSQE(device=device, window_name = 'hann', sample_rate=srate)

def CSIG(ref_wav, deg_wav, srate=16000, device=torch.device('cpu')):
    wss_dist = WSS(ref_wav, deg_wav, srate, device)

    llr_mean = LLR(ref_wav, deg_wav, srate, device)

    ref_spec = take_mag(stft(ref_wav.unsqueeze(0).permute(1, 0, 2)))
    deg_spec = take_mag(stft(deg_wav.unsqueeze(0).permute(1, 0, 2)))
    pmsqe_raw = 1 / torch.mean(PMSQE(deg_spec, ref_spec))

    return -(3.093 - 1.029 * llr_mean + 0.603 * pmsqe_raw - 0.009 * wss_dist)

def CBAK(ref_wav, deg_wav, srate=16000, device=torch.device('cpu')):
    wss_dist = WSS(ref_wav, deg_wav, srate, device)
    
    segSNR = SSNR(ref_wav, deg_wav, srate, device)
    
    ref_spec = take_mag(stft(ref_wav.unsqueeze(0).permute(1, 0, 2)))
    deg_spec = take_mag(stft(deg_wav.unsqueeze(0).permute(1, 0, 2)))
    pmsqe_raw = 1 / torch.mean(PMSQE(deg_spec, ref_spec))
    
    return -(1.634 + 0.478 * pmsqe_raw - 0.007 * wss_dist + 0.063 * -segSNR)

def COVL(ref_wav, deg_wav, srate=16000, device=torch.device('cpu')):
    wss_dist = WSS(ref_wav, deg_wav, srate, device)
    
    llr_mean = LLR(ref_wav, deg_wav, srate, device)
    
    ref_spec = take_mag(stft(ref_wav.unsqueeze(0).permute(1, 0, 2)))
    deg_spec = take_mag(stft(deg_wav.unsqueeze(0).permute(1, 0, 2)))
    pmsqe_raw = 1 / torch.mean(PMSQE(deg_spec, ref_spec))
    
    return -(1.594 + 0.805 * pmsqe_raw - 0.512 * llr_mean - 0.007 * wss_dist)

def CompositeEval(ref_wav, deg_wav, srate, device=torch.device('cpu'), log_all=False):
    """
    Returns [sig, bak, ovl]
    """
    
    # Equalize the length of ref_wav and deg_wav
    ref_wav, deg_wav = preprocessing(ref_wav, deg_wav)

    # Compute WSS measure
    wss_dist = WSS(ref_wav, deg_wav, srate, device)

    # Compute LLR measure
    llr_mean = LLR(ref_wav, deg_wav, srate, device)

    # Compute the SSNR
    segSNR = SSNR(ref_wav, deg_wav, srate, device)

    # Compute the PMSQE
    # To compute the STFT, first add a dimension of 1 to get shape (batch size, channels, time)
    ref_spec = take_mag(stft(ref_wav.unsqueeze(0).permute(1, 0, 2)))
    deg_spec = take_mag(stft(deg_wav.unsqueeze(0).permute(1, 0, 2)))
    pmsqe_raw = 1 / torch.mean(PMSQE(deg_spec, ref_spec))

    # Used to restrict Csig, Cbak, and Covl to the range (1-5).
    '''
    def trim_mos(val):
        return min(max(val, torch.tensor(1, dtype=torch.float32, requires_grad=True, device=device)), torch.tensor(5, dtype=torch.float32, requires_grad=True, device=device))
    '''

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pmsqe_raw - 0.009 * wss_dist
#    Csig = -trim_mos(Csig)
    Cbak = 1.634 + 0.478 * pmsqe_raw - 0.007 * wss_dist + 0.063 * -segSNR
#    Cbak = -trim_mos(Cbak)
    Covl = 1.594 + 0.805 * pmsqe_raw - 0.512 * llr_mean - 0.007 * wss_dist
#    Covl = -trim_mos(Covl)

    if log_all:
        return Csig, Cbak, Covl, pmsqe_raw, segSNR, llr_mean, wss_dist
    else:
        return Csig, Cbak, Covl

"""
[1] P. C. Loizou, Speech Enhancement: Theory and Practice, 2nd ed.
    Boca Raton, FL, USA: CRC Press, Inc., 2013.
"""
