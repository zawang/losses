All the loss functions (aside from PMSQE) in loss_functions_composite.py are slightly modified versions of those used in SEGAN [1].

PMSQE_asteroid contains a slightly modified version of the PMSQE implementation used in the Pytorch-based audio source separation toolkit Asteroid [2].

PMSQE_asteroid explanation:
	- The "bark_matrix_Xk.mat" matlab files contain the matrices required for Bark-spectrum transformation (8 and 16 kHz).
	- To handle 16 and 8 kHz files, PMSQE uses a 512 and 256 length DFT respectively. SingleSrcPMSQE's forward function computes the PMSQE between each corresponding sample in est_targets and targets. For example, if the batch size is 4, then the tensor returned will have shape (4,).
	- The Filterbank class is a base filterbank class. STFTB is a subclass of Filterbank. We can think of the STFT as implementing a uniform filter bank, i.e. spectral samples are uniformly spaced and correspond to equal bandwidths. Encoder turns a Filterbank into an encoder. Specifically, Encoder's forward function will 1D convolve an input tensor with the filters from a Filterbank.

[1] https://github.com/santi-pdp/segan_pytorch/blob/master/segan/utils.py (downloaded 3/3/20)
[2] https://github.com/mpariente/asteroid (downloaded 7/13/20)