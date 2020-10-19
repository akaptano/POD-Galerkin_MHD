"""
Spectral Proper Orthogonal Decomposition
-----------------------------------------

This module implements the Spectral Proper Orthogonal Decomposition class. The
present implementation corresponds to the batch algorithm originally proposed
in [1]. Note that a streaming algorithm has also been proposed in [2].

References
----------

[1] Towne, A., Schmidt, O. T. and Colonius, T. (2018). Spectral proper orthogonal
decomposition and its relationship to dynamic mode decomposition and resolvent
analysis. Journal of Fluid Mechanics, 847, 821-867.

[2] Schmidt, O. T. & Towne, A. (2019). An efficient streaming algorithm for
spectral proper orthogonal decomposition. Computer Physics Communications, 237,
98-109.

"""

# Author : Jean-Christophe Loiseau <jean-christophe.loiseau@ensam.eu>
# Date : April 2019
# License : ??

# --> Import NumPy.
import numpy as np

# --> Import utility functions from scikit-learn.
from sklearn.base import BaseEstimator, TransformerMixin

# --> Import utility functions from SciPy.
from scipy.linalg import svd
from scipy.signal.spectral import _fft_helper, _triage_segments
from scipy.fftpack import fftfreq


def detrend_func(d):
    return d


class SPOD(BaseEstimator, TransformerMixin):
    """Spectral proper orthogonal decomposition (SPOD).

    Linear dimensionality and spectral analysis using the spectral proper ortho-
    gonal decomposition of the data matrix.

    It uses the basic SVD solver from scipy.linalg. It additionaly relies on
    fast Fourier transforms using a set of helper functions from the
    scipy.signal module.

    Parameters
    ----------
    n_components : int (default is 2)
        Number of components to keep for each frequency.

    dt : float (default is 1.0)
        Sampling period of the snapshots used to compute SPOD.

    nperseg : int (default is 256)
        Length of each segment for the block estimation of the PSD.

    Attributes
    ----------
    mean_ : array, shape (n_dofs, 1)
        Empirical mean of the training data matrix (i.e. time-averaged mean flow).

    spod_modes : array, shape (n_dofs, n_components, n_freqs)
        SPOD modes representing the spatio-temporal coherent structures in the
        data. For each frequency, they are sorted according to the corresponding
        singular value distribution.

    modal_energy : array, shape (n_freqs, n_components)
        Portion of variance explained by each components for the various
        frequencies.

    freqs : array, shape (n_freqs)
        Frequencies computed during the power spectral density estimation.
    """

    def __init__(self, n_components=2, dt=1.0, nperseg=256):

        # --> Number of SPOD modes to keep per frequency.
        self.n_components = n_components

        # --> Sampling period of the snapshots.
        self.dt = dt

        # --> Lenght of the Hanning window.
        self.nperseg = nperseg

    def fit(self, X, y=None):
        """Fit the SPOD model with the training data in X.

        Parameters
        ----------
        X : array-like, shape (n_dofs, n_samples)
            Trainign data where n_samples is the number of samples and n_dofs is
            the number of degrees of freedom of the system.

        y : Ignored

        Returns
        -------
        self : object
            Return the instance itself.

        """
        self._fit(X)
        return self

    def _fit(self, X):
        """Compute the spectral proper orthogonal decomposition"""

        # --> Center the data.
        self.mean_ = np.mean(X, axis=1).reshape(-1, 1)
        X = X - self.mean_

        # --> Setup the Hamming window for the block-estimate of the PSD.
        win, nperseg = _triage_segments(
            "hamming",
            self.nperseg,
            input_length=X.shape[1]
        )

        winWeight = 1.0 / np.mean(win)

        nfft = nperseg

        # --> Overlap factor.
        noverlap = nperseg // 2

        # --> Frequency array.
        freqs = fftfreq(nfft, self.dt)

        # --> Compute the windowed FFTs.
        X_blk = winWeight/nfft * _fft_helper(
            X, win, detrend_func, nperseg, noverlap, nfft, "twosided"
        )

        print(np.shape(X_blk),nfft,nperseg,noverlap,np.shape(freqs))

        # --> Rescale for unitary transformation.
        X_blk *= np.sqrt(self.dt / X_blk.shape[1])

        # --> List to store the various SPOD modes.
        Psi, modal_energy = list(), list()

        # --> Loop through the frequencies.
        for i in range(len(freqs)):
            # --> Get the ensemble of Fourier realization for frequency i.
            Q = X_blk[:, :, i]

            # --> Compute the SPOD modes.
            U, S, _ = svd(Q, full_matrices=False)

            # --> Store the desired SPOD modes.
            Psi.append(U[:, :self.n_components])
            modal_energy.append(S[:self.n_components]**2)

        # -->
        self.spod_modes = np.moveaxis(np.asarray(Psi), [0, 1, 2], [2, 0, 1])
        self.modal_energy = np.asarray(modal_energy)
        self.freqs = freqs

        return np.asarray(Psi), np.asarray(modal_energy), freqs

    def transform(self, X):
        """Apply dimensionality on X.

        Parameters
        ----------
        X : array-like, shape (n_dofs, n_samples)
            Data to be embedded into the low-dimensional subspace spanned by
            the leading SPOD modes.

        Returns
        -------
        X_new ; array-like, shape (n_components, n_samples)
            Projection of the data matrix X into the low-dimensional subspace.

        NOTE : THIS IS STILL EXPERIMENTAL !

        """
        m = self.spod_modes.shape
        U = self.spod_modes.reshape((m[0], -1))

        X = X - self.mean_

        return pinv(U) @ X

    def inverse_transform(self, X):
        """Transform the data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_components, n_samples)
            Low-dimensional representation of the data in the POD subspace.

        Returns
        -------
        X_original : array-like, shape (n_dofs, n_samples)

        NOTE : THIS IS STILL EXPERIMENTAL!

        """

        m = self.spod_modes.shape
        U = self.spod_modes.reshape((m[0], -1))

        return U @ X + self.mean_
