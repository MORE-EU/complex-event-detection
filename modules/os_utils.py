import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy import signal
import math as mt
import scipy.linalg as sl
import scipy.optimize as opt
import statsmodels.api as sm


def estimate_freq(series):
    """
    Estimate a principal frequency component using FFT.

    Parameters:
        series (array-like): The input signal to estimate.

    Returns:
        freq: estimated frequency
    """
    # Nearest size with power of 2
    size = 2 ** np.ceil(np.log2(2*len(series) - 1)).astype('int')
    # Variance
    var = np.var(series)
    # Normalized data
    ndata = series - np.mean(series)
    # Compute the FFT
    fft = np.fft.fft(ndata, size)
    # Get the power spectrum
    pwr = np.abs(fft) ** 2
    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = np.fft.ifft(pwr).real / var / len(series)
    peaks = signal.find_peaks(acorr[:len(series)])
    #### does it take the highest?
    if len(peaks[0]) == 0:
        return np.nan
    else:
        peak_idx = peaks[0][0]
        freq = 1/(series.index[peak_idx]-series.index[0]).total_seconds()
        return freq

def fit_sin_simple(df):
    """
    Estimate a signal using an FFT-based method.

    Parameters:
        df (array-like): The input signal to estimate.
        

    Returns:
        A tuple (data_fit, y_norm) containing the estimated (normalized) signal and the normalized signal.
    """
    y = df.values
    index = (df.index - min(df.index)).total_seconds()
    var = np.var(y)
    # Normalized data
    y_norm = (y - np.mean(y))/var
    acorr = sm.tsa.acf(y_norm, nlags = len(y_norm)-1)
    peaks = signal.find_peaks(acorr[:len(y_norm)])
    try:
        peak_idx = peaks[0][0]
        est_freq = 1/(df.index[peak_idx]-df.index[0]).total_seconds()
    except:
        est_freq = 1
    
    #find amplitude
    optimize_func = lambda x: (x[0]*np.sin(2*np.pi*index) - y_norm)
    #find phase
    yf = np.fft.fft(y_norm)
    T = 1/30
    freq = np.fft.fftfreq(y.size, d=T)
    ind, = np.where(np.isclose(freq, est_freq, atol=1/(T*len(y))))
    est_phase = np.angle(yf[ind[0]])
    est_amp = np.sqrt(2)*np.std(y_norm)
    data_fit = est_amp*np.sin(2*np.pi*est_freq*index+est_phase) 
    return data_fit, y_norm
    
def fit_sin_lsm(df):
    """
    Estimate a signal using the non-linear least squared method.

    Parameters:
        df (array-like): The input signal to estimate.
        

    Returns:
        A tuple (data_fit, y_norm) containing the estimated (normalized) signal and the normalized signal.
    """
    y = df.values
    index = (df.index - min(df.index)).total_seconds()
    var = np.var(y)
    # Normalized data
    y_norm = (y - np.mean(y))/var
    guess_mean = np.mean(y_norm)
    acorr = sm.tsa.acf(y_norm, nlags = len(y_norm)-1)
    peaks = signal.find_peaks(acorr[:len(y_norm)])
    try:
        peak_idx = peaks[0][0]
        guess_freq = 1/(df.index[peak_idx]-df.index[0]).total_seconds()
    except:
        guess_freq = 1
    #guess_amp = np.mean(y_norm)
    guess_amp = np.sqrt(2)*np.std(y_norm)
 
    #find amplitude
    optimize_func = lambda x: (x[0]*np.sin(2*np.pi*index) - y_norm)

    #find phase     
    yf = np.fft.fft(y_norm)
    T = 1/30
    freq = np.fft.fftfreq(y.size, d=T)
    try:
        ind, = np.where(np.isclose(freq, guess_freq, atol=1/(T*len(y))))
        guess_phase = np.angle(yf[ind[0]])
    except:
        guess_phase = 0
    
    
    #optimize_func = lambda x: np.abs(x[0]*np.sin(2*np.pi*x[1]*index+x[2]) - y_norm)
    #est_amp, est_freq, est_phase = opt.leastsq(optimize_func, [guess_amp, guess_freq, guess_phase], col_deriv = 1)[0]
    est_freq = guess_freq
    optimize_func = lambda x: np.abs(x[0]*np.sin(2*np.pi*index+x[1]) - y_norm)
    est_amp, est_phase = opt.leastsq(optimize_func, [guess_amp, guess_phase], col_deriv = 1)[0]
    data_fit = est_amp*np.sin(2*np.pi*est_freq*index+est_phase) 
    return data_fit, y_norm    


def prony_method(signal, p):
    """
    Estimate a signal using the Prony method.

    Parameters:
        signal (array-like): The input signal to estimate.
        p (int): The number of poles to use for the Prony method.

    Returns:
        A tuple (signal_hat, poles) containing the estimated signal and the estimated poles.
    """
    N = len(signal)

    # Construct the Hankel matrix
    H = np.zeros((N-p, p))
    for i in range(N-p):
        for j in range(p):
            H[i, p-1-j] = signal[i+j]
            
    
    # Solve the Prony system of equations
    b = -signal[p:N]
    
    #a = np.linalg.solve(H, b)
    a = np.linalg.lstsq(H, b)[0]
    
    a = np.concatenate([[1],a])
    # Compute the estimated poles
    z = np.roots(a)
    Z = np.zeros([p,p],dtype=complex)
   
    for i in range(p):
        for j in range(p):
            Z[i,j]=z[j]**i
            
    
    #h=np.linalg.solve(Z,signal[:p])
    h = np.linalg.lstsq(Z,signal[:p])[0]
    
    # Compute the estimated signal
    signal_hat = np.zeros(N)
    for i in range(N):
        signal_hat[i] = np.real(np.sum(h*(z**i)))

    return signal_hat, z


#following is based on https://github.com/Tan-0321/PronyMethod/blob/main/PronyMethod.py
def MPM(phi,p):    
    """
    Estimate a signal using the Matrix Pencil method.

    Parameters:
        phi (array-like): The input signal to estimate.
        p (int): The pencil parameter.

    Returns:
        signal_hat containing the estimated signal.
    """
    end=len(phi)-1
    Y=sl.hankel(phi[:end-p],phi[end-p:])
    Y1=Y[:,:-1]
    Y2=Y[:,1:]
    Y1p=np.linalg.pinv(Y1)
    EV=np.linalg.eigvals(np.dot(Y1p, Y2))
    EL=len(EV)
    
    #complex residues (amplitudes and angle phases)as Prony's method
    
    Z=np.zeros([EL,EL],dtype=complex)
    rZ=np.empty([EL,EL])
    iZ=np.empty([EL,EL])
    for i in range(EL):
        for j in range(EL):
            Z[i,j]=EV[j]**i
            rZ[i,j]=Z[i,j].real
            iZ[i,j]=Z[i,j].imag
    
    h=np.linalg.solve(Z,phi[0:EL])
    theta=np.empty([EL])
    amp=abs(h)
    for i in range(EL):
        theta[i]=mt.atan2(h[i].imag, h[i].real)
        
    # Compute the estimated signal
    signal_hat = np.zeros(len(phi))
    for i in range(len(phi)):
        signal_hat[i] = np.real(np.sum(h*(EV**i)))
        
    #answer=np.c_[amp,theta,alpha,frequency]
    return signal_hat



