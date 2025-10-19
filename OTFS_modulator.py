import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.constants import speed_of_light
from scipy.signal.windows import dpss
from scipy.spatial import distance
import commpy as cp
# np.set_printoptions(threshold=np.inf)

class Simulator():

    def helperOTFSmod(self, x, pad_len, pad_type):
        M = x.shape[0]
        # Inverse Zak transform
        y = np.fft.ifft(x.T, axis=0).T / M
        # ISFFT to produce the TF grid output
        isfftout = np.fft.fft(y, axis=0)
        # Add cyclic prefix/zero padding according to pad_type
        if pad_type == 'CP':
            # Cyclic prefix
            y = np.vstack([y[-pad_len:], y])
            # print(y.shape)
            y = y.T.flatten()
        elif pad_type == 'ZP':
            # Zero padding
            y = np.vstack([y, np.zeros((pad_len, y.shape[1]))])
            # print(y.shape)
            y = y.T.flatten()
        elif pad_type == 'RZP':
            # Serialize then append OTFS symbol with zeros
            y = y.T.flatten()
            y = np.concatenate([y, np.zeros(pad_len)])
        elif pad_type == 'RCP':
            # Reduced cyclic prefix
            y = y.T.flatten()
            y = np.concatenate([y[-pad_len:], y])
        elif pad_type == 'NONE':
            y = y.T.flatten()
        else:
            raise ValueError('Invalid pad type')
        
        return y, isfftout


    def helperOTFSdemod(self, x, M, padlen, offset, padtype):
        if padtype not in ['CP', 'RCP', 'ZP', 'RZP', 'NONE']:
            raise ValueError('Invalid pad type')
            
        if padtype in ['CP', 'ZP']:
            # Full CP or Zero Padding
            N = len(x) // (M + padlen)
            assert np.isclose(N, round(N)), "M*N should be an integer"
            rx = x.reshape(N, M + padlen).T
            Y = rx[offset:M + offset, :]  # remove CPs
        elif padtype == 'NONE':
            # No CP/ZP
            N = len(x) // M
            assert np.isclose(N, round(N)), "M*N should be an integer"
            Y = x.reshape(M, N)
        elif padtype in ['RCP', 'RZP']:
            # Reduced CP or ZP
            N = (len(x) - padlen) // M
            assert np.isclose(N, round(N)), "M*N should be an integer"
            rx = x[offset:M * N + offset]  # remove CP
            Y = rx.reshape(M, N)
        
        # This code segment shows the SFFT/OFDM demod representation
        tfout = np.fft.fft(Y, axis=0)

        # This code segment shows the simpler Zak transform representation
        y = np.fft.fft(Y.T, axis=0).T * M

        return y, tfout


    def frequency_offset(self, x, sample_rate, offset):
        # Validate inputs
        if not isinstance(x, np.ndarray):
            raise TypeError("Input signal x must be a numpy array.")
        if not np.isscalar(sample_rate) or sample_rate <= 0:
            raise ValueError("Sample rate must be a positive scalar.")
        if not (np.isscalar(offset) or (isinstance(offset, np.ndarray) and offset.ndim == 1 and offset.size == x.shape[1])):
            raise ValueError("Offset must be a scalar or a 1D numpy array with the same number of columns as x.")
        
        # Cast sample_rate and offset to have same data type as input signal
        fs = np.array(sample_rate, dtype=x.dtype)
        freq_offset = np.array(offset, dtype=x.dtype)
        pi_val = np.array(np.pi, dtype=x.dtype)

        # Create vector of time samples
        t = (np.arange(x.shape[0]) / fs).T
        t = t.reshape((len(t),1))
        x = x.reshape((len(x),1))
        # For each column, apply the frequency offset
        saazdf = np.multiply(1j * 2 * pi_val * freq_offset, t)
        y = np.multiply(x, np.exp(saazdf))

        return y.flatten()


    def dopplerChannel(self, x, fs, chan_params):
        num_paths = len(chan_params['pathDelays'])
        max_path_delay = max(chan_params['pathDelays'])
        tx_out_size = len(x)
        y = np.zeros(tx_out_size + max_path_delay, dtype=complex)

        for k in range(num_paths):
            path_out = np.zeros(tx_out_size + max_path_delay, dtype=complex)
            # Apply Doppler shift
            path_shift = self.frequency_offset(x, fs, chan_params['pathDopplerFreqs'][k])
            # Apply delay and gain
            start_idx = chan_params['pathDelays'][k]
            path_out[chan_params['pathDelays'][k]:chan_params['pathDelays'][k] + tx_out_size] = path_shift * chan_params['pathGains'][k]
            y += path_out
        
        return y


    def getG(self, M, N, chanParams, padLen, padType):
        # Form time domain channel matrix from detected DD paths
        if padType in ['ZP', 'CP']:
            Meff = M + padLen  # account for subsymbol pad length in forming channel
            lmax = padLen  # max delay
        else:
            Meff = M
            lmax = max(chanParams['pathDelays'])  # max delay
        MN = Meff * N
        P = len(chanParams['pathDelays'])  # number of paths
        # print(P)
        # Form an array of channel responses for each path
        g = np.zeros((lmax + 1, MN), dtype=complex)
        # print(chanParams['pathDelays'])
        for p in range(P):
            gp = chanParams['pathGains'][p]
            # lp = chanParams['pathDelays'][p]
            if padType == 'ZP':
                lp = chanParams['pathDelays'][p]
            else:
                lp = chanParams['pathDelays'][p] - 10

            vp = chanParams['pathDopplers'][p]
            # For each DD path, compute the channel response.
            # Each path is a complex sinusoid at the Doppler frequency (kp)
            # shifted by a delay (lp) and scaled by the path gain (gp)
            # print(lp)
            # print(gp)
            # print(MN)
            # print(vp)
            # print('-----------------')
            g[lp, :] = g[lp, :] + gp * np.exp(1j * 2 * np.pi / MN * vp * (np.arange(MN) - lp))
        # Form the MN-by-MN channel matrix G
        G = np.zeros((MN, MN), dtype=complex)
        # Each DD path is a diagonal in G offset by its path delay l
        for l in np.unique(chanParams['pathDelays']):
            if padType == 'ZP':
                G += np.diag(g[l, l:], -l)
            else:
                l = l-10
                G += np.diag(g[l, l:], -l)
            
        return G


    def awgn(self, sig, SNR):
        x_watts = np.mean(np.abs(sig)** 2)
        db = 10 * np.log10(x_watts)
        noise_avg_db = db - SNR
        noise_avg_watts = 10 ** (noise_avg_db / 10 )
        noise_volts = np.random.normal(0, np.sqrt(noise_avg_watts), len(sig))
        y = sig + noise_volts

        return y


    def ofdm_modulate(self, symbols, num_subcarriers, cp_length):
        # Reshape input symbols to match the number of subcarriers
        symbols_reshaped = symbols.reshape((-1, num_subcarriers))
        
        # Perform IFFT
        ifft_data = np.fft.ifft(symbols_reshaped, n=num_subcarriers, axis=1)
        
        # Add cyclic prefix
        cyclic_prefix = ifft_data[:, -cp_length:]
        ofdm_symbols = np.hstack((cyclic_prefix, ifft_data))
        
        # Flatten the array for transmission
        ofdm_signal = ofdm_symbols.flatten()
        
        return ofdm_signal


    def ofdm_demodulate(self, ofdm_signal, num_subcarriers, cp_length):
        # Reshape the received signal into OFDM symbols
        num_ofdm_symbols = len(ofdm_signal) // (num_subcarriers + cp_length)
        ofdm_symbols = ofdm_signal.reshape((num_ofdm_symbols, num_subcarriers + cp_length))
        
        # Remove cyclic prefix
        symbols_no_cp = ofdm_symbols[:, cp_length:]
        
        # Perform FFT
        demodulated_data = np.fft.fft(symbols_no_cp, n=num_subcarriers, axis=1)
        
        # Flatten the array to retrieve the original symbols
        demodulated_symbols = demodulated_data.flatten()
        
        return demodulated_symbols


    def plot_constellation(self, symbols, ref, t, title='Constellation Diagram of '):
        plt.figure(figsize=(6, 6))
        plt.plot(symbols.real, symbols.imag, 'o', color= 'b',markersize=2)
        plt.plot(ref.real, ref.imag, 'o', color= 'r', markersize=4)
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
        plt.grid(True, which='both', linestyle='--', lw=0.5)
        plt.xlabel('In-Phase Amplitude')
        plt.ylabel('Quadrature Amplitude')
        plt.xticks([-4, -3, -2, -1, 1, 2, 3, 4])
        plt.yticks([-4, -3, -2, -1, 1, 2, 3, 4])
        plt.title(title+t)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()




    def __init__(self):
        # Constant for demos
        # 16QAM constellation
        self.ref = np.array([-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j, -1+3j, 
            3-3j, 3-1j, 3+1j, 3+3j, 1-3j, 1-1j, 1+1j, 1+3j])

        self.mapping_table ={(0,0,0,0) : -3-3j, (0,0,0,1) : -3-1j, (0,0,1,0) : -3+3j,
                            (0,0,1,1) : -3+1j, (0,1,0,0) : -1-3j, (0,1,0,1) : -1-1j,
                            (0,1,1,0) : -1+3j, (0,1,1,1) : -1+1j, (1,0,0,0) :  3-3j,
                            (1,0,0,1) :  3-1j, (1,0,1,0) :  3+3j, (1,0,1,1) :  3+1j,
                            (1,1,0,0) :  1-3j, (1,1,0,1) :  1-1j, (1,1,1,0) :  1+3j, (1,1,1,1) :  1+1j}
                        

        self.demapping_table = {v : k for k, v in self.mapping_table.items()}
        self.constellation = np.array([x for x in self.demapping_table.keys()])

        # Define constants
        self.M = 64  # number of subcarriers
        self.N = 32  # number of subsymbols/frame
        self.df = 15e3  # frequency bin spacing of LTE
        self.fc = 5e9  # carrier frequency in Hz
        self.padLen = 10  # padding length
        self.padType = 'ZP'
        self.M_mod = 16  # size of QAM constellation
        self.M_bits = 4
        self.fsamp = self.M * self.df
        self.Meff = self.M + self.padLen
        self.T = (self.M + self.padLen) / (self.M * self.df)
        
        

    def run(self, SNR_input, speed_input, mode, num_n, height):
        self.N = num_n  # number of subsymbols
        self.numSamps = self.Meff * self.N
        self.one_doppler_tap = 1 / (self.N * self.T)

        self.SNRdB = SNR_input  # signal-to-noise ratio in dB
        self.speed = speed_input # km/h
        self.height = height

        self.spped_h = self.speed * (1000 / 3600) # m/s
        self.Doppler_vel = (self.spped_h * self.fc) / 299792458
        self.Doppler_tap = self.Doppler_vel/self.one_doppler_tap
        self.Es = 10
        self.n0 = self.Es / (10 ** (self.SNRdB / 10))

        # Data generation
        self.Xgrid = np.zeros((64, self.N), dtype=complex)
        self.Xdata = np.random.randint(0, 2, (4 * 64, self.N))
        self.Xdata_flatten = np.zeros((64*self.N, 4), dtype=int)
        self.ccc = 0
        self.rrr = 0
        # Convert the new array
        for self.jjj in range(self.N):
            for self.iii in range(256):
                self.colIdx = self.iii % 4
                self.Xdata_flatten[self.rrr, self.colIdx] = self.Xdata[self.iii, self.jjj]
                self.ccc += 1
                if self.ccc == 4:
                    self.ccc = 0
                    self.rrr += 1
        self.qew = cp.QAMModem(self.M_mod).modulate(self.Xdata_flatten.flatten())
        self.Xgrid[:self.M, :] = self.qew.reshape(self.M, self.N)
        # Pilot generation and grid population
        self.pilotBin = self.N // 2
        self.Pdd = np.zeros((self.M, self.N), dtype=complex)
        self.Pdd[0, self.pilotBin] = -3 + 3j

        # OTFS modulation
        self.txOut = self.helperOTFSmod(self.Pdd, self.padLen, self.padType)
        # Channel parameters
        if self.height == 100:
            self.chanParams = {'pathDelays': [0, 5, 8], 'pathGains': [1, 0.9, 0.8], 'pathDopplers': [0, -0.6, 1]}

        elif self.height == 1000:
            self.chanParams = {'pathDelays': [0, 5, 8], 'pathGains': [1, 0.7, 0.5], 'pathDopplers': [0, -0.6, 1]}

        elif self.height == 10000:
            self.chanParams = {'pathDelays': [0, 5, 8], 'pathGains': [1, 0.5, 0.2], 'pathDopplers': [0, -0.6, 1]}

        self.chanParams = {'pathDelays': [0, 5, 8], 'pathGains': [1, 0.7, 0.5], 'pathDopplers': [0, -0.6, 1]}
        for i,D in enumerate(self.chanParams['pathDopplers']):
            self.chanParams['pathDopplers'][i] = int(self.chanParams['pathDopplers'][i] * (0.043 * (self.Doppler_tap ** 2) - 0.015 * self.Doppler_tap + 5))
        self.chanParams['pathDopplerFreqs'] = np.array(self.chanParams['pathDopplers']) * (1 / (self.N * self.T))


        if mode == 'OFDM':
            return self.OFDM()
        elif mode == 'OTFS':
            return self.OTFS()
        else:
            print('Wrong Mode')




    def OFDM(self):
        # Transmit pilots over all subcarriers and symbols to sound the channel
        pilotSymbols = -3 + 3j
        pilotGrid = np.tile(pilotSymbols, (self.M, self.N))
        txOut = self.ofdm_modulate(pilotGrid, self.M, self.padLen)  
        dopplerOut = self.dopplerChannel(txOut, self.fsamp, self.chanParams)
        chOut = cp.awgn(dopplerOut, self.SNRdB)
        Yofdm = self.ofdm_demodulate(chOut[0:(self.M+self.padLen)*self.N],self.M,self.padLen) 
        Yofdm = Yofdm.reshape(self.M,self.N)
        Hofdm = Yofdm * np.conj(self.Pdd[0, self.pilotBin]) / (np.abs(self.Pdd[0, self.pilotBin]) ** 2 + self.n0)
        Hofdm = Hofdm.reshape(self.M,self.N)

        # OFDM
        # Transmit data over the same channel and use channel estimates to equalize
        txOut_ofdm = self.ofdm_modulate(self.Xgrid, self.M, self.padLen) 
        dopplerOut_ofdm = self.dopplerChannel(txOut_ofdm, self.fsamp, self.chanParams)
        chOut_ofdm = cp.awgn(dopplerOut_ofdm, self.SNRdB)
        rxWindow_ofdm = chOut_ofdm[:(self.M + self.padLen) * self.N]
        Yofdm_ofdm = self.ofdm_demodulate(rxWindow_ofdm, self.M, self.padLen)
        Yofdm_ofdm = Yofdm_ofdm.reshape(self.M, self.N)
        Xhat_ofdm = np.conj(Hofdm) * Yofdm_ofdm / (np.abs(Hofdm) ** 2 + self.n0)
        Xhat_ofdm = Xhat_ofdm.reshape(self.M, self.N)

        # Demodulate received data
        XhatDataSymbols_ofdm = cp.QAMModem(self.M_mod).demodulate(Xhat_ofdm.flatten(),'hard')
        # # Calculate BER
        XdataReshaped = self.Xdata_flatten.flatten()
        ber = np.sum(XdataReshaped != XhatDataSymbols_ofdm) / XdataReshaped.size
        # print(f'OFDM BER with single-tap equalizer = {ber:.3f}')

        dists_ofdm = abs(Xhat_ofdm.flatten().reshape((-1,1)) - self.constellation.reshape((1,-1)))
        const_index_ofdm = dists_ofdm.argmin(axis=1)
        hardDecision_ofdm = self.constellation[const_index_ofdm]
        D_ofdm = distance.euclidean(Xhat_ofdm.flatten(), hardDecision_ofdm)
        return ber, D_ofdm/len(Xhat_ofdm.flatten())
        # print(f'Average Euclidean distance of OFDM constellation matrix = {D_ofdm/len(Xhat_ofdm.flatten()):.3f}')

        # self.plot_constellation(Xhat_ofdm, self.ref, 'OFDM over High-Doppler Channel')


    def OTFS(self):
        # Send the OTFS modulated signal through the channel
        dopplerOut = self.dopplerChannel(self.txOut[0], self.fsamp, self.chanParams)
        # Add white Gaussian noise
        chOut = self.awgn(dopplerOut, self.SNRdB)
        # Get a sample window
        rxIn = chOut[:self.numSamps]
        # OTFS demodulation
        Ydd = self.helperOTFSdemod(rxIn, self.M, self.padLen, 0, self.padType)
        Hdd = Ydd[0] * np.conj(self.Pdd[0, self.pilotBin])
        Hdd = Hdd / (np.abs(self.Pdd[0, self.pilotBin]) ** 2 + self.n0)
        lp, vp = np.where(np.abs(Hdd) >= 0.05)
        chanEst = { 'pathGains': np.diag(Hdd[lp, vp]),
                    'pathDelays': lp ,
                    'pathDopplers': vp - self.pilotBin}
        chanEst['pathGains'] = np.diagonal(chanEst['pathGains'])

        # OTFS
        txOut_otfs = self.helperOTFSmod(self.Xgrid, self.padLen, self.padType)
        aa = np.array(txOut_otfs[0])
        # Add channel and noise
        dopplerOut_otfs = self.dopplerChannel(aa, self.fsamp, self.chanParams)
        chOut_otfs = cp.awgn(dopplerOut_otfs, self.SNRdB)
        # Form G matrix using channel estimates
        G = self.getG(self.M, self.N, chanEst, self.padLen, self.padType)
        rxWindow_otfs = chOut_otfs[:self.numSamps]
        y_otfs = inv(G.T @ G + self.n0 * np.eye(self.Meff * self.N)) @ (G.T @ rxWindow_otfs)
        Xhat_otfs = self.helperOTFSdemod(y_otfs, self.M, self.padLen, 0, self.padType)
        Xhat_otfs = Xhat_otfs[0].reshape(self.M, self.N)

        # Demodulate received data
        XhatDataSymbols_otfs = cp.QAMModem(self.M_mod).demodulate(Xhat_otfs.flatten(),'hard')
        # # Calculate BER
        XdataReshaped = self.Xdata_flatten.flatten()
        # print(XdataReshaped.shape)
        ber = np.sum(XdataReshaped != XhatDataSymbols_otfs) / XdataReshaped.size
        # print(f'OTFS BER with single-tap equalizer = {ber:.3f}')

        dists_otfs = abs(Xhat_otfs.flatten().reshape((-1,1)) - self.constellation.reshape((1,-1)))
        const_index_otfs = dists_otfs.argmin(axis=1)
        hardDecision_otfs = self.constellation[const_index_otfs]
        D_otfs = distance.euclidean(Xhat_otfs.flatten(), hardDecision_otfs)
        # print(f'Average Euclidean distance of OTFS constellation matrix = {D_otfs/len(Xhat_otfs.flatten()):.3f}')

        return [ber, D_otfs/len(Xhat_otfs.flatten())]
        # self.plot_constellation(Xhat_otfs, self.ref, 'OTFS over High-Doppler Channel')