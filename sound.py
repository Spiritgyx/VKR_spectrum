"""
:author: Mardanov Rinat Ildarovich
:email: gyxepm@gmail.com
"""
import pyaudio
import struct
import numpy as np
from scipy.fftpack import rfftfreq, rfft, irfft
import threading as TH
import sys, time, os, json
import wave


CHUNK = 512


def add(a: int, b: int,
        MIN: int = np.iinfo(np.int16).min, MAX: int = np.iinfo(np.int16).max):
    """
    Return sum in interval [MIN, MAX]\n
    :param int a: Value a
    :param int b: Value b
    :param int MIN: Minimum sum value
    :param int MAX: Maximum sum value
    :return: int
    """
    return min(a + b, MAX) if a + b >= MAX else max(a + b, MIN)


def gen_noise(sound: np.ndarray, level: int, ch: int=1) -> np.ndarray:
    """
    Generate noise to wave {sound} with {level} and return additive noised wave\n
    :param np.ndarray sound:
    :param int level:
    :return: np.ndarray
    """
    res = np.empty((0, CHUNK, 2), dtype=np.int16)
    # sound_ = np.transpose(sound, axes=(0, 2, 1))
    for c in sound:
        res = np.concatenate((
            res,
            c.reshape((1, CHUNK, 2)) +
            np.random.randint(-level, level, (1, CHUNK, 2), dtype=np.int16)
        ), axis=0)
        pass
    return res


class Audio:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 16000
        self.CHUNK = CHUNK
        self.AUD = np.iinfo(np.uint16).max

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.soundCallback
        )
        # (LIST OF FILES, LIST OF SAMPLES, LIST OF PIKS [2048], LIST OF CHANNELS [2])
        self.wf_datas = []
        # (LIST OF FILES, LIST OF SAMPLES, LIST OF SPEC PIKS [2048])
        self.spec_datas = []
        # (LIST OF SAMPLES, LIST OF PIKS [2048], LIST OF CHANNELS [2])
        self.play_data = np.array([], dtype=np.int16)
        self.play_ind = 0
        # (LIST OF FILES, DICTS OF PARAMS {'name': '..._i.spec', 'count': count_specs}
        # _0 - is TARGET specs; _1, _2, ... - is FROM specs
        self.spec_params = []
        self.spec_bytes = {}

    def soundCallback(self, in_data, frame_count, time_info, status):
        # data: np.array
        # data = []
        try:
            L = len(self.play_data)
            if L > 0 and self.play_ind < L:
                # print(len(self.play_data), self.play_data.shape)
                # data, self.play_data = self.play_data[0], np.delete(self.play_data, 0, axis=0)
                data = self.play_data[self.play_ind].tobytes()
                # print(len(data), data[:4])
                self.play_ind += 1
                # print(len(self.play_data), self.play_data.shape)
                # del self.play_data[0]
            else:
                data = np.array([[0, 0]] * self.CHUNK, dtype=np.int16)
                time.sleep(0.005)
        except:
            ex_info = sys.exc_info()
            print('except callback\n', str(ex_info))
            data = np.array([[0, 0]] * self.CHUNK, dtype=np.int16)
        return (data, pyaudio.paContinue)

    def readFromFile(self, path: str, ch: int = 0):
        if os.path.exists(path):
            if path.endswith('wav'):
                try:
                    f = wave.open(path, mode='rb')
                    nframes = f.getnframes()
                    nchannels = f.getnchannels()
                    print('debug\npath: %s\nFRATE: %d\nCHANNELS: %d\nFRAMES: %d' % \
                          (path, f.getframerate(), nchannels, nframes))
                    # print(nframes//self.CHUNK+int(nframes%self.CHUNK>0))
                    N = nframes // self.CHUNK + int(nframes % self.CHUNK > 0)
                    wf_data = np.empty((1, self.CHUNK, 2), dtype=np.int16)
                    # print(wf_data)
                    '''
                    for i in range(N):
                        data = f.readframes(self.CHUNK)
                        if i == N - 1 and len(data) < self.CHUNK * nchannels * 2:
                            data = data + b'\x00' * (self.CHUNK * nchannels * 2 - len(data))
                        # print(i, len(data), data[:8])
                        d1_data = np.fromstring(data, dtype=np.int16)
                        d2_data = np.reshape(d1_data, (-1, 2))
                        # print(self.AUD//2)
                        # res_data = np.array(d2_data, dtype=np.int16)
                        # print(i, type(res_data), res_data.dtype, len(res_data), res_data[:2])
                        #  res_data[0], d2_data[0])
                        # print(res_data.shape, wf_data.shape)
                        wf_data = np.concatenate((
                            wf_data,
                            d2_data.reshape((1, 2048, 2))
                        ), axis=0)
                        if i == 0:
                            wf_data = wf_data[1:]
                        # print(len(wf_data))
                    '''
                    # TEST new optimize
                    data = f.readframes(nframes)
                    data = data + b'\x00' * ((-len(data)) % (self.CHUNK * nchannels * 2))
                    data = np.fromstring(data, dtype=np.int16)
                    data = data.reshape((len(data)//(self.CHUNK*nchannels), self.CHUNK, nchannels))
                    wf_data = data
                    # data = f.readframes(self.CHUNK)
                    f.close()
                    # print(len(self.wf_datas), self.wf_datas.shape)
                    '''self.wf_datas = np.concatenate((
                        self.wf_datas.reshape((len(self.wf_datas), len(wf_data), 2048, 2)),
                        wf_data.reshape((1, len(wf_data), 2048, 2))
                    ), axis=0)'''
                    self.wf_datas.append(wf_data.copy())
                    return True
                    # print(self.wf_datas.shape, len(self.wf_datas), len(self.wf_datas[0]), len(self.wf_datas[0][0]))
                except:
                    ex_info = sys.exc_info()
                    print('except wav\n', str(ex_info))
                    return False
                    pass
            elif path.endswith('spec'):
                try:
                    with open(path, mode='rb') as f:
                        data = f.read(self.CHUNK * 8)
                        L = 0 if len(self.spec_datas) == 0 else len(self.spec_datas[0])
                        # i = 0
                        d = np.empty((1, 1, self.CHUNK), dtype=np.float64)
                        while data:
                            s = np.array(
                                struct.unpack(str(self.CHUNK) + 'd', data),
                                dtype=np.float64
                            ).reshape((1, 1, self.CHUNK))
                            d = np.concatenate((d, s), axis=1)
                            # if i == 0:
                            data = f.read(self.CHUNK * 8)
                            # i += 1
                        d = np.delete(d, 0, axis=1)
                        # np.concatenate((self.spec_datas, d), axis=0)
                        self.spec_datas.append(d.copy())
                    return True
                    pass
                except:
                    ex_info = sys.exc_info()
                    print('except spec\n', str(ex_info))
                    return False
                    pass

    def specToWav(self, spec_data: np.ndarray):
        """
        Transform Spectrum to Wave signal
        :param np.ndarray spec_data:
        :return: (np.ndarray) Wave signal
        """
        wf_data = np.empty((0, 2, self.CHUNK), dtype=np.int16)
        for spec in spec_data:
            signal = irfft(spec).astype(dtype=np.int16)
            wf_data = np.concatenate((
                wf_data,
                np.array([signal.copy(), signal.copy()], dtype=np.int16).reshape((1, 2, self.CHUNK))
            ), axis=0)
            pass
        wf_data = np.transpose(wf_data, axes=(0, 2, 1))
        return wf_data.copy()
        pass

    def wavToSpec(self, wf_data: np.ndarray, channel: int = 0):
        """
        Transform Wave signal to Spectrum
        :param np.ndarray wf_data:
        :param bool channel: Get False for Left channel or True for Right channel
        :return: (np.ndarray) Spectrum
        """
        # wf_data_ = np.transpose(wf_data, axes=(0, 2, 1))
        wf_data_ = wf_data[:, :, channel]
        # print('W2S', wf_data_.shape)
        # spec_data = np.array([rfft(wf_data_[i]) for i in range(len(wf_data_))], dtype=np.float64)
        # print(spec_data.shape)
        spec_data = np.empty((0, self.CHUNK), dtype=np.float64)
        spec_full = []
        # print(rfft(wf_data_[0]).shape)
        i = 0
        for c in wf_data_:
            # print(len(c), type(c), c[:4])
            # if i%500 == 0:
            #     print(i, len(spec_full))
            spec = np.array(rfft(c), dtype=spec_data.dtype).reshape((1, self.CHUNK))
            spec_full.append(spec)
            # print(len(spec), type(spec), spec[:4])
            # print(len(spec_data), type(spec_data), spec_data[:4])
            #spec_data = np.concatenate(
            #    (spec_data,
            #     spec), axis=0
            #)
            # print('spec_data comp', spec_data.)
            i += 1
        # print('end spec full')
        spec_data = np.array(spec_full, dtype=np.float64).reshape(wf_data_.shape)
        # print('spec data', spec_data.shape)
        # spec_data = np.delete(spec_data, 0)
        '''
        spec_data = np.empty((0, 2048), dtype=np.float64)
        for c in wf_data_:
            signal = c[channel]
            # print(len(signal), signal[:4])
            spec = np.array([rfft(signal)], dtype=np.float64)
            spec_data = np.concatenate((spec_data, spec), axis=0)
        '''
        return spec_data

    def readWav(self):
        pass

    def readSpec(self):
        pass
