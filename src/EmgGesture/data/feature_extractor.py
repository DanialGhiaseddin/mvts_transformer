import numpy as np
import pywt


def rms(data):
    return np.sqrt(np.mean(data ** 2, axis=0))


def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data, axis=0)), axis=0)


def zero_crossing(data):
    return np.argmax(np.diff(np.sign(data), axis=0), axis=0)


def wavelets_transform(raw_signal, axis=0, max_level=5):
    wavelets = dict()
    wavelets['A0'] = raw_signal
    for level in range(1, max_level + 1):
        wavelets[f'A{level}'], wavelets[f'D{level}'] = pywt.dwt(wavelets[f'A{level - 1}'], 'db4', axis=axis)

    # A0 to A4 should no longer be kept in the dictionary
    for level in range(0, max_level):
        del wavelets[f"A{level}"]
    return wavelets


def wavelet_based_function(func):
    def wrapper(*args):
        band = (func.__name__.split('_')[-1]).upper()


def wrms_d1(raw_signal):
    band_energy = wavelets_transform(raw_signal, 0, 5)['D1']
    return rms(band_energy)


def wrms_d2(raw_signal):
    band_energy = wavelets_transform(raw_signal, 0, 5)['D2']
    return rms(band_energy)


def wrms_d3(raw_signal):
    band_energy = wavelets_transform(raw_signal, 0, 5)['D3']
    return rms(band_energy)


def wrms_d4(raw_signal):
    band_energy = wavelets_transform(raw_signal, 0, 5)['D4']
    return rms(band_energy)


def wrms_d5(raw_signal):
    band_energy = wavelets_transform(raw_signal, 0, 5)['D5']
    return rms(band_energy)


def wrms_a5(raw_signal):
    band_energy = wavelets_transform(raw_signal, 0, 5)['A5']
    return rms(band_energy)


def wre_d1(raw_signal):
    wlt_dict = wavelets_transform(raw_signal, 0, 5)
    energy_futures = None
    total_energy = 0
    for idx, key in enumerate(wlt_dict.keys()):
        band = wlt_dict[key]
        band_energy = np.sum(band * band, axis=0)
        total_energy += band_energy
        if key == 'D1':
            energy_futures = band_energy
    # energy_futures /= total_energy
    return energy_futures


def wre_d2(raw_signal):
    wlt_dict = wavelets_transform(raw_signal, 0, 5)
    energy_futures = None
    total_energy = 0
    for idx, key in enumerate(wlt_dict.keys()):
        band = wlt_dict[key]
        band_energy = np.sum(band * band, axis=0)
        total_energy += band_energy
        if key == 'D2':
            energy_futures = band_energy
    # energy_futures /= total_energy
    return energy_futures


def wre_d3(raw_signal):
    wlt_dict = wavelets_transform(raw_signal, 0, 5)
    energy_futures = None
    total_energy = 0
    for idx, key in enumerate(wlt_dict.keys()):
        band = wlt_dict[key]
        band_energy = np.sum(band * band, axis=0)
        total_energy += band_energy
        if key == 'D3':
            energy_futures = band_energy
    # energy_futures /= total_energy
    return energy_futures


def wre_d4(raw_signal):
    wlt_dict = wavelets_transform(raw_signal, 0, 5)
    energy_futures = None
    total_energy = 0
    for idx, key in enumerate(wlt_dict.keys()):
        band = wlt_dict[key]
        band_energy = np.sum(band * band, axis=0)
        total_energy += band_energy
        if key == 'D4':
            energy_futures = band_energy
    # energy_futures /= total_energy
    return energy_futures


def wre_d5(raw_signal):
    wlt_dict = wavelets_transform(raw_signal, 0, 5)
    energy_futures = None
    total_energy = 0
    for idx, key in enumerate(wlt_dict.keys()):
        band = wlt_dict[key]
        band_energy = np.sum(band * band, axis=0)
        total_energy += band_energy
        if key == 'D5':
            energy_futures = band_energy
    # energy_futures /= total_energy
    return energy_futures


def wre_a5(raw_signal):
    wlt_dict = wavelets_transform(raw_signal, 0, 5)
    energy_futures = None
    total_energy = 0
    for idx, key in enumerate(wlt_dict.keys()):
        band = wlt_dict[key]
        band_energy = np.sum(band * band, axis=0)
        total_energy += band_energy
        if key == 'A5':
            energy_futures = band_energy
    # energy_futures /= total_energy
    return energy_futures


def log_am_fft_power(signal):
    return np.mean(np.log(np.abs(np.fft.fft(np.abs(signal), axis=0)) ** 2))


def log_fft_power(signal):
    return np.mean(np.log(np.abs(np.fft.fft(signal, axis=0)) ** 2))
