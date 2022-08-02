"""
Functions to prepare training data.

This module provides functions that read data from PhysioNet data base,
transform data, filter data and augment data.

Copyright 2019, Juho Laitala
Licence: MIT, see LICENCE for more details.

"""
import numpy as np
from wfdb import rdsamp, rdann
from wfdb.processing import (
    resample_singlechan,
    find_local_peaks,
    correct_peaks,
    normalize_bound
)
from random import uniform


def get_beats(annotation):
    """
    Extract beat indices and types of the beats.

    Beat indices indicate location of the beat as samples from the
    beg of the signal. Beat types are standard character
    annotations used by the PhysioNet.

    Parameters
    ----------
    annotation : wfdbdb.io.annotation.Annotation
        wfdbdb annotation object

    Returns
    -------
    beats : array
        beat locations (samples from the beg of signal)
    symbols : array
        beat symbols (types of beats)

    """
    # All beat annotations
    beat_annotations = ['N', 'L', 'R', 'B', 'A',
                        'a', 'e', 'J', 'V', 'r',
                        'F', 'S', 'j', 'n', 'E',
                        '/', 'Q', 'f', '?']

    # Get indices and symbols of the beat annotations
    indices = np.isin(annotation.symbol, beat_annotations)
    symbols = np.asarray(annotation.symbol)[indices]
    beats = annotation.sample[indices]

    return beats, symbols


def data_from_records(records, channel, db):
    """
    Extract ECG, beat locations and beat types from Physionet database.

    Takes a list of record names, ECG channel index and name of the
    PhysioNet data base. Tested only with db == 'mitdb'.

    Parameters
    ----------
    records : list
        list of file paths to the wfdbdb-records
    channel : int
        ECG channel that is wanted from each record
    db : string
        Name of the PhysioNet ECG database

    Returns
    -------
    signals : list
        list of single channel ECG records stored as numpy arrays
    beat_locations : list
        list of numpy arrays where each array stores beat locations as
        samples from the beg of one resampled single channel
        ECG recording
    beat_types : list
        list of numpy arrays where each array stores the information of
        the beat types for the corresponding array in beat_locations

    """
    signals = []
    beat_locations = []
    beat_types = []

    for record in records:
        print('processing record: ', record)
        signal = (rdsamp(record, pn_dir=db))
        signal_fs = signal[1]['fs']
        annotation = rdann(record, 'atr', pn_dir=db)

        # resample to 250 Hz
        signal, annotation = resample_singlechan(
                                signal[0][:, channel],
                                annotation,
                                fs=signal_fs,
                                fs_target=250)

        beat_loc, beat_type = get_beats(annotation)

        signals.append(signal)
        beat_locations.append(beat_loc)
        beat_types.append(beat_type)

    return signals, beat_locations, beat_types


def fix_labels(signals, beats, labels):
    """
    Change labeling of the normal beats.

    Beat index of some normal beats doesn't occur at the local maxima
    of the ECG signal in MIT-BIH Arrhytmia database. Function checks if
    beat index occurs within 5 samples from the local maxima. If this is
    not true, beat labeling is changed to -1.

    Parameters
    ----------
    signals : list
        List of ECG signals as numpy arrays
    beats : list
        List of numpy arrays that store beat locations
    labels : list
        List of numpy arrays that store beat types

    Returns
    -------
    fixed_labels : list
        List of numpy arrays where -1 has been added for beats that are
        not located in local maxima

    """
    fixed_labels = []
    for s, b, l in zip(signals, beats, labels):

        # Find local maximas
        localmax = find_local_peaks(sig=s, radius=5)
        localmax = correct_peaks(sig=s,
                                 peak_inds=localmax,
                                 search_radius=5,
                                 smooth_window_size=20,
                                 peak_dir='up')

        # Make sure that beat is also in local maxima
        fixed_p = correct_peaks(sig=s,
                                peak_inds=b,
                                search_radius=5,
                                smooth_window_size=20,
                                peak_dir='up')

        # Check what beats are in local maximas
        beat_is_local_peak = np.isin(fixed_p, localmax)
        fixed_l = l

        # Add -1 if beat is not in local max
        fixed_l[~beat_is_local_peak] = -1
        fixed_labels.append(fixed_l)

    return fixed_labels


def create_sine(sampling_frequency, time_s, sine_frequency):
    """
    Create sine wave.

    Function creates sine wave of wanted frequency and duration on a
    given sampling frequency.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency used to sample the sine wave
    time_s : float
        Lenght of sine wave in seconds
    sine_frequency : float
        Frequency of sine wave

    Returns
    -------
    sine : array
        Sine wave

    """
    samples = np.arange(time_s * sampling_frequency) / sampling_frequency
    sine = np.sin(2 * np.pi * sine_frequency * samples)

    return sine


def get_noise(ma, bw, win_size):
    """
    Create noise that is typical in ambulatory ECG recordings.

    Creates win_size of noise by using muscle artifact, baseline
    wander, and mains interefence (60 Hz sine wave) noise. Windows from
    both ma and bw are randomly selected to
    maximize different noise combinations. Selected noise windows from
    all of the sources are multiplied by different random numbers to
    give variation to noise strengths. Mains interefence is always added
    to signal, while addition of other two noise sources varies.

    Parameters
    ----------
    ma : array
        Muscle artifact signal
    bw : array
        Baseline wander signal
    win_size : int
        Wanted noise length

    Returns
    -------
    noise : array
        Noise signal of given window size

    """
    # Get the slice of data
    beg = np.random.randint(ma.shape[0]-win_size)
    end = beg + win_size
    beg2 = np.random.randint(ma.shape[0]-win_size)
    end2 = beg2 + win_size

    # Get mains_frequency US 60 Hz (alter strenght by multiplying)
    mains = create_sine(250, int(win_size/250), 60)*uniform(0, 0.5)

    # Choose what noise to add
    mode = np.random.randint(3)

    # Add noise with different strengths
    ma_multip = uniform(0, 5)
    bw_multip = uniform(0, 10)

    # Add noise
    if mode == 0:
        noise = ma[beg:end]*ma_multip
    elif mode == 1:
        noise = bw[beg:end]*bw_multip
    else:
        noise = (ma[beg:end]*ma_multip)+(bw[beg2:end2]*bw_multip)

    return noise+mains


def ecg_generator(signals, peaks, labels, ma, bw, win_size, batch_size):
    """
    Generate ECG data with R-peak labels.

    Data generator that yields training data as batches. Every instance
    of training batch is composed as follows:
    1. Randomly select one ECG signal from given list of ECG signals
    2. Randomly select one window of given win_size from selected signal
    3. Check that window has at least one beat and that all beats are
       labled as normal
    4. Create label window corresponding the selected window
        -beats and four samples next to beats are labeled as 1 while
         rest of the samples are labeled as 0
    5. Normalize selected signal window from -1 to 1
    6. Add noise into signal window and normalize it again to (-1, 1)
    7. Add noisy signal and its labels to trainig batch
    8. Transform training batches to arrays of needed shape and yield
       training batch with corresponding labels when needed

    Parameters
    ----------
    signals : list
        List of ECG signals
    peaks : list
        List of peaks locations for the ECG signals
    labels : list
        List of labels (peak types) for the peaks
    ma : array
        Muscle artifact signal
    bw : array
        Baseline wander signal
    win_size : int
        Number of time steps in the training window
    batch_size : int
        Number of training examples in the batch

    Yields
    ------
    (X, y) : tuple
        Contains training samples with corresponding labels

    """
    while True:

        X = []
        y = []

        while len(X) < batch_size:
            random_sig_idx = np.random.randint(0, len(signals))
            random_sig = signals[random_sig_idx]
            p4sig = peaks[random_sig_idx]
            plabels = labels[random_sig_idx]

            # Select one window
            beg = np.random.randint(random_sig.shape[0]-win_size)
            end = beg + win_size

            # Select peaks that fall into selected window.
            # Buffer of 3 to the window edge is needed as labels are
            # inserted also next to point)
            p_in_win = p4sig[(p4sig >= beg+3) & (p4sig <= end-3)]-beg

            # Check that there is at least one peak in the window
            if p_in_win.shape[0] >= 1:

                # Select labels that fall into selected window
                lab_in_win = plabels[(p4sig >= beg+3) & (p4sig <= end-3)]

                # Check that every beat in the window is normal beat
                if np.all(lab_in_win == 1):

                    # Create labels for data window
                    window_labels = np.zeros(win_size)
                    np.put(window_labels, p_in_win, lab_in_win)

                    # Put labels also next to peak
                    np.put(window_labels, p_in_win+1, lab_in_win)
                    np.put(window_labels, p_in_win+2, lab_in_win)
                    np.put(window_labels, p_in_win-1, lab_in_win)
                    np.put(window_labels, p_in_win-2, lab_in_win)

                    # Select data for window and normalize it (-1, 1)
                    data_win = normalize_bound(random_sig[beg:end],
                                               lb=-1, ub=1)

                    # Add noise into data window and normalize it again
                    data_win = data_win + get_noise(ma, bw, win_size)
                    data_win = normalize_bound(data_win, lb=-1, ub=1)

                    X.append(data_win)
                    y.append(window_labels)

        X = np.asarray(X)
        y = np.asarray(y)

        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = y.reshape(y.shape[0], y.shape[1], 1).astype(int)

        yield (X, y)
