"""
Functions to process model inputs and outputs.

This module provides functions that process ECG signals in the form
expected by the LSTM model and functions that extract R-peak locations
from the predictions of the LSTM model.

Copyright 2019, Juho Laitala
Licence: MIT, see LICENCE for more details.

"""
import numpy as np
from scipy.signal import resample_poly
from wfdb import processing
from tensorflow.keras.models import load_model
from functools import partial
from importlib.resources import path


class Detector:
    """
    Process model inputs and outputs.

    Contains methods to process model inputs and outputs. Methods are
    more general than methods in the subclass ECG_detector (e.g. split
    signal into overlapping windows), and thus, they are not exclusively
    for ECG signals.
    """

    def __init__(self, model, window_size, stride):
        """
        Initialize Detector.

        Parameters
        ----------
        model : str
            Path to the Keras model (h5 file)
        window_size : int
            Size of the moving window
        stride : int
            Stride used to move window
        """
        self.model = load_model(model)
        self.win_size = window_size
        self.stride = stride

    def _extract_windows(self, signal):
        """
        Extract overlapping data windows from the input signal.

        Function splits input signal into separate windows. Padding is
        added into both ends of the signal before splitting. First or
        value of the signal is used to pad beginning while last value
        is used to pad end.

        Parameters
        ----------
        signal : array
            Signal that is split into data windows.

        Returns
        -------
        win_idx : array
            Array with shape (data_windows*self.win_size,). This vector
            contains all original signal indices for the different data
            windows.

        data_windows : array
            Array with shape (data_windows, self.win_size, 1). Contains all of
            the extracted data windows.

        """
        pad_sig = np.pad(signal,
                         (self.win_size-self.stride, self.win_size),
                         mode='edge')

        # Lists of data windows and corresponding indices
        data_windows = []
        win_idx = []

        # Indices for padded signal
        pad_id = np.arange(pad_sig.shape[0])

        # Split into windows and save corresponding padded indices
        for win_id in range(0, len(pad_sig), self.stride):
            if win_id + self.win_size < len(pad_sig):
                data_windows.append(pad_sig[win_id:win_id+self.win_size])
                win_idx.append(pad_id[win_id:win_id+self.win_size])

        data_windows = np.asarray(data_windows)
        data_windows = data_windows.reshape(data_windows.shape[0],
                                            data_windows.shape[1], 1)
        win_idx = np.asarray(win_idx)
        win_idx = win_idx.reshape(win_idx.shape[0]*win_idx.shape[1])

        return win_idx, data_windows

    def _calculate_means(self, indices, values):
        """
        Calculate means of the values that have same index.

        Function calculates average from the values that have same
        index in the indices array.

        Parameters
        ----------
        indices : array
            Array of indices.
        values : array
            Value for every indice in the indices array.

        Returns
        -------
        mean_values : array
            Contains averages for the values that have the duplicate
            indices while rest of the values are unchanged.

        """
        assert(indices.shape == values.shape)

        # Combine indices with predictions
        comb = np.column_stack((indices, values))

        # Sort based on window indices and split when indice changes
        comb = comb[comb[:, 0].argsort()]
        split_on = np.where(np.diff(comb[:, 0]) != 0)[0]+1

        # Take mean from the values that have same index
        mean_values = [arr[:, 1].mean() for arr in np.split(comb, split_on)]
        mean_values = np.array(mean_values)

        return mean_values

    def _mean_preds(self, win_idx, preds, orig_len):
        """
        Calculate mean of overlapping predictions.

        Function takes window indices and corresponding predictions as
        input and then calculates mean for predictions. One mean value
        is calculated for every index of the original padded signal. At
        the end padding is removed so that just the predictions for
        every sample of the original signal remain.

        Parameters
        ----------
        win_idx : array
            Array of padded signal indices before splitting.
        preds : array
            Array that contain predictions for every data window.
        orig_len : int
            Lenght of the signal that was used to extract data windows.

        Returns
        -------
        pred_mean : int
            Predictions for every point for the original signal. Average
            prediction is calculated from overlapping predictions.

        """
        # flatten predictions from different windows into one vector
        preds = preds.reshape(preds.shape[0]*preds.shape[1])
        assert(preds.shape == win_idx.shape)

        pred_mean = self._calculate_means(indices=win_idx, values=preds)

        # Remove paddig
        pred_mean = pred_mean[int(self.win_size-self.stride):
                              (self.win_size-self.stride)+orig_len]

        return pred_mean


class ECG_detector(Detector):
    """
    R-peak detector.

    Detect R-peak locations from one channel electrocardiogram (ECG).
    Detects R-peak locations by using LSTM model.
    """

    def __init__(self, sampling_rate, model=None, stride=250,
                 window_size=1000, threshold=0.05):
        """
        Initialize ECG_cetector.

        Parameters
        ----------
        sampling_rate : int
            Sampling rate used in ECG signals.
        model : String
            Path to trained model as h5 file.
        stride : int
            Amount (step) to move the window. Must be 100, 200, 250 or 500.
        window_size : int
            Width of the moving window. Must be 1000 for now.
        threshold : float
            Threshold value used to do initial filtering of model
            predictions.
        """
        if stride not in [100, 200, 250, 500]:
            print('Unallowed stride chosen, setting stride to 250')
            stride = 250
        if window_size != 1000:
            print('Unallowed window size chosen, setting size to 1000')
            window_size = 1000

        if model is None:
            with path('ecg2rr', 'lstm.h5') as path_to_model:
                self.model_path = path_to_model
        else:
            self.model_path = model

        super().__init__(self.model_path, window_size, stride)
        self.iput_fs = sampling_rate
        self.threshold = threshold

        if sampling_rate == 250:
            self.resample = False
        else:
            self.resample = True

    def _filter_predictions(self, signal, preds):
        """
        Filter model predictions.

        Function filters model predictions by using following steps:
        1. selects only the predictions that are above the given
        probability threshold.
        2. Correct these predictions upwards with respect the given ECG
        3. Check if at least five points are corrected into the same
        location.
        4. If step 3 is true, then location is classified as an R-peak
        5. Calculate probability of location being an R-peak by taking
        mean of the probabilities from predictions in the same location.

        Aforementioned steps can be thought as an noise reducing measure as
        in original training data every R-peak was labeled with 5 points.

        Parameters
        ----------
        signal : array
            Same signal that was used with extract_windows function. It is
            used in correct_peaks function.
        preds : array
            Predictions for the sample points of the signal.

        Returns
        -------
        filtered_peaks : array
            locations of the filtered peaks.
        filtered_probs : array
            probability that filtered peak is an R-peak.

        """
        assert(signal.shape == preds.shape)

        # Select points probabilities and indices that are above
        # self.threshold
        above_thresh = preds[preds > self.threshold]
        above_threshold_idx = np.where(preds > self.threshold)[0]

        # Keep only points above self.threshold and correct them upwards
        correct_up = processing.correct_peaks(sig=signal,
                                              peak_inds=above_threshold_idx,
                                              search_radius=5,
                                              smooth_window_size=20,
                                              peak_dir='up')

        filtered_peaks = []
        filtered_probs = []

        for peak_id in np.unique(correct_up):
            # Select indices and take probabilities from the locations
            # that contain at leas 5 points
            points_in_peak = np.where(correct_up == peak_id)[0]
            if points_in_peak.shape[0] >= 5:
                filtered_probs.append(above_thresh[points_in_peak].mean())
                filtered_peaks.append(peak_id)

        filtered_peaks = np.asarray(filtered_peaks)
        filtered_probs = np.asarray(filtered_probs)

        return filtered_peaks, filtered_probs

    def find_peaks(self, signal, verbose=False):
        """
        Execute the peak detection algorithm.

        Function uses LSTM model that was trained with 1000 sample windows
        of simulated noisy ECG data with sampling rate of 250 Hz. Following
        steps are executed:
        1. Input ECG is sig to 250 Hz
        2. Input ECG is divided into overlapping 1000 sample windows
        3. LSTM model is used to make predictions for windows from step 2
        4. R-peak locations are decided based on predictions
        5. R-peak locations are mapped back into original sampling frequency
        and they are corrected upwards

        Parameters
        ----------
        signal : array
            Single channel ECG signal.
        verbose : bool
            Whether print information.

        Returns
        -------
        orig_peaks : array
            Indices of the R-peaks as samples from the beginning from the
            original signal (not resampled signal).
        filtered_proba : array
            Probability values (probability that point is an R-peak) for
            all points in orig_peaks array.

        """
        if self.resample:
            if verbose:
                print("Resampling signal from ", self.iput_fs, "Hz to 250 Hz")
            sig = resample_poly(signal, up=250, down=self.iput_fs)

        else:
            sig = signal

        if verbose:
            print("Extracting windows, window size:",
                  self.win_size, " stride:", self.stride)
        padded_indices, data_windows = self._extract_windows(signal=sig)

        # Normalize each window to -1, 1 range
        normalize = partial(processing.normalize_bound, lb=-1, ub=1)
        data_windows = np.apply_along_axis(normalize, 1, data_windows)

        if verbose:
            print("Predicting peaks")
        predictions = self.model.predict(data_windows, verbose=0)

        if verbose:
            print("Calculating means for overlapping predictions (windows)")
        means_for_predictions = self._mean_preds(win_idx=padded_indices,
                                                 preds=predictions,
                                                 orig_len=sig.shape[0])

        predictions = means_for_predictions

        if verbose:
            print("Filtering out predictions below probabilty threshold ",
                  self.threshold)
        filtered_peaks, filtered_proba = self._filter_predictions(
                                              signal=sig,
                                              preds=predictions
                                              )
        if self.resample:

            # Resample peaks back to original frequency
            orig_peaks = processing.resample_ann(filtered_peaks, 250, self.iput_fs)

            # Correct peaks with respect to original signal
            orig_peaks = processing.correct_peaks(sig=signal,
                                                  peak_inds=orig_peaks,
                                                  search_radius=int(
                                                      self.iput_fs/50
                                                      ),
                                                  smooth_window_size=20,
                                                  peak_dir='up')

            # In some very rare cases final correction can introduce duplicate
            # peak values. If this is the case, then mean of the duplicate
            # values is taken.
            filtered_proba = self._calculate_means(indices=orig_peaks,
                                                   values=filtered_proba)
            orig_peaks = np.unique(orig_peaks)

        else:
            orig_peaks = filtered_peaks

        if verbose:
            print("Everything done")

        return orig_peaks, filtered_proba

    def remove_close(self, peaks, peak_probs, threshold_ms=200, verbose=False):
        """
        Remove peaks that are within self.threshold distance.

        At first, set of peaks (close_peaks) that occur within self.threshold
        distance are determined. Then, in the every iteration of while loop,
        peak with maximum probability is removed from this set and it's
        distance to neigbouring peaks is calculated. If distance to the
        neighbours is greater than threshold_ms, then peak is added back to
        accepted peaks, otherwise it is discarded. This continues until set
        of close_peaks is epmty.

        Parameters
        ----------
        peaks : array
            Indices of the peaks.
        peak_probs : array
            Probabilities for every peak that it is an R-peak.
        treshold_ms : int
            Minimum distance between two peaks in milliseconds.
        verbose : book
            Whether print information.

        Returns
        -------
        ok_peaks : array
            Indices of the peaks where all peaks are at least threshold_ms
            away from each other.

        """
        assert(peaks.shape == peak_probs.shape)

        # Threshold as seconds
        threshold = threshold_ms/1000

        # treshold distance as samples
        td = int(np.ceil(threshold * self.iput_fs))

        close_peaks = []
        for p in peaks:
            # Select peaks that are within threshold distance
            in_td = np.array(
                (peaks > p - td) * (peaks < p + td) * (peaks != p)
            )
            peaks_within_td = peaks[in_td]
            if peaks_within_td.size > 0:
                close_peaks.append([*peaks_within_td])

        # Peaks that are within threshold distance
        close_peaks = [val for sublist in close_peaks for val in sublist]
        close_peaks = np.unique(np.array(close_peaks))
        close_probs = peak_probs[np.isin(peaks, close_peaks)]
        close_indices = np.arange(close_peaks.shape[0])

        # Peaks without peaks that occur too close to each other
        ok_peaks = peaks[~np.isin(peaks, close_peaks)]
        if verbose:
            print("All R-peaks: ", peaks.shape[0])
            print("R-peaks that are within threshold distance: ",
                  len(close_peaks))
            print("R-peaks that aren't within threshold distance: ",
                  ok_peaks.shape[0])

        while close_probs.shape[0] > 0:
            # From peaks that occur too close,
            # select peak with maximum probability
            max_peak = close_peaks[np.argmax(close_probs)]

            # Remove the selected peak from the set of too close peaks
            max_idx = np.argmax(close_probs)
            close_probs = close_probs[close_indices != max_idx]
            close_peaks = close_peaks[close_indices != max_idx]
            close_indices = np.arange(close_peaks.shape[0])

            if max_peak < ok_peaks[0]:
                # If selected peak is the first peak, compare it to the
                # following peak
                nxt_peak = ok_peaks[ok_peaks > max_peak][0]
                if ((nxt_peak - max_peak) > td):
                    ok_peaks = np.append(ok_peaks, max_peak)
                    ok_peaks = np.sort(ok_peaks)

            elif max_peak > ok_peaks[-1]:
                # If selected peak is the last peak, compare it to the
                # preceding peak
                prv_peak = ok_peaks[ok_peaks < max_peak][-1]
                if ((max_peak-prv_peak) > td):
                    ok_peaks = np.append(ok_peaks, max_peak)
                    ok_peaks = np.sort(ok_peaks)

            else:
                # Compare selected peak to the following and preceding
                nxt_peak = ok_peaks[ok_peaks > max_peak][0]
                prv_peak = ok_peaks[ok_peaks < max_peak][-1]
                if ((nxt_peak - max_peak) > td) and ((max_peak-prv_peak) > td):
                    ok_peaks = np.append(ok_peaks, max_peak)
                    ok_peaks = np.sort(ok_peaks)
        if verbose:
            print("final number of peaks:", ok_peaks.shape[0])

        return np.asarray(ok_peaks)

