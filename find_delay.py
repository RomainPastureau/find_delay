"""Contains a series of functions directed at calculating the delay between two arrays.
* find_delay finds the delay between two time series using cross-correlation.
* find_delays does the same, but for multiple excerpts from one big time series.

Author: Romain Pastureau, BCBL (Basque Center on Cognition, Brain and Language)
Current version: 1.1 (2024-04-16)

Version history
---------------
1.2 (2024-04-17) · Added transparency of the second (orange) array on the graph overlay
                 · Clarified README.md and added figures
1.1 (2024-04-16) · Added find_delays
                 · Created _create_figure containing all the plotting-related code
                 · Modified the graph plot when the max correlation is below threshold
                 · Minor corrections in docstrings
1.0 (2024-04-12) · Initial release
"""

from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
from scipy.io import wavfile
from scipy.signal import butter, correlate, hilbert, lfilter
import numpy as np
import datetime as dt


def _filter_frequencies(array, frequency, filter_below=None, filter_over=None, verbosity=1):
    """Applies a low-pass, high-pass or band-pass filter to the data in the attribute :attr:`samples`.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    frequency: int or float
        The sampling frequency of the array, in Hz.

    filter_below: float or None, optional
    	The value below which you want to filter the data. If set on None or 0, this parameter will be ignored.
    	If this parameter is the only one provided, a high-pass filter will be applied to the samples; if
    	``filter_over`` is also provided, a band-pass filter will be applied to the samples.

    filter_over: float or None, optional
    	The value over which you want to filter the data. If set on None or 0, this parameter will be ignored.
    	If this parameter is the only one provided, a low-pass filter will be applied to the samples; if
    	``filter_below`` is also provided, a band-pass filter will be applied to the samples.

    verbosity: int, optional
    	Sets how much feedback the code will provide in the console output:

    	• *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
    	• *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
    	  current steps.
    	• *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
    	  may clutter the output and slow down the execution.

    Returns
    -------
    np.array
    	The array with filtered values.
    """

    # Band-pass filter
    if filter_below not in [None, 0] and filter_over not in [None, 0]:
        if verbosity > 0:
            print("\tApplying a band-pass filter for frequencies between " + str(filter_below) + " and " +
                  str(filter_over) + " Hz.")
        b, a = butter(2, [filter_below, filter_over], "band", fs=frequency)
        filtered_array = lfilter(b, a, array)

    # High-pass filter
    elif filter_below not in [None, 0]:
        if verbosity > 0:
            print("\tApplying a high-pass filter for frequencies over " + str(filter_below) + " Hz.")
        b, a = butter(2, filter_below, "high", fs=frequency)
        filtered_array = lfilter(b, a, array)

    # Low-pass filter
    elif filter_over not in [None, 0]:
        if verbosity > 0:
            print("\tApplying a low-pass filter for frequencies below " + str(filter_over) + " Hz.")
        b, a = butter(2, filter_over, "low", fs=frequency)
        filtered_array = lfilter(b, a, array)

    else:
        filtered_array = array

    return filtered_array


def _get_number_of_windows(array_length_or_array, window_size, overlap=0, add_incomplete_window=True):
    """Given an array, calculates how many windows from the defined `window_size` can be created, with or
    without overlap.

    Parameters
    ----------
    array_length_or_array: list, np.ndarray or int
        An array of numerical values, or its length.
    window_size: int
        The number of array elements in each window.
    overlap: int
        The number of array elements overlapping in each window.
    add_incomplete_window: bool
        If set on ``True``, the last window will be included even if its size is smaller than ``window_size``.
        Otherwise, it will be ignored.

    Returns
    -------
    int
        The number of windows than can be created from the array.
    """

    if not isinstance(array_length_or_array, int):
        array_length = len(array_length_or_array)
    else:
        array_length = array_length_or_array

    if overlap >= window_size:
        raise Exception("The size of the overlap (" + str(overlap) + ") cannot be bigger than or equal to the size " +
                        "of the window (" + str(window_size) + ").")
    if overlap > array_length or window_size > array_length:
        raise Exception("The size of the window (" + str(window_size) + ") or the overlap (" + str(overlap) + ") " +
                        "cannot be bigger than the size of the array (" + str(array_length) + ").")

    number_of_windows = (array_length - overlap) / (window_size - overlap)

    if add_incomplete_window and array_length + (overlap * (window_size - 1)) % window_size != 0:
        return int(np.ceil(number_of_windows))

    else:
        return int(number_of_windows)


def _get_window_length(array_length_or_array, number_of_windows, overlap_ratio):
    """Given an array to be split in a given overlapping number of windows, calculates the number of elements in each
    window.

    Parameters
    ----------
    array_length_or_array: list, np.ndarray or int
        An array of numerical values, or its length.
    number_of_windows: int
        The number of windows to split the array in.
    overlap_ratio: float
        The ratio of overlapping elements between each window.

    Returns
    -------
    int
        The number of elements in each window. Note: the last window may have fewer elements than the others if the
        number of windows does not divide the result of :math:`array_length + (number_of_windows - 1) × overlap`.
    """

    if not isinstance(array_length_or_array, int):
        array_length = len(array_length_or_array)
    else:
        array_length = array_length_or_array

    if overlap_ratio >= 1 or overlap_ratio < 0:
        raise Exception("The size of the overlap ratio (" + str(overlap_ratio) + ") must be superior or equal to 0, " +
                        "and strictly inferior to 1.")

    return array_length / (number_of_windows + overlap_ratio - number_of_windows * overlap_ratio)


def _get_envelope(array, frequency, number_of_windows=100, overlap_ratio=0.5, filter_below=None, filter_over=None,
                 verbosity=1):
    """Calculates the envelope of an array, and returns it. The function can also optionally perform a band-pass
    filtering, if the corresponding parameters are provided.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    frequency: int or float
        The sampling frequency of the array, in Hz.

    number_of_windows: int or None, optional
        The number of windows in which to cut the original array. The lower this parameter is, the more
        resources the computation will need. If this parameter is set on `None`, the window size will be set on
        the number of samples. Note that this number has to be inferior to 2 times the number of samples in the array;
        otherwise, at least some windows would only contain one sample.

    overlap_ratio: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap.

    filter_below: int, float or None, optional
        If not ``None`` nor 0, this value will be provided as the lowest frequency of the band-pass filter.

    filter_over: int, float or None, optional
        If not ``None`` nor 0, this value will be provided as the highest frequency of the band-pass filter.

    verbosity: int, optional
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    Returns
    -------
    np.array
        The envelope of the original array.
    """

    time_before = dt.datetime.now()

    # Settings
    if number_of_windows is None:
        number_of_windows = 1

    if overlap_ratio is None:
        overlap_ratio = 0

    if number_of_windows > 2 * len(array):
        raise Exception("The number of windows is too big, and will lead to windows having only one sample." +
                        " Please consider using a lower number of windows.")

    window = int(_get_window_length(len(array), number_of_windows, overlap_ratio))
    overlap = int(np.ceil(overlap_ratio * window))

    # Hilbert transform
    if verbosity > 0:
        print("\tGetting the Hilbert transform...", end=" ")
    elif verbosity > 1:
        print("\tGetting the Hilbert transform...")
        print("\t\tDividing the samples in " + str(number_of_windows) + " window(s) of " + str(window) +
              " samples, with an overlap of " + str(overlap) + " samples.")

    envelope = np.zeros(len(array))
    j = 0
    next_percentage = 10

    for i in range(number_of_windows):

        if verbosity == 1:
            while i / number_of_windows > next_percentage / 100:
                print(str(next_percentage) + "%", end=" ")
                next_percentage += 10

        # Get the Hilbert transform of the window
        array_start = i * (window - overlap)
        array_end = np.min([(i + 1) * window - i * overlap, len(array)])
        if verbosity > 1:
            print("\t\t\tGetting samples from window " + str(i + 1) + "/" + str(number_of_windows) + ": samples " +
                  str(array_start) + " to " + str(array_end) + "... ", end=" ")
        hilbert_window = np.abs(hilbert(array[array_start:array_end]))

        # Keep only the center values
        if i == 0:
            slice_start = 0
        else:
            slice_start = overlap // 2  # We stop one before if the overlap is odd

        if i == number_of_windows - 1:
            slice_end = len(hilbert_window)
        else:
            slice_end = window - int(np.ceil(overlap / 2))

        if verbosity > 1:
            print("\n\t\t\tKeeping the samples from " + str(slice_start) + " to " + str(slice_end) + " in the " +
                  "window: samples " + str(array_start + slice_start) + " to " + str(array_start + slice_end) + "...",
                  end=" ")

        preserved_samples = hilbert_window[slice_start:slice_end]
        envelope[j:j + len(preserved_samples)] = preserved_samples
        j += len(preserved_samples)

        if verbosity > 1:
            print("Done.")

    if verbosity == 1:
        print("100% - Done.")
    elif verbosity > 1:
        print("Done.")

    # Filtering
    if filter_below is not None or filter_over is not None:
        envelope = _filter_frequencies(envelope, frequency, filter_below, filter_over, verbosity)

    if verbosity > 0:
        print("\tEnvelope calculated in: " + str(dt.datetime.now() - time_before))

    return envelope


def _resample_window(array, original_timestamps, resampled_timestamps, index_start_original, index_end_original,
                    index_start_resampled, index_end_resampled, method="cubic", verbosity=1):
    """Performs and returns the resampling on a subarray of samples.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    original_timestamps: list or np.ndarray
        An array containing the timestamps for each sample of the original array.

    resampled_timestamps: list or np.ndarray
        An array containing the timestamps for each desired sample in the resampled array.

    index_start_original: int
        The index in the array where the window starts.

    index_end_original: int
        The index in the array where the window ends.

    index_start_resampled: int
        The index in the resampled array where the window starts.

    index_end_resampled: int
        The index in the resampled array where the window ends.

    method: str, optional
        This parameter allows for various values:

        • ``"linear"`` performs a linear
          `numpy.interp <https://numpy.org/devdocs/reference/generated/numpy.interp.html>`_ interpolation. This method,
          though simple, may not be very precise for upsampling naturalistic stimuli.
        • ``"cubic"`` performs a cubic interpolation via `scipy.interpolate.CubicSpline
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_. This method,
          while smoother than the linear interpolation, may lead to unwanted oscillations nearby strong variations in
          the data.
        • ``"pchip"`` performs a monotonic cubic spline interpolation (Piecewise Cubic Hermite Interpolating
          Polynomial) via `scipy.interpolate.PchipInterpolator
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_.
        • ``"akima"`` performs another type of monotonic cubic spline interpolation, using
        `scipy.interpolate.Akima1DInterpolator
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_.
        • ``"take"`` keeps one out of n samples from the original array. While being the fastest computation, it will
          be prone to imprecision if the downsampling factor is not an integer divider of the original frequency.
        • ``"interp1d_XXX"`` uses the function `scipy.interpolate.interp1d
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`. The XXX part of the
          parameter can be replaced by ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``, "slinear"``,
          ``"quadratic"``, ``"cubic"``”, ``"previous"``, and ``"next"`` (see the documentation of this function for
          specifics).

    verbosity: int, optional
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    Returns
    -------
    np.array
        The envelope of the original array.
    """

    array_window = array[index_start_original:index_end_original + 1]
    original_timestamps_window = original_timestamps[index_start_original:index_end_original + 1]
    resampled_timestamps_window = resampled_timestamps[index_start_resampled:index_end_resampled + 1]

    if verbosity > 1:
        print("\t\t\t\tIn the original array, the window contains samples " + str(index_start_original) + " to " +
              str(index_end_original) + " (from timestamps " + str(original_timestamps_window[0]) + " to " +
              str(original_timestamps_window[-1]) + ").")
        print("\t\t\t\tIn the new array, the window contains samples " + str(index_start_resampled) + " to " +
              str(index_end_resampled) + " (from timestamps " + str(resampled_timestamps_window[0]) + " to " +
              str(resampled_timestamps_window[-1]) + ").")
        print("\t\t\t\tInterpolating the data...", end=" ")

    if np.size(array_window) == 1:
        raise Exception("Only one sample is present in the current window. Please select a larger window size.")

    if method == "linear":
        return np.interp(resampled_timestamps_window, array_window, original_timestamps_window)
    elif method == "cubic":
        interp = CubicSpline(original_timestamps_window, array_window)
        return interp(resampled_timestamps_window)
    elif method == "pchip":
        interp = PchipInterpolator(original_timestamps_window, array_window)
        return interp(resampled_timestamps_window)
    elif method == "akima":
        interp = Akima1DInterpolator(original_timestamps_window, array_window)
        return interp(resampled_timestamps_window)
    elif method.startswith("interp1d"):
        interp = interp1d(original_timestamps, array, kind=method.split("_")[1])
        return interp(resampled_timestamps_window)
    else:
        raise Exception("Invalid resampling method: " + str(method) + ".")


def _resample(array, original_frequency, resampling_frequency, number_of_windows=100, overlap_ratio=0.5,
             method="cubic", verbosity=1):
    """Resamples an array to the `resampling_frequency` parameter. It first creates a new set of timestamps at the
    desired frequency, and then interpolates the original data to the new timestamps.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    original_frequency: int or float
        The sampling frequency of the array, in Hz.

    resampling_frequency: int or float
        The frequency at which you want to resample the array, in Hz. A frequency of 4 will return samples
        at 0.25 s intervals.

    number_of_windows: int or None, optional
        The number of windows in which to cut the original array. The lower this parameter is, the more
        resources the computation will need. If this parameter is set on `None`, the window size will be set on
        the number of samples. Note that this number has to be inferior to 2 times the number of samples in the array;
        otherwise, at least some windows would only contain one sample.

    overlap_ratio: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap.

    method: str, optional
        This parameter allows for various values:

        • ``"linear"`` performs a linear
          `numpy.interp <https://numpy.org/devdocs/reference/generated/numpy.interp.html>`_ interpolation. This method,
          though simple, may not be very precise for upsampling naturalistic stimuli.
        • ``"cubic"`` performs a cubic interpolation via `scipy.interpolate.CubicSpline
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_. This method,
          while smoother than the linear interpolation, may lead to unwanted oscillations nearby strong variations in
          the data.
        • ``"pchip"`` performs a monotonic cubic spline interpolation (Piecewise Cubic Hermite Interpolating
          Polynomial) via `scipy.interpolate.PchipInterpolator
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_.
        • ``"akima"`` performs another type of monotonic cubic spline interpolation, using
        `scipy.interpolate.Akima1DInterpolator
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_.
        • ``"take"`` keeps one out of n samples from the original array. While being the fastest computation, it will
          be prone to imprecision if the downsampling factor is not an integer divider of the original frequency.
        • ``"interp1d_XXX"`` uses the function `scipy.interpolate.interp1d
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`. The XXX part of the
          parameter can be replaced by ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``, "slinear"``,
          ``"quadratic"``, ``"cubic"``”, ``"previous"``, and ``"next"`` (see the documentation of this function for
          specifics).

    verbosity: int, optional
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    Returns
    -------
    array
        The resampled array.

    Warning
    -------
    This function allows both the **upsampling** and the **downsampling** of arrays. However, during any of
    these operations, the algorithm only **estimates** the real values of the samples. You should then consider
    the upsampling (and the downsampling, to a lesser extent) with care.
    """

    time_before = dt.datetime.now()

    if resampling_frequency == original_frequency:
        if verbosity > 0:
            print("\tNot performing the resampling as the resampling frequency is the same as the original frequency" +
                  " (" + str(resampling_frequency) + " Hz).")
        return array

    if verbosity > 0:
        print("\tResampling the array at " + str(resampling_frequency) + " Hz (mode: " + str(method) + ")...")
        print("\t\tOriginal frequency: " + str(round(original_frequency, 2)) + " Hz.")
        if verbosity > 1:
            print("\t\tPerforming the resampling...")
        else:
            print("\t\tPerforming the resampling...", end=" ")

    if method == "take":
        if resampling_frequency > original_frequency:
            raise Exception("""The mode "take" does not allow for upsampling of the data. Please input a resampling
            frequency inferior to the original (""" + str(original_frequency) + ").")
        factor_resampling = original_frequency / resampling_frequency
        print(factor_resampling)
        if factor_resampling != int(factor_resampling):
            print("Warning: The downsampling factor is not an integer (" + str(factor_resampling) + "), meaning that " +
                  "the downsampling may not be accurate. To ensure an accurate resampling with the \"take\" mode, use" +
                  " a resampling frequency that is an integer divider of the original frequency (" +
                  str(original_frequency) + " Hz).")
        indices = np.arange(0, len(array), factor_resampling, dtype=int)
        resampled_array = np.take(array, indices)

    else:

        # Define the timestamps
        original_timestamps = np.arange(0, np.size(array)) / original_frequency
        resampled_timestamps = np.arange(0, np.max(original_timestamps) + (1 / resampling_frequency),
                                         1 / resampling_frequency)

        # If the last timestamp is over the last original timestamp, we remove it
        if resampled_timestamps[-1] > max(original_timestamps):
            resampled_timestamps = resampled_timestamps[:-1]

        # Settings
        if number_of_windows is None:
            number_of_windows = 1

        if overlap_ratio is None:
            overlap_ratio = 0

        if number_of_windows > 2 * len(array):
            raise Exception("The number of windows is too big, and will lead to windows having only one sample." +
                            " Please consider using a number of windows lower than " + str(2 * len(array)) + ".")

        window = int(_get_window_length(len(array), number_of_windows, overlap_ratio))
        overlap = int(np.ceil(overlap_ratio * window))

        if verbosity > 1 and number_of_windows != 1:
            print("\t\t\tCreating " + str(number_of_windows) + " window(s), each containing " + str(window) +
                  " samples, with a " + str(round(overlap_ratio * 100, 2)) + " % overlap (" + str(overlap) +
                  " samples).")

        resampled_array = np.zeros(np.size(resampled_timestamps))
        j = 0
        next_percentage = 10

        for i in range(number_of_windows):

            window_start_original = i * (window - overlap)
            window_end_original = np.min([(i + 1) * window - i * overlap, len(original_timestamps) - 1])
            window_start_resampled = int(np.round(original_timestamps[window_start_original] * resampling_frequency))
            window_end_resampled = int(np.round(original_timestamps[window_end_original] * resampling_frequency))

            if verbosity == 1:
                while i / number_of_windows > next_percentage / 100:
                    print(str(next_percentage) + "%", end=" ")
                    next_percentage += 10

            if verbosity > 1:
                print("\t\t\tGetting samples from window " + str(i + 1) + "/" + str(number_of_windows) + ".")

            resampled_window = _resample_window(array, original_timestamps, resampled_timestamps, window_start_original,
                                               window_end_original, window_start_resampled, window_end_resampled,
                                               method, verbosity)

            if verbosity > 1:
                print("Done.\n\t\t\t\tThe resampled window contains " + str(np.size(resampled_window)) + " sample(s).")

            # Keep only the center values
            if i == 0:
                window_slice_start = 0
                resampled_slice_start = 0
            else:
                window_slice_start = (j - window_start_resampled) // 2
                resampled_slice_start = window_start_resampled + window_slice_start

            preserved_samples = resampled_window[window_slice_start:]

            if verbosity > 1:
                print("\t\t\tKeeping the samples from " + str(resampled_slice_start) + " to " +
                      str(resampled_slice_start + len(preserved_samples) - 1) + " (in the window, samples " +
                      str(window_slice_start) + " to " + str(len(resampled_window) - 1) + ")", end="")
                if resampled_slice_start != j:
                    print(" · Rewriting samples " + str(resampled_slice_start) + " to " + str(j - 1) + "...", end=" ")
                else:
                    print("...", end=" ")

            resampled_array[resampled_slice_start:resampled_slice_start + len(preserved_samples)] = preserved_samples
            j = resampled_slice_start + len(preserved_samples)

            if verbosity > 1:
                print("Done.")

        if verbosity == 1:
            print("100% - Done.")
        elif verbosity > 1:
            print("\t\tResampling done.")

    if verbosity > 0:
        print("\t\tThe original array had " + str(len(array)) + " samples.")
        print("\t\tThe new array has " + str(len(resampled_array)) + " samples.")
        print("\tResampling performed in: " + str(dt.datetime.now() - time_before))

    return resampled_array


def _create_figure(array_1, array_2, freq_array_1, freq_array_2, name_array_1, name_array_2, envelope_1, envelope_2,
                   y1, y2, compute_envelope, number_of_windows_env, overlap_ratio_env, filter_below, filter_over,
                   resampling_rate, number_of_windows_res, overlap_ratio_res, number_of_plots, plot_intermediate_steps,
                   cross_correlation, return_delay_format, return_value, correlation_value, delay, rate, plot_figure,
                   path_figure, threshold, verbosity):
    """
    Creates and/or saves a figure given the parameters of the find_delay function.
    """

    fig, ax = plt.subplots(int(np.ceil(number_of_plots / 2)), 2, constrained_layout=True, figsize=(16, 8))

    i = 0

    if plot_intermediate_steps:
        ax[i // 2][i % 2].set_title(name_array_1 + ": " + str(round(freq_array_1, 2)) + " Hz")
        ax[i // 2][i % 2].plot(np.arange(0, len(array_1)) / freq_array_1, array_1)
        i += 1

        ax[i // 2][i % 2].set_title(name_array_2 + ": " + str(round(freq_array_2, 2)) + " Hz")
        ax[i // 2][i % 2].plot(np.arange(0, len(array_2)) / freq_array_2, array_2, color="orange")
        i += 1

        if compute_envelope:
            band_pass_low = "0" if filter_below is None else filter_below
            band_pass_high = "∞" if filter_over is None else filter_over

            title = "Envelope " + name_array_1 + ": " + str(freq_array_1) + "Hz, " + str(number_of_windows_env) + \
                    " w, " + str(overlap_ratio_env) + " o"
            if filter_below is not None or filter_over is not None:
                title += " · Band-pass [" + str(band_pass_low) + ", " + str(band_pass_high) + "]"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(np.arange(0, len(envelope_1)) / freq_array_1, envelope_1)
            i += 1

            title = "Envelope " + name_array_2 + ": " + str(freq_array_2) + "Hz, " + str(number_of_windows_env) + \
                    " w, " + str(overlap_ratio_env) + " o"
            if filter_below is not None or filter_over is not None:
                title += " · Band-pass [" + str(band_pass_low) + ", " + str(band_pass_high) + "]"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(np.arange(0, len(envelope_2)) / freq_array_2, envelope_2, color="orange")
            i += 1

        if resampling_rate is not None:
            if compute_envelope is False:
                env_or_array = ""
            else:
                env_or_array = "Envelope "

            title = "Resampled " + env_or_array + name_array_1 + ": " + str(resampling_rate) + " Hz, " + \
                    str(number_of_windows_res) + " w, " + str(overlap_ratio_res) + " o"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(np.arange(0, len(y1)) / resampling_rate, y1)
            i += 1

            title = "Resampled " + env_or_array + name_array_2 + ": " + str(resampling_rate) + " Hz, " + \
                    str(number_of_windows_res) + " w, " + str(overlap_ratio_res) + " o"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(np.arange(0, len(y2)) / resampling_rate, y2, color="orange")
            i += 1

        band_pass_low = "0" if filter_below is None else filter_below
        band_pass_high = "∞" if filter_over is None else filter_over

        # Title
        title = "Cross-correlation"
        if compute_envelope:
            title += ": Envelope " + str(number_of_windows_env) + " w, " + str(overlap_ratio_env) + " o"
            if filter_below is not None or filter_over is not None:
                title += " · Band-pass [" + str(band_pass_low) + ", " + str(band_pass_high) + "]"
            if resampling_rate is not None:
                title += "\n"
        if resampling_rate is not None:
            if not compute_envelope:
                title += ": "
            title += "Resampling " + str(resampling_rate) + " Hz, " + str(number_of_windows_res) + " w, " + \
                     str(overlap_ratio_res) + " o"

        ax[i // 2][i % 2].set_title(title)
        ax[i // 2][i % 2].set_ylim(np.min(cross_correlation), 1.5)
        ax[i // 2][i % 2].plot(cross_correlation, color="green")
        text = ""
        if return_delay_format == "index":
            text = "Sample "
        text += str(return_value)
        if return_delay_format in ["ms", "s"]:
            text += return_delay_format

        if correlation_value >= threshold:
            text += " · Correlation value: " + str(round(correlation_value, 3))
            bbox_props = dict(boxstyle="square,pad=0.3", fc="#99cc00", ec="k", lw=0.72)
        else:
            text += " · Correlation value (below threshold): " + str(round(correlation_value, 3))
            bbox_props = dict(boxstyle="square,pad=0.3", fc="#ff0000", ec="k", lw=0.72)
        arrow_props = dict(arrowstyle="->", connectionstyle="angle,angleA=90")
        kw = dict(xycoords='data', textcoords="data",
                  arrowprops=arrow_props, bbox=bbox_props, ha="center", va="center")
        ax[i // 2][i % 2].annotate(text, xy=(delay, correlation_value), xytext=(delay, 1.4), **kw)

        i += 1

        ax[i // 2][i % 2].set_title("Aligned arrays")
        ax[i // 2][i % 2].plot(np.arange(0, len(array_1)) / freq_array_1, array_1, color="#04589388", linewidth=1)

        resampled_timestamps_array2 = np.arange(0, len(array_2)) / freq_array_2 + delay / rate
        resampled_timestamps_array2 = resampled_timestamps_array2[:len(array_2)]
        ax[i // 2][i % 2].plot(resampled_timestamps_array2, array_2, color="#ffa500aa", linewidth=2)

        if plot_figure:
            if verbosity > 0:
                print("\nShowing the graph...")
            plt.show()

        if path_figure:
            if verbosity > 0:
                print("\nSaving the graph under " + str(path_figure) + "...", end=" ")
            plt.savefig(path_figure)
            if verbosity > 0:
                print("Done.")


def find_delay(array_1, array_2, freq_array_1=1, freq_array_2=1, compute_envelope=True, number_of_windows_env=10000,
               overlap_ratio_env=0.5, filter_below=None, filter_over=50, resampling_rate=None,
               number_of_windows_res=100, overlap_ratio_res=0.5, resampling_mode="cubic",
               return_delay_format="index", return_correlation_value=False, threshold=0.9,
               plot_figure=False, plot_intermediate_steps=False, path_figure=None,
               verbosity=1):
    """This function tries to find the timestamp at which an excerpt (array_2) begins in a time series (array_1).
    The computation is performed through cross-correlation. Before so, the envelopes of both arrays can first be
    calculated and filtered (recommended for audio files), and resampled (necessary when the sampling rate of the two
    arrays is unequal). The function returns the timestamp of the maximal correlation value, or `None` if this value is
    below threshold. Optionally, it can also return a second element, the maximal correlation value.

    Important
    ---------
    Because it is easy to get confused: this function returns the timestamp in array_1 where array_2 begins. This means
    that, if you want to align array_1 and array_2, you need to remove the delay to each timestamp of array_1: that way,
    the value at timestamp 0 in array_1 will be aligned with the value at timestamp 0 in array_2.

    Note
    ----
    Using Numpy arrays rather than lists will drastically increase the speed of the computation.

    Parameters
    ----------
    array_1: list or np.ndarray
        An first array of samples.

    array_2: list or np.ndarray
        An second array of samples, smaller than or of equal size to the first one, that is allegedly an excerpt
        from the first one. The amplitude, frequency or values do not have to match exactly the ones from the first
        array.

    freq_array_1: int or float, optional
        The sampling frequency of the first array, in Hz (default: 1).

    freq_array_2: int or float
        The sampling frequency of the second array, in Hz (default: 1).

    compute_envelope: bool, optional
        If `True` (default), calculates the envelope of the array values before performing the cross-correlation.

    number_of_windows_env: int or None, optional
        The number of windows in which to cut the original array to calculate the envelope. The lower this parameter is,
        the more resources the computation will need. If this parameter is set on `None`, the window size will be set on
        the number of samples. Note that this number has to be inferior to 2 times the number of samples in the array;
        otherwise, at least some windows would only contain one sample.
        A good value for this parameter is generally len(array)//10000.

    overlap_ratio_env: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap.

    filter_below: int or None, optional
        If set, a high-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        0 Hz).

    filter_over: int or None, optional
        If set, a low-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        50 Hz).

    resampling_rate: int or None, optional
        The sampling rate at which to downsample the arrays for the cross-correlation. A larger value will result in
        longer computation times (default: 1000). Setting the parameter on `None` will not downsample the arrays,
        which will result in an error if the two arrays are not the same frequency. If this parameter is `None`, the
        next parameters related to resampling can be ignored.

    number_of_windows_res: int or None, optional
        The number of windows in which to cut the original array. The lower this parameter is, the more
        resources the computation will need. If this parameter is set on `None`, the window size will be set on
        the number of samples. Note that this number has to be inferior to 2 times the number of samples in the array;
        otherwise, at least some windows would only contain one sample.

    overlap_ratio_res: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap.

    resampling_mode: str, optional
        This parameter allows for various values:

        • ``"linear"`` performs a linear
          `numpy.interp <https://numpy.org/devdocs/reference/generated/numpy.interp.html>`_ interpolation. This method,
          though simple, may not be very precise for upsampling naturalistic stimuli.
        • ``"cubic"`` performs a cubic interpolation via `scipy.interpolate.CubicSpline
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_. This method,
          while smoother than the linear interpolation, may lead to unwanted oscillations nearby strong variations in
          the data.
        • ``"pchip"`` performs a monotonic cubic spline interpolation (Piecewise Cubic Hermite Interpolating
          Polynomial) via `scipy.interpolate.PchipInterpolator
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_.
        • ``"akima"`` performs another type of monotonic cubic spline interpolation, using
          `scipy.interpolate.Akima1DInterpolator
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_.
        • ``"take"`` keeps one out of n samples from the original array. While being the fastest computation, it will
          be prone to imprecision if the downsampling factor is not an integer divider of the original frequency.
        • ``"interp1d_XXX"`` uses the function `scipy.interpolate.interp1d
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`. The XXX part of the
          parameter can be replaced by ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``, "slinear"``,
          ``"quadratic"``, ``"cubic"``”, ``"previous"``, and ``"next"`` (see the documentation of this function for
          specifics).

    threshold: float, optional
        The threshold of the minimum correlation value between the two arrays to accept a delay as a solution. If
        multiple delays are over threshold, the delay with the maximum correlation value will be returned. This value
        should be between 0 and 1; if the maximum found value is below the threshold, the function will return `None`
        instead of a timestamp.

    return_delay_format: str, optional
        This parameter can be either ``"index"``, ``"ms"``, ``"s"``, or ``"timedelta"``:

            • If ``"index"`` (default), the function will return the index in array_1 at which array_2 has the highest
              cross-correlation value.
            • If ``"ms"``, the function will return the timestamp in array_1, in milliseconds, at which array_2 has the
              highest cross-correlation value.
            • If ``"s"``, the function will return the timestamp in array_1, in seconds, at which array_2 has the
              highest cross-correlation value.
            • If ``"timedelta"``, the function will return the timestamp in array_1 at which array_2 has the
              highest cross-correlation value as a
              `datetime.timedelta <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_ object.

    return_correlation_value: bool, optional
        If `True`, the function returns a second value: the correlation value at the returned delay. This value will
        be None if it is below the specified threshold.

    plot_figure: bool, optional
        If set on `True`, plots a graph showing the result of the cross-correlation using Matplotlib. Note that plotting
        the figure causes an interruption of the code execution.

    plot_intermediate_steps: bool, optional
        If set on `True`, plots the original arrays, the envelopes (if calculated) and the resampled arrays (if
        calculated) besides the cross-correlation.

    path_figure: str or None, optional
        If set, saves the figure at the given path.

    verbosity: int, optional
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    Returns
    -------
    int, float, timedelta or None
        The sample index, timestamp or timedelta of array_1 at which array_2 can be found (defined by the parameter
        return_delay_format), or `None` if array1 is not contained in array_2.
    float or None, optional
        Optionally, if return_correlation_value is `True`, the correlation value at the corresponding index/timestamp.
    """

    time_before_function = dt.datetime.now()

    # Introduction
    if verbosity > 0:
        print("Trying to find when the second array starts in the first.")
        print("\t The first array contains " + str(len(array_1)) + " samples, at a rate of " + str(freq_array_1) +
              " Hz.")
        print("\t The second array contains " + str(len(array_2)) + " samples, at a rate of " + str(freq_array_2) +
              " Hz.\n")

    # Turn lists into ndarray
    if isinstance(array_1, list):
        array_1 = np.array(array_1)
    if isinstance(array_2, list):
        array_2 = np.array(array_2)

    number_of_plots = 2
    if plot_intermediate_steps:
        number_of_plots += 2

    # Envelope
    if compute_envelope:
        if plot_intermediate_steps:
            number_of_plots += 2
        if verbosity > 0:
            print("Getting the envelope from array 1...")
        envelope_1 = _get_envelope(array_1, freq_array_1, number_of_windows_env, overlap_ratio_env, filter_below,
                                  filter_over, verbosity)
        if verbosity > 0:
            print("Getting the envelope from array 2...")
        envelope_2 = _get_envelope(array_2, freq_array_2, number_of_windows_env, overlap_ratio_env, filter_below,
                                  filter_over, verbosity)
        if verbosity > 0:
            print("Envelopes calculated.\n")
    else:
        envelope_1 = array_1
        envelope_2 = array_2

    # Resampling
    if resampling_rate is not None:
        if plot_intermediate_steps:
            number_of_plots += 2
        rate = resampling_rate
        if verbosity > 0:
            print("Resampling array 1...")
        y1 = _resample(envelope_1, freq_array_1, resampling_rate, number_of_windows_res, overlap_ratio_res,
                      resampling_mode, verbosity)
        if verbosity > 0:
            print("Resampling array 2...")
        y2 = _resample(envelope_2, freq_array_2, resampling_rate, number_of_windows_res, overlap_ratio_res,
                      resampling_mode, verbosity)
        if verbosity > 0:
            print("Resampling done.\n")
    else:
        rate = freq_array_1
        if freq_array_1 != freq_array_2:
            raise Exception("The rate of the two arrays you are trying to correlate are different (" +
                            str(freq_array_1) + " Hz and " + str(freq_array_2) + " Hz). You must indicate a " +
                            "resampling rate to perform the cross-correlation.")
        y1 = envelope_1
        y2 = envelope_2

    time_before_cross_correlation = dt.datetime.now()

    if verbosity > 0:
        print("Computing the correlation...", end=" ")

    y2_normalized = (y2 - y2.mean()) / y2.std() / np.sqrt(y2.size)
    y1_m = correlate(y1, np.ones(y2.size), "valid") ** 2 / y2_normalized.size
    y1_m2 = correlate(y1 ** 2, np.ones(y2.size), "valid")
    cross_correlation = correlate(y1, y2_normalized, "valid") / np.sqrt(y1_m2 - y1_m)
    correlation_value = np.max(cross_correlation)

    delay = np.argmax(cross_correlation)
    index = int(round(delay * freq_array_1 / rate, 0))
    delay_in_seconds = delay / rate
    t = dt.timedelta(days=delay_in_seconds // 86400, seconds=int(delay_in_seconds % 86400),
                     microseconds=(delay_in_seconds % 1) * 1000000)

    if verbosity > 0:
        print("Done.")
        print("\tCross-correlation calculated in: " + str(dt.datetime.now() - time_before_cross_correlation))

        if correlation_value >= threshold:
            print("\tMaximum correlation (" + str(round(correlation_value, 3)) + ") found at sample " +
                  str(index) + " (timestamp " + str(t) + ").")

        else:
            print("\tNo correlation over threshold found (max correlation: " + str(round(correlation_value, 3)) +
                  ") found at sample " + str(index) + " (timestamp " + str(t) + ").")

        print("\nComplete delay finding function executed in: " + str(dt.datetime.now() - time_before_function))

    # Return values: None if below threshold
    if return_delay_format == "index":
        return_value = index
    elif return_delay_format == "ms":
        return_value = delay_in_seconds * 1000
    elif return_delay_format == "s":
        return_value = delay_in_seconds
    elif return_delay_format == "timedelta":
        return_value = t
    else:
        raise Exception("Wrong value for the parameter return_delay_format: " + str(return_delay_format) + ". The " +
                        'value should be either "index", "ms", "s" or "timedelta".')

    # Plot and/or save the figure
    if plot_figure or path_figure is not None:
        _create_figure(array_1, array_2, freq_array_1, freq_array_2, "Array 1", "Array 2", envelope_1, envelope_2,
        y1, y2, compute_envelope, number_of_windows_env, overlap_ratio_env, filter_below, filter_over,
        resampling_rate, number_of_windows_res, overlap_ratio_res, number_of_plots, plot_intermediate_steps,
        cross_correlation, return_delay_format, return_value, correlation_value, delay, rate, plot_figure,
        path_figure, threshold, verbosity)

    if correlation_value >= threshold:

        if return_correlation_value:
            return return_value, correlation_value
        else:
            return return_value

    else:

        if return_correlation_value:
            return None, None
        else:
            return None


def find_delays(array, excerpts, freq_array=1, freq_excerpts=1, compute_envelope=True,
                number_of_windows_env=10000, overlap_ratio_env=0.5, filter_below=None, filter_over=50,
                resampling_rate=None, number_of_windows_res=100, overlap_ratio_res=0.5, resampling_mode="cubic",
                return_delay_format="index", return_correlation_values=False, threshold=0.9,
                plot_figure=False, plot_intermediate_steps=False, path_figure=None,
                verbosity=1):
    """This function tries to find the timestamp at which multiple excerpts begins in an array.
    The computation is performed through cross-correlation. Before so, the envelopes of both arrays can first be
    calculated and filtered (recommended for audio files), and resampled (necessary when the sampling rate of the two
    arrays is unequal). The function returns the timestamp of the maximal correlation value, or `None` if this value is
    below threshold. Optionally, it can also return a second element, the maximal correlation value.

    Important
    ---------
    Because it is easy to get confused: this function returns the timestamp in array where each excerpt begins. This
    means that, if you want to align the array and an excerpt, you need to remove the delay to each timestamp of the
    array: that way, the value at timestamp 0 in the array will be aligned with the value at timestamp 0 in the excerpt.

    Note
    ----
    Using Numpy arrays rather than lists will drastically increase the speed of the computation.

    Note
    ----
    Compared to find_delay (without an "s"), this function allows to compute only once the envelope of the first array,
    allowing to gain computation time.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    excerpts: list(list or np.ndarray)
        A list of excerpts, each being an array of samples. Each excerpt must be smaller than or of equal size to the
        array in which to locate it. The amplitude, frequency or values do not have to match exactly the ones from the
        first array.

    freq_array: int or float, optional
        The sampling frequency of the array, in Hz (default: 1).

    freq_excerpts: int or float or list(int or float)
        The sampling frequency of the excerpts, in Hz (default: 1). This parameter accepts a single value that will be
        applied for each excerpt, or a list of values that has to be the same length as the number of excerpts, with
        each value corresponding to the frequency of the corresponding excerpt.

    compute_envelope: bool, optional
        If `True` (default), calculates the envelope of the array values before performing the cross-correlation.

    number_of_windows_env: int or None, optional
        The number of windows in which to cut the original array to calculate the envelope. The lower this parameter is,
        the more resources the computation will need. If this parameter is set on `None`, the window size will be set on
        the number of samples. Note that this number has to be inferior to 2 times the number of samples in the array;
        otherwise, at least some windows would only contain one sample.
        A good value for this parameter is generally len(array)//10000.

    overlap_ratio_env: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap.

    filter_below: int or None, optional
        If set, a high-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        0 Hz).

    filter_over: int or None, optional
        If set, a low-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        50 Hz).

    resampling_rate: int or None, optional
        The sampling rate at which to downsample the arrays for the cross-correlation. A larger value will result in
        longer computation times (default: 1000). Setting the parameter on `None` will not downsample the arrays,
        which will result in an error if the two arrays are not the same frequency. If this parameter is `None`, the
        next parameters related to resampling can be ignored.

    number_of_windows_res: int or None, optional
        The number of windows in which to cut the original array. The lower this parameter is, the more
        resources the computation will need. If this parameter is set on `None`, the window size will be set on
        the number of samples. Note that this number has to be inferior to 2 times the number of samples in the array;
        otherwise, at least some windows would only contain one sample.

    overlap_ratio_res: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap.

    resampling_mode: str, optional
        This parameter allows for various values:

        • ``"linear"`` performs a linear
          `numpy.interp <https://numpy.org/devdocs/reference/generated/numpy.interp.html>`_ interpolation. This method,
          though simple, may not be very precise for upsampling naturalistic stimuli.
        • ``"cubic"`` performs a cubic interpolation via `scipy.interpolate.CubicSpline
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_. This method,
          while smoother than the linear interpolation, may lead to unwanted oscillations nearby strong variations in
          the data.
        • ``"pchip"`` performs a monotonic cubic spline interpolation (Piecewise Cubic Hermite Interpolating
          Polynomial) via `scipy.interpolate.PchipInterpolator
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_.
        • ``"akima"`` performs another type of monotonic cubic spline interpolation, using
          `scipy.interpolate.Akima1DInterpolator
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_.
        • ``"take"`` keeps one out of n samples from the original array. While being the fastest computation, it will
          be prone to imprecision if the downsampling factor is not an integer divider of the original frequency.
        • ``"interp1d_XXX"`` uses the function `scipy.interpolate.interp1d
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`. The XXX part of the
          parameter can be replaced by ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``, "slinear"``,
          ``"quadratic"``, ``"cubic"``”, ``"previous"``, and ``"next"`` (see the documentation of this function for
          specifics).

    threshold: float, optional
        The threshold of the minimum correlation value between the two arrays to accept a delay as a solution. If
        multiple delays are over threshold, the delay with the maximum correlation value will be returned. This value
        should be between 0 and 1; if the maximum found value is below the threshold, the function will return `None`
        instead of a timestamp.

    return_delay_format: str, optional
        This parameter can be either ``"index"``, ``"ms"``, ``"s"``, or ``"timedelta"``:

            • If ``"index"`` (default), the function will return the index in array_1 at which array_2 has the highest
              cross-correlation value.
            • If ``"ms"``, the function will return the timestamp in array_1, in milliseconds, at which array_2 has the
              highest cross-correlation value.
            • If ``"s"``, the function will return the timestamp in array_1, in seconds, at which array_2 has the
              highest cross-correlation value.
            • If ``"timedelta"``, the function will return the timestamp in array_1 at which array_2 has the
              highest cross-correlation value as a
              `datetime.timedelta <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_ object.

    return_correlation_values: bool, optional
        If `True`, the function returns a second value: the correlation value at the returned delay. This value will
        be None if it is below the specified threshold.

    plot_figure: bool, optional
        If set on `True`, plots a graph showing the result of the cross-correlation using Matplotlib. Note that plotting
        the figure causes an interruption of the code execution.

    plot_intermediate_steps: bool, optional
        If set on `True`, plots the original arrays, the envelopes (if calculated) and the resampled arrays (if
        calculated) besides the cross-correlation.

    path_figure: str or None, optional
        If set, saves the figure at the given path.

    verbosity: int, optional
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    Returns
    -------
    int, float, timedelta or None
        The sample index, timestamp or timedelta of array1 at which array2 can be found (defined by the parameter
        return_delay_format), or `None` if array1 is not contained in array2.
    float or None, optional
        Optionally, if return_correlation_value is `True`, the correlation value at the corresponding index/timestamp.
    """

    time_before_function = dt.datetime.now()

    delays = []
    correlation_values = []

    # Introduction
    if verbosity > 0:
        print("Trying to find when the excerpts starts in the array.")
        print("\t The main array contains " + str(len(array)) + " samples, at a rate of " + str(freq_array) +
              " Hz.")
        print("\t" + str(len(excerpts)) + " excerpts to find.")

    # Turn list into ndarray
    if isinstance(array, list):
        array = np.array(array)

    # Check that the length of the excerpts equals the length of the frequencies
    if isinstance(freq_excerpts, list):
        if len(excerpts) != len(freq_excerpts):
            raise Exception("The number of frequencies given for the excerpts (" + str(len(freq_excerpts)) + ") " +
                            "is inconsistent with the number of excerpts (" + str(len(excerpts)) + ").")

    number_of_plots = 2
    if plot_intermediate_steps:
        number_of_plots += 2

    # Envelope
    if compute_envelope:
        if plot_intermediate_steps:
            number_of_plots += 2
        if verbosity > 0:
            print("Getting the envelope from the array...")
        envelope_array = _get_envelope(array, freq_array, number_of_windows_env, overlap_ratio_env, filter_below,
                                       filter_over, verbosity)
        if verbosity > 0:
            print("Envelope calculated.\n")
    else:
        envelope_array = array

    # Resampling
    if resampling_rate is not None:
        if plot_intermediate_steps:
            number_of_plots += 2

        rate = resampling_rate
        if verbosity > 0:
            print("Resampling array...")
        y1 = _resample(envelope_array, freq_array, resampling_rate, number_of_windows_res, overlap_ratio_res,
                      resampling_mode, verbosity)
        if verbosity > 0:
            print("Resampling done.\n")
    else:
        rate = freq_array
        y1 = envelope_array

    for i in range(len(excerpts)):

        # Introduction
        if verbosity > 0:
            print("Excerpt " + str(i+1) + "/" + str(len(excerpts)))

        # Get the excerpt
        excerpt = excerpts[i]

        # Turn list into ndarray
        if isinstance(excerpt, list):
            excerpt = np.array(excerpt)

        # Get the frequency
        if isinstance(freq_excerpts, list):
            freq_excerpt = freq_excerpts[i]
        else:
            freq_excerpt = freq_excerpts

        if verbosity > 0:
            print("\t The excerpt contains " + str(len(excerpt)) + " samples, at a rate of " + str(freq_excerpt) +
                  " Hz.\n")

        # Envelope
        if compute_envelope:
            if verbosity > 0:
                print("Getting the envelope from the excerpt " + str(i+1) + "...")
            envelope_excerpt = _get_envelope(excerpt, freq_excerpt, number_of_windows_env, overlap_ratio_env,
                                            filter_below, filter_over, verbosity)
            if verbosity > 0:
                print("Envelope calculated.\n")
        else:
            envelope_excerpt = excerpt

        # Resampling
        if resampling_rate is not None:
            rate = resampling_rate
            if verbosity > 0:
                print("Resampling excerpt...")
            y2 = _resample(envelope_excerpt, freq_excerpt, resampling_rate, number_of_windows_res, overlap_ratio_res,
                          resampling_mode, verbosity)
            if verbosity > 0:
                print("Resampling done.\n")
        else:
            if freq_array != freq_excerpt:
                raise Exception("The rate of the two arrays you are trying to correlate are different (" +
                                str(freq_array) + " Hz and " + str(freq_excerpt) + " Hz). You must indicate a " +
                                "resampling rate to perform the cross-correlation.")
            y2 = envelope_excerpt

        time_before_cross_correlation = dt.datetime.now()

        if verbosity > 0:
            print("Computing the correlation...", end=" ")

        y2_normalized = (y2 - y2.mean()) / y2.std() / np.sqrt(y2.size)
        y1_m = correlate(y1, np.ones(y2.size), "valid") ** 2 / y2_normalized.size
        y1_m2 = correlate(y1 ** 2, np.ones(y2.size), "valid")
        cross_correlation = correlate(y1, y2_normalized, "valid") / np.sqrt(y1_m2 - y1_m)
        correlation_value = np.max(cross_correlation)

        delay = np.argmax(cross_correlation)
        index = int(round(delay * freq_array / rate, 0))
        delay_in_seconds = delay / rate
        t = dt.timedelta(days=delay_in_seconds // 86400, seconds=int(delay_in_seconds % 86400),
                         microseconds=(delay_in_seconds % 1) * 1000000)

        if verbosity > 0:
            print("Done.")
            print("\tCross-correlation calculated in: " + str(dt.datetime.now() - time_before_cross_correlation))

            if correlation_value >= threshold:
                print("\tMaximum correlation (" + str(round(correlation_value, 3)) + ") found at sample " +
                      str(index) + " (timestamp " + str(t) + ").")

            else:
                print("\tNo correlation over threshold found (max correlation: " + str(round(correlation_value, 3)) +
                      ") found at sample " + str(index) + " (timestamp " + str(t) + ").")

            print("\nComplete delay finding function executed in: " + str(dt.datetime.now() - time_before_function))

        # Return values: None if below threshold
        if return_delay_format == "index":
            return_value = index
        elif return_delay_format == "ms":
            return_value = delay_in_seconds * 1000
        elif return_delay_format == "s":
            return_value = delay_in_seconds
        elif return_delay_format == "timedelta":
            return_value = t
        else:
            raise Exception("Wrong value for the parameter return_delay_format: " + str(return_delay_format) +
                            '. The value should be either "index", "ms", "s" or "timedelta".')

        # Plot and/or save the figure
        if plot_figure or path_figure is not None:
            _create_figure(array, excerpt, freq_array, freq_excerpt, "Array", "Excerpt", envelope_array,
                           envelope_excerpt, y1, y2, compute_envelope, number_of_windows_env, overlap_ratio_env,
                           filter_below, filter_over, resampling_rate, number_of_windows_res, overlap_ratio_res,
                           number_of_plots, plot_intermediate_steps, cross_correlation, return_delay_format,
                           return_value, correlation_value, delay, rate, plot_figure, path_figure, threshold, verbosity)

        if correlation_value >= threshold:

            if return_correlation_values:
                delays.append(return_value)
                correlation_values.append(correlation_value)
            else:
                delays.append(return_value)

        else:

            if return_correlation_values:
                delays.append(None)
                correlation_values.append(None)
            else:
                delays.append(None)

    if return_correlation_values:
        return delays, correlation_values
    else:
        return delays


if __name__ == "__main__":
    # Example 1: random numbers
    array_1 = [24, 70, 28, 59, 13, 97, 63, 30, 89, 4, 8, 15, 16, 23, 42, 37, 70, 18, 59, 48, 41, 83, 99, 6, 24, 86]
    array_2 = [4, 8, 15, 16, 23, 42]

    find_delay(array_1, array_2, 1, 1, compute_envelope=False, resampling_rate=None, plot_figure=True,
               plot_intermediate_steps=True)

    # Example 2: sine function, different frequencies
    timestamps_1 = np.linspace(0, np.pi * 2, 200001)
    array_1 = np.sin(timestamps_1)
    timestamps_2 = np.linspace(np.pi * 0.5, np.pi * 0.75, 6001)
    array_2 = np.sin(timestamps_2)

    find_delay(array_1, array_2, 100000 / np.pi, 6000 / (np.pi / 4),
               compute_envelope=False, resampling_rate=1000, number_of_windows_res=10, overlap_ratio_res=0.5,
               resampling_mode="cubic", plot_figure=True, plot_intermediate_steps=True, verbosity=1)

    # Example 3: audio files
    audio_path = "i_have_a_dream_full.wav"
    audio_wav = wavfile.read(audio_path)
    audio_frequency = audio_wav[0]
    audio_array = audio_wav[1][:, 0]  # Turn to mono

    excerpt_path = "i_have_a_dream_excerpt.wav"
    excerpt_wav = wavfile.read(excerpt_path)
    excerpt_frequency = excerpt_wav[0]
    excerpt_array = excerpt_wav[1][:, 0]  # Turn to mono

    find_delay(audio_array, excerpt_array, audio_frequency, excerpt_frequency,
               compute_envelope=True, number_of_windows_env=100, overlap_ratio_env=0.5,
               resampling_rate=1000, number_of_windows_res=10, overlap_ratio_res=0.5, return_delay_format="timedelta",
               resampling_mode="cubic", plot_figure=True, plot_intermediate_steps=True, verbosity=1)

    # Example 4: multiple audio files
    excerpt2_path = "i_have_a_dream_excerpt2.wav"
    excerpt2_wav = wavfile.read(excerpt2_path)
    excerpt2_frequency = excerpt2_wav[0]
    excerpt2_array = excerpt2_wav[1][:, 0]  # Turn to mono

    excerpt_not_present_path = "au_revoir.wav"
    excerpt_not_present_wav = wavfile.read(excerpt_not_present_path)
    excerpt_not_present_frequency = excerpt_not_present_wav[0]
    excerpt_not_present_array = excerpt_not_present_wav[1][:, 0]  # Turn to mono

    find_delays(audio_array, [excerpt_array, excerpt2_array, excerpt_not_present_array], audio_frequency,
                [excerpt_frequency, excerpt2_frequency, excerpt_not_present_frequency],
                compute_envelope=True, number_of_windows_env=100, overlap_ratio_env=0.5,
                resampling_rate=1000, number_of_windows_res=10, overlap_ratio_res=0.5, return_delay_format="timedelta",
                resampling_mode="cubic", plot_figure=True, plot_intermediate_steps=True, verbosity=1)
