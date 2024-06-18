"""Contains a series of functions directed at calculating the delay between two arrays.
* find_delay finds the delay between two time series using cross-correlation.
* find_delays does the same, but for multiple excerpts from one big time series.

Author: Romain Pastureau, BCBL (Basque Center on Cognition, Brain and Language)
Current version: 2.8 (2024-06-18)

Version history
---------------
2.8 (2024-06-19) · Added tests with random numbers
                 · Corrected the link to the documentation on the PyPI page
                 · Replaced the strings by f-strings
2.7 (2024-05-09) · Simplified `from find_delay.find_delay import find_delay` to `from find_delay import find_delay`
                 · Corrected scaling (again) on the aligned arrays graph
                 · Reestablished audio examples with downloadable WAV files when running the demo
                 · Added an example with randomly generated numbers
2.6 (2024-05-08) · Removed demo audio files to lighten the Python package; they are still available on the main branch
2.5 (2024-05-08) · Turned find_delay into a Python package, install with `py -m pip install find-delay`
2.4 (2024-05-08) · The functions now look for correlation at the edges of the first array, in the case where the second
                   array contains information that starts before the beginning, or ends after the end of the first
                 · Example 4 has been updated with one new audio file to demonstrate this change
                 · Adding a parameter x_format_figure that allows to display HH:MM:SS time on the x-axis
                 · Corrected a bug in the percentage progressions that prevented to display all the steps
                 · Added "Quick use for audio files" segment in the README file
2.3 (2024-05-02) · Corrected a bug that prevented the figures to be saved as a file
                 · Plotting without intermediate steps now plots the graphs on top of each other, not side-by-side
2.2 (2024-05-02) · Arrays with different amplitudes now appear scaled on the graph overlay
                 · Excerpt numbers now start at 1 instead of 0 on the graphs in find_delays
                 · "i_have_a_dream_excerpt2.wav" is now of lower amplitude to test the scaling on the graph overlay
2.1 (2024-04-25) · Modified the overall functions so that they take a window size instead of a number of windows
2.0 (2024-04-24) · Changed the parameter asking for a number of windows by a parameter asking for a window size instead
                 · Clarified the docstrings in the documentation of the functions
                 · Modified `find_delays` so that saving the figures would iterate the filenames instead of overwriting
                 · Modified `_get_envelope` and `_resample` so that a number of windows inferior to 1 would be set at 1
                 · Added documentation for `_create_figure` and simplified unused parameters
                 · Corrected broken figure saving
                 · Added figure saving for the 3 first examples
1.3 (2024-04-18) · Removed unused function `_get_number_of_windows`
1.2 (2024-04-17) · Added transparency of the second (orange) array on the graph overlay
                 · Clarified README.md and added figures
1.1 (2024-04-16) · Added `find_delays`
                 · Created _create_figure containing all the plotting-related code
                 · Modified the graph plot when the max correlation is below threshold
                 · Minor corrections in docstrings
1.0 (2024-04-12) · Initial release
"""

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
from scipy.signal import butter, correlate, hilbert, lfilter
import numpy as np
import datetime as dt


def _filter_frequencies(array, frequency, filter_below=None, filter_over=None, verbosity=1):
    """Applies a low-pass, high-pass or band-pass filter to the data in the attribute :attr:`samples`.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Turned into a "private" function by adding a leading underscore.

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
            print(f"\tApplying a band-pass filter for frequencies between {filter_below} and {filter_over} Hz.")
        b, a = butter(2, [filter_below, filter_over], "band", fs=frequency)
        filtered_array = lfilter(b, a, array)

    # High-pass filter
    elif filter_below not in [None, 0]:
        if verbosity > 0:
            print(f"\tApplying a high-pass filter for frequencies over {filter_below} Hz.")
        b, a = butter(2, filter_below, "high", fs=frequency)
        filtered_array = lfilter(b, a, array)

    # Low-pass filter
    elif filter_over not in [None, 0]:
        if verbosity > 0:
            print(f"\tApplying a low-pass filter for frequencies below {filter_over} Hz.")
        b, a = butter(2, filter_over, "low", fs=frequency)
        filtered_array = lfilter(b, a, array)

    else:
        filtered_array = array

    return filtered_array


def _get_number_of_windows(array_length_or_array, window_size, overlap_ratio=0, add_incomplete_window=True):
    """Given an array, calculates how many windows from the defined `window_size` can be created, with or
    without overlap.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Turned into a "private" function by adding a leading underscore.

    .. versionchanged:: 1.3
        Function temporarily removed as it was unused at the time.

    .. versionchanged:: 2.1
        Function reinstated.

    Parameters
    ----------
    array_length_or_array: list(int or float) or np.array(int or float) or int
        An array of numerical values, or its length.
    window_size: int
        The number of array elements in each window.
    overlap_ratio: int
        The ratio of array elements overlapping between each window and the next.
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

    overlap = int(np.ceil(overlap_ratio * window_size))

    if overlap_ratio >= 1:
        raise Exception(f"The size of the overlap ({overlap}) cannot be bigger than or equal to the size of the " +
                        f"window ({window_size}).")
    if window_size > array_length:
        window_size = array_length

    number_of_windows = (array_length - overlap) / (window_size - overlap)

    if add_incomplete_window and array_length + (overlap * (window_size - 1)) % window_size != 0:
        return int(np.ceil(number_of_windows))

    return int(number_of_windows)


def _get_envelope(array, frequency, window_size=1e6, overlap_ratio=0.5, filter_below=None, filter_over=None,
                  verbosity=1):
    """Calculates the envelope of an array, and returns it. The function can also optionally perform a band-pass
    filtering, if the corresponding parameters are provided.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Turned into a "private" function by adding a leading underscore.

    .. versionchanged:: 2.0
        Sets the number of windows to 1 if the parameter number_of_windows is lower than 1 (deprecated by version 2.1).

    .. versionchanged:: 2.1
        The function now takes a `window_size` parameter instead of a `number_of_windows`.

    .. versionchanged:: 2.4
        Corrected the progression percentage display.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    frequency: int or float
        The sampling frequency of the array, in Hz.

    window_size: int or None, optional
        The size of the windows (in samples) in which to cut the array to calculate the envelope. Cutting large arrays
        into windows allows to speed up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 1 million. If this parameter is set on 0,
        on None or on a number of samples bigger than the amount of elements in the array, the window size is set on
        the length of the samples.

        .. versionadded:: 2.1

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
    if window_size == 0 or window_size > len(array) or window_size is None:
        window_size = len(array)

    if overlap_ratio is None:
        overlap_ratio = 0

    window_size = int(window_size)
    overlap = int(np.ceil(overlap_ratio * window_size))
    number_of_windows = _get_number_of_windows(len(array), window_size, overlap_ratio, True)

    # Hilbert transform
    if verbosity == 0:
        print("\tGetting the Hilbert transform...", end=" ")
    elif verbosity > 1:
        print("\tGetting the Hilbert transform...")
        print(f"\t\tDividing the samples in {number_of_windows} window(s) of {window_size} samples, with an overlap " +
              f"of {overlap} samples.")

    envelope = np.zeros(len(array))
    j = 0
    next_percentage = 10

    for i in range(number_of_windows):

        if verbosity == 1:
            while i / number_of_windows > next_percentage / 100:
                print(f"{next_percentage}%", end=" ")
                next_percentage += 10

        # Get the Hilbert transform of the window
        array_start = i * (window_size - overlap)
        array_end = np.min([(i + 1) * window_size - i * overlap, len(array)])
        if verbosity > 1:
            print(f"\t\t\tGetting samples from window {i + 1}/{number_of_windows}: samples {array_start} " +
                  f"to {array_end}... ", end=" ")
        hilbert_window = np.abs(hilbert(array[array_start:array_end]))

        # Keep only the center values
        if i == 0:
            slice_start = 0
        else:
            slice_start = overlap // 2  # We stop one before if the overlap is odd

        if i == number_of_windows - 1:
            slice_end = len(hilbert_window)
        else:
            slice_end = window_size - int(np.ceil(overlap / 2))

        if verbosity > 1:
            print(f"\n\t\t\tKeeping the samples from {slice_start} to {slice_end} in the window: samples " +
                  f"{array_start + slice_start} to {array_start + slice_end}...", end=" ")

        preserved_samples = hilbert_window[slice_start:slice_end]
        envelope[j:j + len(preserved_samples)] = preserved_samples
        j += len(preserved_samples)

        if verbosity > 1:
            print("Done.")

    if verbosity == 1:
        while 1 > next_percentage / 100:
            print(f"{next_percentage}%", end=" ")
            next_percentage += 10
        print("100% - Done.")
    elif verbosity > 1:
        print("Done.")

    # Filtering
    if filter_below is not None or filter_over is not None:
        envelope = _filter_frequencies(envelope, frequency, filter_below, filter_over, verbosity)

    if verbosity > 0:
        print(f"\tEnvelope calculated in: {dt.datetime.now() - time_before}")

    return envelope


def _resample_window(array, original_timestamps, resampled_timestamps, index_start_original, index_end_original,
                     index_start_resampled, index_end_resampled, method="cubic", verbosity=1):
    """Performs and returns the resampling on a subarray of samples.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Turned into a "private" function by adding a leading underscore.

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
        print(f"\t\t\t\tIn the original array, the window contains samples {index_start_original} to " +
              f"{index_end_original} (from timestamps {original_timestamps_window[0]} to " +
              f"{original_timestamps_window[-1]}).")
        print(f"\t\t\t\tIn the new array, the window contains samples {index_start_resampled} to "
              f"{index_end_resampled} (from timestamps {resampled_timestamps_window[0]} to "
              f"{resampled_timestamps_window[-1]}).")
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
        raise Exception(f"Invalid resampling method: {method}.")


def _resample(array, original_frequency, resampling_frequency, window_size=1e7, overlap_ratio=0.5,
              method="cubic", verbosity=1):
    """Resamples an array to the `resampling_frequency` parameter. It first creates a new set of timestamps at the
    desired frequency, and then interpolates the original data to the new timestamps.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Turned into a "private" function by adding a leading underscore.

    .. versionchanged:: 2.0
        Sets the number of windows to 1 if the parameter number_of_windows is lower than 1 (deprecated).

    .. versionchanged:: 2.1
        The function now takes a `window_size` parameter instead of a `number_of_windows`.

    .. versionchanged:: 2.4
        Corrected the progression percentage display.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    original_frequency: int or float
        The sampling frequency of the array, in Hz.

    resampling_frequency: int or float
        The frequency at which you want to resample the array, in Hz. A frequency of 4 will return samples
        at 0.25 s intervals.

    window_size: int or None, optional
        The size of the windows (in samples) in which to cut the array before resampling. Cutting large arrays
        into windows allows to speed up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 10 million. If this parameter is set on
        0, on None or on a number of samples bigger than the amount of elements in the array, the window size is set on
        the length of the samples.

        .. versionadded:: 2.1

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
            print(f"\tNot performing the resampling as the resampling frequency is the same as the original frequency" +
                  f" ({resampling_frequency} Hz).")
        return array

    if verbosity > 0:
        print(f"\tResampling the array at {resampling_frequency} Hz (mode: {method})...")
        print(f"\t\tOriginal frequency: {round(original_frequency, 2)} Hz.")
        if verbosity > 1:
            print("\t\tPerforming the resampling...")
        else:
            print("\t\tPerforming the resampling...", end=" ")

    if method == "take":
        if resampling_frequency > original_frequency:
            raise Exception(f"The mode \"take\" does not allow for upsampling of the data. Please input a resampling " +
                            f"frequency inferior to the original ({original_frequency}).")
        factor_resampling = original_frequency / resampling_frequency
        print(factor_resampling)
        if factor_resampling != int(factor_resampling):
            print(f"Warning: The downsampling factor is not an integer ({factor_resampling}), meaning that the " +
                  f"downsampling may not be accurate. To ensure an accurate resampling with the \"take\" mode, use" +
                  f" a resampling frequency that is an integer divider of the original frequency " +
                  f"({original_frequency} Hz).")
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
        if window_size == 0 or window_size > len(array) or window_size is None:
            window_size = len(array)

        if overlap_ratio is None:
            overlap_ratio = 0

        window_size = int(window_size)
        overlap = int(np.ceil(overlap_ratio * window_size))
        number_of_windows = _get_number_of_windows(len(array), window_size, overlap_ratio, True)

        if verbosity > 1 and number_of_windows != 1:
            print(f"\t\t\tCreating {number_of_windows} window(s), each containing {window_size} samples, with a " +
                  f"{round(overlap_ratio * 100, 2)}% overlap ({overlap} samples).")

        resampled_array = np.zeros(np.size(resampled_timestamps))
        j = 0
        next_percentage = 10

        for i in range(number_of_windows):

            window_start_original = i * (window_size - overlap)
            window_end_original = np.min([(i + 1) * window_size - i * overlap, len(original_timestamps) - 1])
            window_start_resampled = int(np.round(original_timestamps[window_start_original] * resampling_frequency))
            window_end_resampled = int(np.round(original_timestamps[window_end_original] * resampling_frequency))

            if verbosity == 1:
                while i / number_of_windows > next_percentage / 100:
                    print(f"{next_percentage}%", end=" ")
                    next_percentage += 10

            if verbosity > 1:
                print(f"\t\t\tGetting samples from window {i + 1}/{number_of_windows}.")

            resampled_window = _resample_window(array, original_timestamps, resampled_timestamps, window_start_original,
                                                window_end_original, window_start_resampled, window_end_resampled,
                                                method, verbosity)

            if verbosity > 1:
                print(f"Done.\n\t\t\t\tThe resampled window contains {np.size(resampled_window)} sample(s).")

            # Keep only the center values
            if i == 0:
                window_slice_start = 0
                resampled_slice_start = 0
            else:
                window_slice_start = (j - window_start_resampled) // 2
                resampled_slice_start = window_start_resampled + window_slice_start

            preserved_samples = resampled_window[window_slice_start:]

            if verbosity > 1:
                print(f"\t\t\tKeeping the samples from {resampled_slice_start} to " +
                      f"{resampled_slice_start + len(preserved_samples) - 1} (in the window, samples " +
                      f"{window_slice_start} to {len(resampled_window) - 1})", end="")
                if resampled_slice_start != j:
                    print(f" · Rewriting samples {resampled_slice_start} to {j - 1}...", end=" ")
                else:
                    print("...", end=" ")

            resampled_array[resampled_slice_start:resampled_slice_start + len(preserved_samples)] = preserved_samples
            j = resampled_slice_start + len(preserved_samples)

            if verbosity > 1:
                print("Done.")

        if verbosity == 1:
            while 1 > next_percentage / 100:
                print(f"{next_percentage}%", end=" ")
                next_percentage += 10
            print("100% - Done.")
        elif verbosity > 1:
            print("\t\tResampling done.")

    if verbosity > 0:
        print(f"\t\tThe original array had {len(array)} samples.")
        print(f"\t\tThe new array has {len(resampled_array)} samples.")
        print(f"\tResampling performed in: {dt.datetime.now() - time_before}")

    return resampled_array


def _create_figure(array_1, array_2, freq_array_1, freq_array_2, name_array_1, name_array_2, envelope_1, envelope_2,
                   y1, y2, compute_envelope, window_size_env, overlap_ratio_env, filter_below, filter_over,
                   resampling_rate, window_size_res, overlap_ratio_res, cross_correlation, threshold,
                   number_of_plots, return_delay_format, return_value, max_correlation_value,
                   index_max_correlation_value, plot_figure, path_figure, name_figure, plot_intermediate_steps,
                   x_format_figure, verbosity):
    """
    Creates and/or saves a figure given the parameters of the find_delay function.

    .. versionadded:: 1.1

    .. versionchanged:: 1.2
        Added transparency in the graph overlay.

    .. versionchanged:: 2.0
        Modified subplot titles to reflect changes of separated number of windows between both arrays.
        Corrected the figure saving to a file and prevented overwriting the same figure using find_delays.

    .. versionchanged:: 2.2
        Arrays with different amplitudes now appear scaled on the aligned arrays graph.
        Added a second y axis to the aligned arrays graph.

    .. versionchanged:: 2.3
        Corrected the figure saving to a file.
        If `plot_intermediate_steps` is `False`, the two graphs "cross-correlation" and "aligned arrays" are now on top
        of each other instead of side-by-side.

    .. versionchanged:: 2.4
        Added the new parameter `x_format_figure`, allowing to have HH:MM:SS times on the x axis.
        Modified the scaling of the aligned arrays figure to be more accurate.

    array_1: np.array
        The first array involved in the cross-correlation.
    array_2: np.array
        The second array involved in the cross-correlation, being allegedly an excerpt from the first.
    freq_array_1: int or float
        The sampling rate of `array_1`.
    freq_array_2: int or float
        The sampling rate of `array_2`.
    name_array_1: str
        The name of the first array; will be "Array 1" for `find_delay` and "Array" for `find_delays`.
    name_array_2: str
        The name of the second array; will be "Array 2" for `find_delay` and "Excerpt n" for `find_delays`, with n
        being the index of the excerpt in the folder.
    envelope_1: np.array
        The envelope of `array_1` (if calculated).
    envelope_2: np.array
        The envelope of `array_2` (if calculated).
    y1: np.array
        The resampled `array_1` or `envelope_1` (if calculated).
    y2: np.array
        The resampled `array_2` or `envelope_2` (if calculated).
    compute_envelope: bool
        A boolean describing if the envelope has been computed or not.
    window_size_env: int
        The size of the windows in which the arrays were cut for the envelope calculation.
    overlap_ratio_env: float
        The ratio of overlapping between each envelope window with the previous and the next windows.
    filter_below: int, float or None
        The lower limit of the bandpass filter applied to the envelopes.
    filter_over: int, float or None
        The upper limit of the bandpass filter applied to the envelopes.
    resampling_rate: int, float or None
        The rate at which the arrays or the envelopes have been resampled.
    window_size_res: int
        The size of the windows in which the arrays or envelopes were cut for the resampling.
    overlap_ratio_res: float
        The ratio of overlapping between each resampling window with the previous and the next windows.
    cross-correlation: np.array
        The array containing the correlation values for each lag between the two arrays.
    threshold: float
        The threshold of the maximum correlation value between the two arrays, relative to the maximum
        correlation value between the excerpt and itself.
    return_delay_format: str
        Indicates the format of the displayed delay, either ``"index"``, ``"ms"``, ``"s"``, or ``"timedelta"``.
    return_value: int, float or timedelta
        The value of the delay in the format specified by the previous parameter.
    max_correlation_value: float
        The maximum correlation value from the cross-correlation.
    index_max_correlation_value: int
        The index at which the maximum correlation value can be found in the cross-correlation array.
    plot_figure: bool
        If set on `True`, plots the figure in a Matplotlib window.
    path_figure: str or None
        If set, saves the figure at the given path.
    name_figure: str or None
        If set, considers that `path_figure` is the directory where to save the figure, and `name_figure` is the name
        of the file.
    plot_intermediate_steps: bool
        If set on `True`, plots the original audio clips, the envelopes and the resampled arrays (if calculated) besides
        the cross-correlation.
    x_format_figure: str
        If set on `"time"`, the values on the x axes of the output will take the HH:MM:SS format (or MM:SS if the time
        series are less than one hour long). If set on `"float"`, the values on the x axes will be displayed as float
        (unit: second). If set on `"auto"` (default), the format of the values on the x axes will be defined depending
        on the value of `return_delay_format`.
    verbosity: int
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.
    """

    # Figure creation
    if plot_intermediate_steps:
        fig, ax = plt.subplots(int(np.ceil(number_of_plots / 2)), 2, constrained_layout=True, figsize=(16, 8))
    else:
        fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(16, 8))

    # Defining the timestamps
    t_array_1 = np.arange(0, len(array_1)) / freq_array_1
    t_array_2 = np.arange(0, len(array_2)) / freq_array_2
    t_res_1 = None
    t_res_2 = None
    if resampling_rate is not None:
        t_res_1 = np.arange(0, len(y1)) / resampling_rate
        t_res_2 = np.arange(0, len(y2)) / resampling_rate
        t_res_2_aligned = t_array_2 + index_max_correlation_value / resampling_rate
        t_cc = (np.arange(0, len(cross_correlation)) - y2.size + 1) / resampling_rate
    else:
        t_res_2_aligned = t_array_2 + index_max_correlation_value / freq_array_2
        t_cc = (np.arange(0, len(cross_correlation)) - y2.size + 1) / freq_array_1

    # Set default x_format_figure and setting timestamps to be datetime
    if x_format_figure == "auto" and return_delay_format in ["s", "ms", "timedelta"]:
        x_format_figure = "time"

        t_array_1 = np.array(t_array_1 * 1000000, dtype="datetime64[us]")
        t_array_2 = np.array(t_array_2 * 1000000, dtype="datetime64[us]")
        t_res_1 = np.array(t_res_1 * 1000000, dtype="datetime64[us]")
        t_res_2 = np.array(t_res_2 * 1000000, dtype="datetime64[us]")
        t_res_2_aligned = np.array(t_res_2_aligned * 1000000, dtype="datetime64[us]")
        t_cc = np.array(t_cc * 1000000, dtype="datetime64[us]")

    # Formatting functions for the x axis (MM:SS and HH:MM:SS)
    def get_label(value, include_hour=True, include_us=True):
        """Returns a label value depending on the selected parameters."""

        neg = False
        # If negative, put positive
        if value < 0:
            neg = True
            value = abs(value)

        # If zero, set zero
        elif value == 0:
            if include_hour:
                return "00:00:00"
            else:
                return "00:00"

        # Turn to timedelta
        td_value = mdates.num2timedelta(value)

        seconds = td_value.total_seconds()
        hh = str(int(seconds // 3600)).zfill(2)
        mm = str(int((seconds // 60) % 60)).zfill(2)
        ss = str(int(seconds % 60)).zfill(2)

        us = str(int((seconds % 1) * 1000000)).rstrip("0")

        label = ""
        if neg:
            label += "-"
        if include_hour:
            label += hh + ":"
        label += mm + ":" + ss
        if include_us and us != "":
            label += "." + us

        return label

    def get_label_hh_mm_ss_no_ms(value, pos=None):
        """Returns a label value as HH:MM:SS, without any ms value."""
        return get_label(value, True, False)

    def get_label_hh_mm_ss(value, pos=None):
        """Returns a label value as HH:MM:SS.ms, without any trailing zero."""
        return get_label(value, True, True)

    def set_label_time_figure(ax):
        """Sets the time formatted labels on the x axes."""
        if x_format_figure == "time":
            formatter = mdates.AutoDateFormatter(ax.xaxis.get_major_locator())
            formatter.scaled[1 / mdates.MUSECONDS_PER_DAY] = get_label_hh_mm_ss
            formatter.scaled[1 / mdates.SEC_PER_DAY] = get_label_hh_mm_ss
            formatter.scaled[1 / mdates.MINUTES_PER_DAY] = get_label_hh_mm_ss_no_ms
            formatter.scaled[1 / mdates.HOURS_PER_DAY] = get_label_hh_mm_ss_no_ms
            formatter.scaled[1] = get_label_hh_mm_ss_no_ms
            formatter.scaled[mdates.DAYS_PER_MONTH] = get_label_hh_mm_ss_no_ms
            formatter.scaled[mdates.DAYS_PER_YEAR] = get_label_hh_mm_ss_no_ms
            ax.xaxis.set_major_formatter(formatter)
            return ax

        return ax

    i = 0

    if plot_intermediate_steps:
        # Original arrays
        ax[i // 2][i % 2].set_title(name_array_1 + ": " + str(round(freq_array_1, 2)) + " Hz")
        ax[i // 2][i % 2].plot(t_array_1, array_1)
        ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
        i += 1

        ax[i // 2][i % 2].set_title(name_array_2 + ": " + str(round(freq_array_2, 2)) + " Hz")
        ax[i // 2][i % 2].plot(t_array_2, array_2, color="orange")
        ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
        i += 1

        # Envelopes
        if compute_envelope:
            band_pass_low = "0" if filter_below is None else filter_below
            band_pass_high = "∞" if filter_over is None else filter_over

            number_of_windows = _get_number_of_windows(len(array_1), window_size_env, overlap_ratio_env)
            title = "Envelope " + name_array_1 + ": " + str(freq_array_1) + "Hz, " + str(number_of_windows) + " w, " + \
                    str(overlap_ratio_env) + " o"
            if filter_below is not None or filter_over is not None:
                title += " · Band-pass [" + str(band_pass_low) + ", " + str(band_pass_high) + "]"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(t_array_1, envelope_1)
            ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
            i += 1

            number_of_windows = _get_number_of_windows(len(array_2), window_size_env, overlap_ratio_env)
            title = "Envelope " + name_array_2 + ": " + str(freq_array_2) + "Hz, " + str(number_of_windows) + " w, " + \
                    str(overlap_ratio_env) + " o"
            if filter_below is not None or filter_over is not None:
                title += " · Band-pass [" + str(band_pass_low) + ", " + str(band_pass_high) + "]"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(t_array_2, envelope_2, color="orange")
            ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
            i += 1

        # Resampled arrays
        if resampling_rate is not None:
            if compute_envelope is False:
                env_or_array = ""
            else:
                env_or_array = "Envelope "

            number_of_windows = _get_number_of_windows(len(envelope_1), window_size_res, overlap_ratio_res)
            title = "Resampled " + env_or_array + name_array_1 + ": " + str(resampling_rate) + " Hz, " + \
                    str(number_of_windows) + " w, " + str(overlap_ratio_res) + " o"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(t_res_1, y1)
            ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
            i += 1

            number_of_windows = _get_number_of_windows(len(envelope_2), window_size_res, overlap_ratio_res)
            title = "Resampled " + env_or_array + name_array_2 + ": " + str(resampling_rate) + " Hz, " + \
                    str(number_of_windows) + " w, " + str(overlap_ratio_res) + " o"
            ax[i // 2][i % 2].set_title(title)
            ax[i // 2][i % 2].plot(t_res_2, y2, color="orange")
            ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
            i += 1

    # Cross-correlation
    title = "Cross-correlation"

    if plot_intermediate_steps:
        ax[i // 2][i % 2].set_title(title)
        ax[i // 2][i % 2].set_ylim(np.nanmin(cross_correlation), 1.5)
        ax[i // 2][i % 2].plot(t_cc, cross_correlation, color="green")
        ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
    else:
        ax[0].set_title(title)
        ax[0].set_ylim(np.nanmin(cross_correlation), 1.5)
        ax[0].plot(t_cc, cross_correlation, color="green")
    text = ""
    if return_delay_format == "index":
        text = "Sample "
    text += str(return_value)
    if return_delay_format in ["ms", "s"]:
        text += return_delay_format

    if max_correlation_value >= threshold:
        text += " · Correlation value: " + str(round(max_correlation_value, 3))
        bbox_props = dict(boxstyle="square,pad=0.3", fc="#99cc00", ec="k", lw=0.72)
    else:
        text += " · Correlation value (below threshold): " + str(round(max_correlation_value, 3))
        bbox_props = dict(boxstyle="square,pad=0.3", fc="#ff0000", ec="k", lw=0.72)
    arrow_props = dict(arrowstyle="->", connectionstyle="angle,angleA=90")
    kw = dict(xycoords='data', textcoords="data",
              arrowprops=arrow_props, bbox=bbox_props, ha="center", va="center")
    if plot_intermediate_steps:
        ax[i // 2][i % 2].annotate(text, xy=(t_cc[index_max_correlation_value + y2.size - 1], max_correlation_value),
                                   xytext=(t_cc[index_max_correlation_value + y2.size - 1], 1.4), **kw)
    else:
        ax[0].annotate(text, xy=(t_cc[index_max_correlation_value + y2.size - 1], max_correlation_value),
                       xytext=(t_cc[index_max_correlation_value + y2.size - 1], 1.4), **kw)

    i += 1

    # Aligned arrays
    if plot_intermediate_steps:
        ax[i // 2][i % 2].set_title("Aligned arrays")
        ax[i // 2][i % 2].plot(t_array_1, array_1, color="#04589388", linewidth=1)
        ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
        ax[i // 2][i % 2].tick_params(axis='y', labelcolor="#045893")
        ax[i // 2][i % 2].set_ylabel(name_array_1, color="#045893")
        ylim = ax[i // 2][i % 2].get_ylim()
        ax2 = ax[i // 2][i % 2].twinx()
    else:
        ax[1].set_title("Aligned arrays")
        ax[1].plot(t_array_1, array_1, color="#04589388", linewidth=1)
        ax[1] = set_label_time_figure(ax[1])
        ax[1].tick_params(axis='y', labelcolor="#045893")
        ax[1].set_ylabel(name_array_1, color="#045893")
        ylim = ax[1].get_ylim()
        ax2 = ax[1].twinx()

    if resampling_rate is None:
        excerpt_in_original = array_1[index_max_correlation_value:
                                      index_max_correlation_value + len(array_2)]
    else:
        index = int(index_max_correlation_value * freq_array_1 / resampling_rate)
        excerpt_in_original = array_1[index:index + int(len(array_2) * freq_array_1 / freq_array_2)]
    resampled_timestamps_array2 = t_res_2_aligned[:len(array_2)]

    min_excerpt_in_original = np.nanmin(excerpt_in_original)
    max_excerpt_in_original = np.nanmax(excerpt_in_original)

    if min_excerpt_in_original != 0:
        min_ratio = np.nanmin(array_2) / min_excerpt_in_original
    else:
        min_ratio = 0

    if max_excerpt_in_original != 0:
        max_ratio = np.nanmax(array_2) / max_excerpt_in_original
    else:
        max_ratio = 0

    ratio = np.nanmax([min_ratio, max_ratio])

    ax2.plot(resampled_timestamps_array2, array_2, color="#ffa500aa", linewidth=2)
    ax2.set_ylim((ylim[0] * ratio, ylim[1] * ratio))
    ax2.tick_params(axis='y', labelcolor="#ffa500")
    ax2.set_ylabel(name_array_2, color="#ffa500")

    if path_figure is not None:
        if name_figure is not None:
            if verbosity > 0:
                print(f"\nSaving the graph under {path_figure}/{name_figure}...", end=" ")
            plt.savefig(str(path_figure) + "/" + str(name_figure))
        else:
            if verbosity > 0:
                print(f"\nSaving the graph under {path_figure}...", end=" ")
            plt.savefig(str(path_figure))
        if verbosity > 0:
            print("Done.")

    if plot_figure:
        if verbosity > 0:
            print("\nShowing the graph...")
        plt.show()


def find_delay(array_1, array_2, freq_array_1=1, freq_array_2=1, compute_envelope=True, window_size_env=1e6,
               overlap_ratio_env=0.5, filter_below=None, filter_over=50, resampling_rate=None,
               window_size_res=1e7, overlap_ratio_res=0.5, resampling_mode="cubic",
               return_delay_format="index", return_correlation_value=False, threshold=0.9,
               plot_figure=False, plot_intermediate_steps=False, x_format_figure="auto", path_figure=None, verbosity=1):
    """This function tries to find the timestamp at which an excerpt (array_2) begins in a time series (array_1).
    The computation is performed through cross-correlation. Before so, the envelopes of both arrays can first be
    calculated and filtered (recommended for audio files), and resampled (necessary when the sampling rate of the two
    arrays is unequal). The function returns the timestamp of the maximal correlation value, or `None` if this value is
    below threshold. Optionally, it can also return a second element, the maximal correlation value.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Separated the figure generation into a new function `_create_figure`.

    .. versionchanged:: 2.0
        Changed parameters `number_of_windows` to `window_size`.

    .. versionchanged:: 2.1
        Decreased default `window_size_res` value from 1e8 to 1e7.

    .. versionchanged:: 2.3
        Corrected the figure saving to a file.

    .. versionchanged:: 2.4
        Modified the cross-correlation to look for the excerpt at the edges of the first array.
        Added the new parameter `x_format_figure`, allowing to have HH:MM:SS times on the x axis.

    Important
    ---------
    Because it is easy to get confused: this function returns the timestamp in array_1 where array_2 begins. This means
    that, if you want to align array_1 and array_2, you need to remove the delay to each timestamp of array_1: that way,
    the value at timestamp 0 in array_1 will be aligned with the value at timestamp 0 in array_2.

    Note
    ----
    Since version 2.4, this function can find excerpts containing data that would be present outside the main array. In
    other words, if the excerpt starts 1 second before the onset of the original array, the function will return a delay
    of -1 sec. However, this should be avoided, as information missing from the original array will result in lower
    correlation - with a substantial amount of data missing from the original array, the function may return erroneous
    results. This is why it is always preferable to use excerpts that are entirely contained in the original array.

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

    window_size_env: int or None, optional
        The size of the windows in which to cut the arrays to calculate the envelope. Cutting long arrays
        in windows allows to speed up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 1 million.

        .. versionadded:: 2.0

    overlap_ratio_env: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap. By default, this parameter is set on 0.5, meaning that each
        window will overlap for half of their values with the previous, and half of their values with the next.

    filter_below: int or None, optional
        If set, a high-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        0 Hz).

    filter_over: int or None, optional
        If set, a low-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        50 Hz).

    resampling_rate: int or None, optional
        The sampling rate at which to downsample the arrays for the cross-correlation. A larger value will result in
        longer computation times. Setting the parameter on `None` will not downsample the arrays,
        which will result in an error if the two arrays are not the same frequency. If this parameter is `None`, the
        next parameters related to resampling can be ignored. A recommended value for this parameter when working with
        audio files is 1000, as it will speed up the computation of the cross-correlation while still giving a
        millisecond-precision delay.

    window_size_res: int or None, optional
        The size of the windows in which to cut the arrays. Cutting lo,g arrays in windows allows to speed up the
        computation. If this parameter is set on `None`, the window size will be set on the number of samples. A good
        value for this parameter is generally 1e7.

        .. versionadded:: 2.0

        .. versionchanged:: 2.1
            Decreased default `window_size_res` value from 1e8 to 1e7.

    overlap_ratio_res: float or None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap. By default, this parameter is set on 0.5, meaning that each window will
        overlap for half of their values with the previous, and half of their values with the next.

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

    x_format_figure: str
        If set on `"time"`, the values on the x axes of the output will take the HH:MM:SS format (or MM:SS if the time
        series are less than one hour long). If set on `"float"`, the values on the x axes will be displayed as float
        (unit: second). If set on `"auto"` (default), the format of the values on the x axes will be defined depending
        on the value of `return_delay_format`.

         .. versionadded:: 2.4

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
        print(f"\t The first array contains {len(array_1)} samples, at a rate of {freq_array_1} Hz.")
        print(f"\t The second array contains {len(array_2)} samples, at a rate of {freq_array_2} Hz.\n")

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
        envelope_1 = _get_envelope(array_1, freq_array_1, window_size_env, overlap_ratio_env, filter_below, filter_over,
                                   verbosity)
        if verbosity > 0:
            print("Getting the envelope from array 2...")
        envelope_2 = _get_envelope(array_2, freq_array_2, window_size_env, overlap_ratio_env, filter_below, filter_over,
                                   verbosity)
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
        y1 = _resample(envelope_1, freq_array_1, resampling_rate, window_size_res, overlap_ratio_res, resampling_mode,
                       verbosity)
        if verbosity > 0:
            print("Resampling array 2...")
        y2 = _resample(envelope_2, freq_array_2, resampling_rate, window_size_res, overlap_ratio_res, resampling_mode,
                       verbosity)
        if verbosity > 0:
            print("Resampling done.\n")
    else:
        rate = freq_array_1
        if freq_array_1 != freq_array_2:
            raise Exception(f"The rate of the two arrays you are trying to correlate are different ({freq_array_1} Hz" +
                            f" and {freq_array_2} Hz). You must indicate a resampling rate to perform the " +
                            f"cross-correlation.")
        y1 = envelope_1
        y2 = envelope_2

    time_before_cross_correlation = dt.datetime.now()

    if verbosity > 0:
        print("Computing the correlation...", end=" ")

    y2_normalized = (y2 - y2.mean()) / y2.std() / np.sqrt(y2.size)
    y1_m = correlate(y1, np.ones(y2.size), "full") ** 2 / y2_normalized.size
    y1_m2 = correlate(y1 ** 2, np.ones(y2.size), "full")
    cross_correlation = correlate(y1, y2_normalized, "full") / np.sqrt(y1_m2 - y1_m)
    max_correlation_value = np.nanmax(cross_correlation)

    index_max_correlation_value = np.nanargmax(cross_correlation) - y2.size + 1
    index = int(np.round(index_max_correlation_value * freq_array_1 / rate, 0))
    delay_in_seconds = index_max_correlation_value / rate
    t = dt.timedelta(days=delay_in_seconds // 86400, seconds=int(delay_in_seconds % 86400),
                     microseconds=(delay_in_seconds % 1) * 1000000)

    if verbosity > 0:
        print("Done.")
        print(f"\tCross-correlation calculated in: {dt.datetime.now() - time_before_cross_correlation}")

        if max_correlation_value >= threshold:
            print(f"\tMaximum correlation ({np.round(max_correlation_value, 3)}) found at sample {index} " +
                  f"(timestamp {t}).")

        else:
            print(f"\tNo correlation over threshold found (max correlation: {np.round(max_correlation_value, 3)} " +
                  f") found at sample {index} (timestamp {t}).")

        print(f"\nComplete delay finding function executed in: {dt.datetime.now() - time_before_function}")

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
        raise Exception(f"Wrong value for the parameter return_delay_format: {return_delay_format}. The value should " +
                        f"be either \"index\", \"ms\", \"s\" or \"timedelta\".")

    # Plot and/or save the figure
    if plot_figure is not None or path_figure is not None:
        _create_figure(array_1, array_2, freq_array_1, freq_array_2, "Array 1", "Array 2", envelope_1, envelope_2,
                       y1, y2, compute_envelope, window_size_env, overlap_ratio_env, filter_below, filter_over,
                       resampling_rate, window_size_res, overlap_ratio_res, cross_correlation, threshold,
                       number_of_plots, return_delay_format, return_value, max_correlation_value,
                       index_max_correlation_value, plot_figure, path_figure, None, plot_intermediate_steps,
                       x_format_figure, verbosity)

    if max_correlation_value >= threshold:

        if return_correlation_value:
            return return_value, max_correlation_value
        else:
            return return_value

    else:

        if return_correlation_value:
            return None, None
        else:
            return None


def find_delays(array, excerpts, freq_array=1, freq_excerpts=1, compute_envelope=True,
                window_size_env=1e6, overlap_ratio_env=0.5, filter_below=None, filter_over=50,
                resampling_rate=None, window_size_res=1e7, overlap_ratio_res=0.5, resampling_mode="cubic",
                return_delay_format="index", return_correlation_values=False, threshold=0.9,
                plot_figure=False, plot_intermediate_steps=False, x_format_figure="auto", path_figures=None,
                name_figures="figure", verbosity=1):
    """This function tries to find the timestamp at which multiple excerpts begins in an array.
    The computation is performed through cross-correlation. Before so, the envelopes of both arrays can first be
    calculated and filtered (recommended for audio files), and resampled (necessary when the sampling rate of the two
    arrays is unequal). The function returns the timestamp of the maximal correlation value, or `None` if this value is
    below threshold. Optionally, it can also return a second element, the maximal correlation value.

    .. versionadded:: 1.1

    .. versionchanged:: 2.0
        Changed parameters `number_of_windows` to `window_size`.
        Modified the name of the parameter `path_figure` by `path_figures`.
        Added the parameter `name_figures`.

    .. versionchanged:: 2.1
        Decreased default `window_size_res` value from 1e8 to 1e7.

    .. versionchanged:: 2.2
        Figure names index now start at 1 instead of 0.

    .. versionchanged:: 2.3
        Corrected the figure saving to a file.

    .. versionchanged:: 2.4
        Modified the cross-correlation to look for the excerpt at the edges of the first array.
        Added the new parameter `x_format_figure`, allowing to have HH:MM:SS times on the x axis.

    Important
    ---------
    Because it is easy to get confused: this function returns the timestamp in array where each excerpt begins. This
    means that, if you want to align the array and an excerpt, you need to remove the delay to each timestamp of the
    array: that way, the value at timestamp 0 in the array will be aligned with the value at timestamp 0 in the excerpt.

    Note
    ----
    Since version 2.4, this function can find excerpts containing data that would be present outside the main array. In
    other words, if the excerpt starts 1 second before the onset of the original array, the function will return a delay
    of -1 sec. However, this should be avoided, as information missing from the original array will result in lower
    correlation - with a substantial amount of data missing from the original array, the function may return erroneous
    results. This is why it is always preferable to use excerpts that are entirely contained in the original array.

    Note
    ----
    Compared to find_delay (without an "s"), this function allows to compute only once the envelope of the first array,
    allowing to gain computation time.

    Parameters
    ----------
    array: list or np.ndarray
        An array of samples.

    excerpts: list(list or np.ndarray)
        A list of excerpts, each being an array of samples. Each excerpt should be smaller than or of equal size to the
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

    window_size_env: int or None, optional
        The size of the windows in which to cut the arrays to calculate the envelope. Cutting long arrays
        in windows allows to speed up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 1 million.

        .. versionadded:: 2.0

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

    window_size_res: int or None, optional
        The size of the windows in which to cut the arrays. Cutting lo,g arrays in windows allows to speed up the
        computation. If this parameter is set on `None`, the window size will be set on the number of samples. A good
        value for this parameter is generally 1e7.

        .. versionadded:: 2.0

        .. versionchanged:: 2.1
            Decreased default `window_size_res` value from 1e8 to 1e7.

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

    path_figures: str or None, optional
        If set, saves the figures in the given directory.

    x_format_figure: str
        If set on `"time"`, the values on the x axes of the output will take the HH:MM:SS format (or MM:SS if the time
        series are less than one hour long). If set on `"float"`, the values on the x axes will be displayed as float
        (unit: second). If set on `"auto"` (default), the format of the values on the x axes will be defined depending
        on the value of `return_delay_format`.

        .. versionadded:: 2.4

    name_figures: str, optional
        The name to give to each figure in the directory set by `path_figures`. The figures will be found in
        `path_figures/name_figures_n.png`, where n is the index of the excerpt in `excerpts`, starting at 1.

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
        print(f"\t The main array contains {len(array)} samples, at a rate of {freq_array} Hz.")
        print(f"\t{len(excerpts)} excerpts to find.")

    # Turn list into ndarray
    if isinstance(array, list):
        array = np.array(array)

    # Check that the length of the excerpts equals the length of the frequencies
    if isinstance(freq_excerpts, list):
        if len(excerpts) != len(freq_excerpts):
            raise Exception(f"The number of frequencies given for the excerpts ({len(freq_excerpts)}) " +
                            f"is inconsistent with the number of excerpts ({len(excerpts)}).")

    number_of_plots = 2
    if plot_intermediate_steps:
        number_of_plots += 2

    # Envelope
    if compute_envelope:
        if plot_intermediate_steps:
            number_of_plots += 2
        if verbosity > 0:
            print("Getting the envelope from the array...")
        envelope_array = _get_envelope(array, freq_array, window_size_env, overlap_ratio_env, filter_below, filter_over,
                                       verbosity)
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
        y1 = _resample(envelope_array, freq_array, resampling_rate, window_size_res, overlap_ratio_res, resampling_mode,
                       verbosity)
        if verbosity > 0:
            print("Resampling done.\n")
    else:
        rate = freq_array
        y1 = envelope_array

    for i in range(len(excerpts)):

        # Introduction
        if verbosity > 0:
            print(f"Excerpt {i + 1}/{len(excerpts)}")

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
            print(f"\t The excerpt contains {len(excerpt)} samples, at a rate of {freq_excerpt} Hz.\n")

        # Envelope
        if compute_envelope:
            if verbosity > 0:
                print(f"Getting the envelope from the excerpt {i+1}...")
            envelope_excerpt = _get_envelope(excerpt, freq_excerpt, window_size_env, overlap_ratio_env, filter_below,
                                             filter_over, verbosity)
            if verbosity > 0:
                print("Envelope calculated.\n")
        else:
            envelope_excerpt = excerpt

        # Resampling
        if resampling_rate is not None:
            rate = resampling_rate
            if verbosity > 0:
                print("Resampling excerpt...")
            y2 = _resample(envelope_excerpt, freq_excerpt, resampling_rate, window_size_res, overlap_ratio_res,
                           resampling_mode, verbosity)
            if verbosity > 0:
                print("Resampling done.\n")
        else:
            if freq_array != freq_excerpt:
                raise Exception(f"The rate of the two arrays you are trying to correlate are different ({freq_array} " +
                                f"Hz and {freq_excerpt} Hz). You must indicate a resampling rate to perform the " +
                                f"cross-correlation.")
            y2 = envelope_excerpt

        time_before_cross_correlation = dt.datetime.now()

        if verbosity > 0:
            print("Computing the correlation...", end=" ")

        y2_normalized = (y2 - y2.mean()) / y2.std() / np.sqrt(y2.size)
        y1_m = correlate(y1, np.ones(y2.size), "full") ** 2 / y2_normalized.size
        y1_m2 = correlate(y1 ** 2, np.ones(y2.size), "full")
        cross_correlation = correlate(y1, y2_normalized, "full") / np.sqrt(y1_m2 - y1_m)
        max_correlation_value = np.nanmax(cross_correlation)

        index_max_correlation_value = np.nanargmax(cross_correlation) - y2.size + 1
        index = int(np.round(index_max_correlation_value * freq_array / rate, 0))
        delay_in_seconds = index_max_correlation_value / rate
        t = dt.timedelta(days=delay_in_seconds // 86400, seconds=int(delay_in_seconds % 86400),
                         microseconds=(delay_in_seconds % 1) * 1000000)

        if verbosity > 0:
            print("Done.")
            print(f"\tCross-correlation calculated in: {dt.datetime.now() - time_before_cross_correlation}")

            if max_correlation_value >= threshold:
                print(f"\tMaximum correlation ({np.round(max_correlation_value, 3)}) found at sample {index} " +
                      f"(timestamp {t}).")

            else:
                print(f"\tNo correlation over threshold found (max correlation: {np.round(max_correlation_value, 3)})" +
                      f" found at sample {index} (timestamp {t}).")

            print(f"\nComplete delay finding function executed in: {dt.datetime.now() - time_before_function}")

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
            raise Exception(f"Wrong value for the parameter return_delay_format: {return_delay_format}. The value " +
                            f"should be either \"index\", \"ms\", \"s\" or \"timedelta\".")

        # Plot and/or save the figure
        if plot_figure is not None or path_figures is not None:
            _create_figure(array, excerpt, freq_array, freq_excerpt, "Array", "Excerpt " + str(i + 1), envelope_array,
                           envelope_excerpt, y1, y2, compute_envelope, window_size_env, overlap_ratio_env,
                           filter_below, filter_over, resampling_rate, window_size_res, overlap_ratio_res,
                           cross_correlation, threshold, number_of_plots, return_delay_format, return_value,
                           max_correlation_value, index_max_correlation_value, plot_figure, path_figures, name_figures
                           + "_" + str(i + 1) + ".png", plot_intermediate_steps, x_format_figure, verbosity)

        if max_correlation_value >= threshold:

            if return_correlation_values:
                delays.append(return_value)
                correlation_values.append(max_correlation_value)
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
