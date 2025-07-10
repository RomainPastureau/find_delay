from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
from scipy.signal import butter, correlate, hilbert, lfilter
import numpy as np
import datetime as dt
import os

# noinspection PyTupleAssignmentBalance
def _filter_frequencies(array, frequency, filter_below=None, filter_over=None, verbosity=1, add_tabs=0):
    """Applies a low-pass, high-pass or band-pass filter to the data in the attribute :attr:`samples`.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Turned into a "private" function by adding a leading underscore.

    Parameters
    ----------
    array: list | :class:`numpy.ndarray`
        An array of samples.

    frequency: int | float
        The sampling frequency of the array, in Hz.

    filter_below: float | None (optional)
        The value below which you want to filter the data. If set on None or 0, this parameter will be ignored.
        If this parameter is the only one provided, a high-pass filter will be applied to the samples; if
        ``filter_over`` is also provided, a band-pass filter will be applied to the samples.

    filter_over: float | None (optional)
        The value over which you want to filter the data. If set on None or 0, this parameter will be ignored.
        If this parameter is the only one provided, a low-pass filter will be applied to the samples; if
        ``filter_below`` is also provided, a band-pass filter will be applied to the samples.

    verbosity: int (optional)
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    add_tabs: int (optional)
        Adds the specified amount of tabulations to the verbosity outputs (default: 0). This parameter may be used by
        other functions to encapsulate the verbosity outputs by indenting them.

        .. versionadded:: 2.12

    Returns
    -------
    :class:`numpy.ndarray`
        The array with filtered values.
    """
    t = add_tabs * "\t"

    # Band-pass filter
    if filter_below not in [None, 0] and filter_over not in [None, 0]:
        if verbosity > 0:
            print(f"{t}\tApplying a band-pass filter for frequencies between {filter_below} and {filter_over} Hz.")
        b, a = butter(2, [filter_below, filter_over], "band", fs=frequency)
        filtered_array = lfilter(b, a, array)

    # High-pass filter
    elif filter_below not in [None, 0]:
        if verbosity > 0:
            print(f"{t}\tApplying a high-pass filter for frequencies over {filter_below} Hz.")
        b, a = butter(2, filter_below, "high", fs=frequency)
        filtered_array = lfilter(b, a, array)

    # Low-pass filter
    elif filter_over not in [None, 0]:
        if verbosity > 0:
            print(f"{t}\tApplying a low-pass filter for frequencies below {filter_over} Hz.")
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
    array_length_or_array: :class:`numpy.ndarray` | list | int
        An array of numerical values, or its length.

    window_size: int
        The number of array elements in each window.

    overlap_ratio: float (optional)
        The ratio, between 0 (inclusive, default) and 1 (exclusive), of array elements overlapping between each window
        and the next.

    add_incomplete_window: bool (optional)
        If set on ``True`` (default), the last window will be included even if its size is smaller than ``window_size``.
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
                  verbosity=1, add_tabs=0):
    """Calculates the envelope of an array, and returns it. The function can also optionally perform a band-pass
    filtering, if the corresponding parameters are provided. To calculate the envelope, the function calculates the
    absolute values of the
    `scipy.signal.hilbert <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html>`_ transform
    of each window of the envelope.

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
    array: :class:`numpy.ndarray` | list
        An array of samples.

    frequency: int | float
        The sampling frequency of the array, in Hz.

    window_size: int | None (optional)
        The size of the windows (in samples) in which to cut the array to calculate the envelope. Cutting large arrays
        into windows allows speeding up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 1 million (default). If this parameter is
        set on 0, on `None` or on a number of samples bigger than the number of elements in the array, the window size
        is set on the length of the samples.

        .. versionadded:: 2.1

    overlap_ratio: float | None (optional)
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for a number of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows discarding any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap. Default value: 0.5 (each window will overlap at 50% with the previous and
        the next).

    filter_below: int | float | None (optional)
        If not `None` (default) nor 0, this value will be provided as the lowest frequency of the band-pass filter.

    filter_over: int | float | None (optional)
        If not `None` (default) nor 0, this value will be provided as the highest frequency of the band-pass filter.

    verbosity: int (optional)
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    add_tabs: int (optional)
        Adds the specified amount of tabulations to the verbosity outputs (default: 0). This parameter may be used by
        other functions to encapsulate the verbosity outputs by indenting them.

        .. versionadded:: 2.12

    Returns
    -------
    :class:`numpy.ndarray`
        The envelope of the original array.
    """

    time_before = dt.datetime.now()
    t = add_tabs * "\t"

    # Settings
    if window_size == 0 or window_size is None or window_size > len(array):
        window_size = len(array)

    if overlap_ratio is None:
        overlap_ratio = 0

    window_size = int(window_size)
    overlap = int(np.ceil(overlap_ratio * window_size))
    number_of_windows = _get_number_of_windows(len(array), window_size, overlap_ratio, True)

    # Hilbert transform
    if verbosity == 1:
        print(f"{t}\tGetting the Hilbert transform...", end=" ")
    elif verbosity > 1:
        print(f"{t}\tGetting the Hilbert transform...")
        print(f"{t}\t\tDividing the samples in {number_of_windows} window(s) of {window_size} samples, with an " +
              f"overlap of {overlap} samples.")

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
            print(f"{t}\t\t\tGetting samples from window {i + 1}/{number_of_windows}: samples {array_start} " +
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
            print(f"\n{t}\t\t\tKeeping the samples from {slice_start} to {slice_end} in the window: samples " +
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
    # elif verbosity > 1:
    #     print("Done.")

    # Filtering
    if filter_below is not None or filter_over is not None:
        envelope = _filter_frequencies(envelope, frequency, filter_below, filter_over, verbosity, add_tabs)

    if verbosity > 0:
        print(f"{t}\tEnvelope calculated in: {dt.datetime.now() - time_before}")

    return envelope


def _resample_window(array, original_timestamps, resampled_timestamps, index_start_original, index_end_original,
                     index_start_resampled, index_end_resampled, method="cubic", verbosity=1, add_tabs=0):
    """Performs and returns the resampling on a subarray of samples.

    .. versionadded:: 1.0

    .. versionchanged:: 1.1
        Turned into a "private" function by adding a leading underscore.

    Parameters
    ----------
    array: :class:`numpy.ndarray` | list
        An array of samples.

    original_timestamps: :class:`numpy.ndarray` | list
        An array containing the timestamps for each sample of the original array.

    resampled_timestamps: :class:`numpy.ndarray` | list
        An array containing the timestamps for each desired sample in the resampled array.

    index_start_original: int
        The index in the array where the window starts.

    index_end_original: int
        The index in the array where the window ends.

    index_start_resampled: int
        The index in the resampled array where the window starts.

    index_end_resampled: int
        The index in the resampled array where the window ends.

    method: str (optional)
        This parameter allows for various values:

        • ``"linear"`` performs a linear
          `numpy.interp <https://numpy.org/devdocs/reference/generated/numpy.interp.html>`_ interpolation. This method,
          though simple, may not be very precise for upsampling naturalistic stimuli.
        • ``"cubic"`` (default) performs a cubic interpolation via `scipy.interpolate.CubicSpline
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
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_. The XXX part of the
          parameter can be replaced by ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``, "slinear"``,
          ``"quadratic"``, ``"cubic"``, ``"previous"``, and ``"next"`` (see the documentation of this function for
          specifics).

    verbosity: int (optional)
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    add_tabs: int (optional)
        Adds the specified amount of tabulations to the verbosity outputs (default: 0). This parameter may be used by
        other functions to encapsulate the verbosity outputs by indenting them.

        .. versionadded:: 2.12

    Returns
    -------
    :class:`numpy.ndarray`
        The envelope of the original array.
    """

    t = add_tabs * "\t"

    array_window = array[index_start_original:index_end_original + 1]
    original_timestamps_window = original_timestamps[index_start_original:index_end_original + 1]
    resampled_timestamps_window = resampled_timestamps[index_start_resampled:index_end_resampled + 1]

    if verbosity > 1:
        print(f"{t}\t\t\t\tIn the original array, the window contains samples {index_start_original} to " +
              f"{index_end_original} (from timestamps {original_timestamps_window[0]} to " +
              f"{original_timestamps_window[-1]}).")
        print(f"{t}\t\t\t\tIn the new array, the window contains samples {index_start_resampled} to "
              f"{index_end_resampled} (from timestamps {resampled_timestamps_window[0]} to "
              f"{resampled_timestamps_window[-1]}).")
        print(f"{t}\t\t\t\tInterpolating the data...", end=" ")

    if np.size(array_window) == 1:
        raise Exception("Only one sample is present in the current window. Please select a larger window size.")

    if method == "linear":
        return np.interp(resampled_timestamps_window, original_timestamps_window, array_window)
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
              method="cubic", verbosity=1, add_tabs=0):
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
    array: :class:`numpy.ndarray` | list
        An array of samples.

    original_frequency: int | float
        The sampling frequency of the array, in Hz.

    resampling_frequency: int | float
        The frequency at which you want to resample the array, in Hz. A frequency of 4 will return samples
        at 0.25 s intervals.

    window_size: int | None (optional)
        The size of the windows (in samples) in which to cut the array before resampling. Cutting large arrays
        into windows allows speeding up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 10 million (default). If this parameter
        is set on 0, on None or on a number of samples bigger than the number of elements in the array, the window size
        is set on the length of the samples.

        .. versionadded:: 2.1

    overlap_ratio: float | None (optional)
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for a number of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows discarding any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap. Default value: 0.5 (each window will overlap at 50% with the previous and
        the next).

    method: str, optional
        This parameter allows for various values:

        • ``"linear"`` performs a linear
          `numpy.interp <https://numpy.org/devdocs/reference/generated/numpy.interp.html>`_ interpolation. This method,
          though simple, may not be very precise for upsampling naturalistic stimuli.
        • ``"cubic"`` (default) performs a cubic interpolation via `scipy.interpolate.CubicSpline
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
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_. The XXX part of the
          parameter can be replaced by ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``, "slinear"``,
          ``"quadratic"``, ``"cubic"``, ``"previous"``, and ``"next"`` (see the documentation of this function for
          specifics).

    verbosity: int (optional)
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    add_tabs: int, optional
        Adds the specified amount of tabulations to the verbosity outputs (default: 0). This parameter may be used by
        other functions to encapsulate the verbosity outputs by indenting them.

        .. versionadded:: 2.12

    Returns
    -------
    :class:`numpy.ndarray`
        The resampled array.

    Warning
    -------
    This function allows both the **upsampling** and the **downsampling** of arrays. However, during any of
    these operations, the algorithm only **estimates** the real values of the samples. You should then consider
    the upsampling (and the downsampling, to a lesser extent) with care.
    """

    time_before = dt.datetime.now()
    t = add_tabs * "\t"

    if resampling_frequency == original_frequency:
        if verbosity > 0:
            print(f"{t}\tNot performing the resampling as the resampling frequency is the same as the original " +
                  f"frequency ({resampling_frequency} Hz).")
        return array

    if verbosity > 0:
        print(f"{t}\tResampling the array at {resampling_frequency} Hz (mode: {method})...")
        print(f"{t}\t\tOriginal frequency: {round(original_frequency, 2)} Hz.")
        if verbosity > 1:
            print(f"{t}\t\tPerforming the resampling...")
        else:
            print(f"{t}\t\tPerforming the resampling...", end=" ")

    if method == "take":
        if resampling_frequency > original_frequency:
            raise Exception(f"The mode \"take\" does not allow for upsampling of the data. Please input a resampling " +
                            f"frequency inferior to the original ({original_frequency}).")
        factor_resampling = original_frequency / resampling_frequency
        print(factor_resampling)
        if factor_resampling != int(factor_resampling):
            print(f"{t}Warning: The downsampling factor is not an integer ({factor_resampling}), meaning that the " +
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
        if window_size == 0 or window_size is None or window_size > len(array):
            window_size = len(array)

        if overlap_ratio is None:
            overlap_ratio = 0

        window_size = int(window_size)
        overlap = int(np.ceil(overlap_ratio * window_size))
        number_of_windows = _get_number_of_windows(len(array), window_size, overlap_ratio, True)

        if verbosity > 1 and number_of_windows != 1:
            print(f"{t}\t\t\tCreating {number_of_windows} window(s), each containing {window_size} samples, with a " +
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
                print(f"{t}\t\t\tGetting samples from window {i + 1}/{number_of_windows}.")

            resampled_window = _resample_window(array, original_timestamps, resampled_timestamps, window_start_original,
                                                window_end_original, window_start_resampled, window_end_resampled,
                                                method, verbosity, add_tabs)

            if verbosity > 1:
                print(f"Done.\n{t}\t\t\t\tThe resampled window contains {np.size(resampled_window)} sample(s).")

            # Keep only the center values
            if i == 0:
                window_slice_start = 0
                resampled_slice_start = 0
            else:
                window_slice_start = (j - window_start_resampled) // 2
                resampled_slice_start = window_start_resampled + window_slice_start

            preserved_samples = resampled_window[window_slice_start:]

            if verbosity > 1:
                print(f"{t}\t\t\tKeeping the samples from {resampled_slice_start} to " +
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


def _convert_to_mono(array, mono_channel=0, verbosity=1, add_tabs=0):
    """Converts an array to mono.

    .. versionadded:: 2.9

    .. versionchanged:: 2.12
        Added the parameter ``add_tabs``.

    .. versionchanged:: 2.19
        The function now returns an error if the dimension of the array is more than 2.

    Parameters
    ----------
    array: :class:`numpy.ndarray` (1D or 2D)
        Any array, or the parameter data resulting from reading a WAV file with
        `scipy.io.wavfile.read <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html>`_.

    mono_channel: int | str (optional)
        Defines the method to use to convert multiple-channel WAV files to mono, if one of the parameters `array1` or
        `array2` is a path pointing to a WAV file. By default, this parameter value is ``0``: the channel with index 0
        in the WAV file is used as the array, while all the other channels are discarded. This value can be any
        of the channel indices (using ``1`` will preserve the channel with index 1, etc.). This parameter can also
        take the value ``"average"``: in that case, a new channel is created by averaging the values of all the
        channels of the WAV file. Note that this parameter applies to both arrays: in the case where you need to select
        different channels for each WAV file, open the files before calling the function and pass the samples and
        frequencies as parameters.

    verbosity: int (optional)
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    add_tabs: int (optional)
        Adds the specified amount of tabulations to the verbosity outputs (default: 0). This parameter may be used by
        other functions to encapsulate the verbosity outputs by indenting them.

        .. versionadded:: 2.12

    Returns
    -------
    :class:`numpy.ndarray`
        A 1D numpy array containing the audio converted to mono.
    """

    t = add_tabs * "\t"

    print(array)
    print(array.ndim)

    if array.ndim == 1:
        mono_array = array
        if verbosity > 0:
            print(f"{t}\tThe audio data is already in mono, no conversion necessary.")

    elif array.ndim > 2:
        raise Exception(f"The dimension of the array is {array.ndim}, but it should be 1 or 2.")

    # If mono_channel is a channel number
    elif isinstance(mono_channel, int):

        # If the channel number is negative or nonexistent
        if mono_channel >= array.shape[1] or mono_channel < 0:
            if np.size(array[1] == 0):
                raise Exception(f"""The channel chosen for the parameter "mono_channel" ({mono_channel}) is not valid.
                As the audio data is mono, the channel chosen should be 0.""")
            else:
                raise Exception(f"""The channel chosen for the parameter "mono_channel" ({mono_channel}) is not valid.
                Please choose a channel between 0 and {np.size(array[1]) - 1}.""")

        # If the audio data is not mono
        else:
            mono_array = array[:, mono_channel]  # Turn to mono
            if verbosity > 0:
                print(f"{t}\tFile converted to mono by keeping channel with index {mono_channel}. The original file"
                      f" contains {np.size(array[1])} channels.")

    # If mono_channel is "average"
    elif mono_channel == "average":
        mono_array = np.mean(array, 1)

    # Any other case
    else:
        raise Exception(f"""{t}The parameter "mono_channel" should be an integer (the channel index) or "average", not
                        {mono_channel}.""")

    return mono_array


def _cross_correlation(y1, y2, rate, freq_y1_original, threshold, return_delay_format, min_delay, max_delay,
                       verbosity=1, add_tabs=0):
    """Performs a normalized cross-correlation between two arrays.

    .. versionadded:: 2.12

    .. versionchanged:: 2.17
        Added the parameters `min_delay` and `max_delay`, allowing to limit the search to a specific range of delays.

    .. versionchanged:: 2.18
        The parameter `return_delay_format` can now also take the value ``"sample"``, which is an alias for ``"index"``.

    y1: :class:`numpy.ndarray`
        The first array to cross-correlate.

    y2: :class:`numpy.ndarray`
        The second array to cross-correlate.

    rate: int | float
        The frequency rate of the two arrays.

    freq_y1_original: int | float
        The original frequency rate of y1, before resampling.

    threshold: float
        The threshold of the minimum correlation value between the two arrays to accept a delay as a solution. If
        multiple delays are over threshold, the delay with the maximum correlation value will be returned. This value
        should be between 0 and 1; if the maximum found value is below the threshold, the function will return `None`
        instead of a timestamp.

    return_delay_format: str (optional)
        This parameter can be either ``"index"``, ``"ms"``, ``"s"``, or ``"timedelta"``:

            • If ``"index"`` (default) or ``"sample"``, the function will return the index in array_1 at which array_2
              has the highest cross-correlation value.
            • If ``"ms"``, the function will return the timestamp in array_1, in milliseconds, at which array_2 has the
              highest cross-correlation value.
            • If ``"s"``, the function will return the timestamp in array_1, in seconds, at which array_2 has the
              highest cross-correlation value.
            • If ``"timedelta"``, the function will return the timestamp in array_1 at which array_2 has the
              highest cross-correlation value as a
              `datetime.timedelta <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_ object.
              Note that, in the case where the result is negative, the timedelta format may give unexpected display
              results (-1 second returns -1 days, 86399 seconds).

    min_delay: int | float | None (optional)
        The lower limit of the sample or time range in which to look for the highest correlation value. This parameter
        must be specified in the same unit as ``return_delay_format``.

    max_delay: int | float | None (optional)
        The upper limit of the sample or time range in which to look for the highest correlation value. This parameter
        must be specified in the same unit as ``return_delay_format``.

    verbosity: int (optional)
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    add_tabs: int (optional)
        Adds the specified amount of tabulations to the verbosity outputs (default: 0). This parameter may be used by
        other functions to encapsulate the verbosity outputs by indenting them.

        .. versionadded:: 2.12

    Returns
    -------
    :class:`numpy.ndarray`
        The normalized cross-correlation array.
    int | float | timedelta | None
        The sample index, timestamp or timedelta of y1 at which y2 can be found (defined by the parameter
        return_delay_format), or `None` if y1 is not contained in y2.
    float
        The max correlation value from the normalized cross-correlation array.
    int
        The index of this max correlation value in the normalized cross-correlation array.
    """
    time_before_cross_correlation = dt.datetime.now()
    t = add_tabs * "\t"

    if verbosity > 0:
        print(f"{t}Computing the correlation...", end=" ")

    y2_normalized = (y2 - y2.mean()) / y2.std() / np.sqrt(y2.size)
    y1_m = correlate(y1, np.ones(y2.size), "full") ** 2 / y2_normalized.size
    y1_m2 = correlate(y1 ** 2, np.ones(y2.size), "full")

    diff = (y1_m2 - y1_m)
    diff_clipped = diff.clip(min=0)

    cross_corr = correlate(y1, y2_normalized, "full")
    cross_corr_norm = np.divide(cross_corr, np.sqrt(diff_clipped), out=np.zeros_like(cross_corr), where=diff_clipped != 0)

    max_corr_value = np.nanmax(cross_corr_norm)
    index_max_corr_value = np.nanargmax(cross_corr_norm) - y2.size + 1

    # Get the timestamps of the cross-correlation array in the right format
    t_cross_corr = np.arange(0, len(cross_corr_norm)) - y2.size + 1
    if return_delay_format not in ["index", "sample"]:
        t_cross_corr = t_cross_corr / rate
        if return_delay_format == "timedelta":
            t_cross_corr = np.array(t_cross_corr * 1000000, dtype="datetime64[us]")
        if return_delay_format == "ms":
            t_cross_corr = t_cross_corr * 1000

    # Get the max correlation in the specified range
    cross_corr_norm_segment = None
    t_cross_corr_min_idx = None
    is_segment = False
    if min_delay is not None or max_delay is not None:
        is_segment = True

        # Define min/max delay if one is not user-set
        if min_delay is None:
            min_delay = t_cross_corr[0]
        if max_delay is None:
            max_delay = t_cross_corr[-1]

        # Get the closest indices to min_delay and max_delay
        t_cross_corr_min_idx = np.argmin(np.abs(t_cross_corr - min_delay))
        t_cross_corr_max_idx = np.argmin(np.abs(t_cross_corr - max_delay))

        # Get the highest correlation
        cross_corr_norm_segment = cross_corr_norm[t_cross_corr_min_idx:t_cross_corr_max_idx + 1]
        max_corr_value = np.nanmax(cross_corr_norm_segment)
        index_max_corr_value_segment = np.nanargmax(cross_corr_norm_segment)
        index_max_corr_value = t_cross_corr_min_idx + index_max_corr_value_segment  - y2.size + 1

    index = int(np.round(index_max_corr_value * (freq_y1_original / rate), 0))
    delay_in_seconds = index_max_corr_value / rate
    if delay_in_seconds >= 0:
        sign = ""
        time = dt.timedelta(days=delay_in_seconds // 86400, seconds=int(delay_in_seconds % 86400),
                         microseconds=(delay_in_seconds % 1) * 1000000)
    else:
        sign = "-"
        time = dt.timedelta(days=-delay_in_seconds // 86400, seconds=int(-delay_in_seconds % 86400),
                         microseconds=(-delay_in_seconds % 1) * 1000000)

    if verbosity > 0:
        print("Done.")
        print(f"{t}\tCross-correlation calculated in: {dt.datetime.now() - time_before_cross_correlation}")

        if max_corr_value >= threshold:
            if is_segment:
                print(f"{t}\tMaximum correlation ({np.round(max_corr_value, 3)}) in segment [{min_delay}; "
                      f"{max_delay}] found at sample {index} (timestamp {sign}{time}).")
            else:
                print(f"{t}\tMaximum correlation ({np.round(max_corr_value, 3)}) found at sample {index} " +
                      f"(timestamp {sign}{time}).")

        else:
            if is_segment:
                print(f"{t}\tNo correlation over threshold found in the segment [{min_delay}; {max_delay}] (max "
                      f"correlation: {np.round(max_corr_value, 3)}) found at sample {index} (timestamp "
                      f"{sign}{time}).")
            else:
                print(f"{t}\tNo correlation over threshold found (max correlation: "
                      f"{np.round(max_corr_value, 3)} found at sample {index}, timestamp {sign}{time}).")

    # Return values: None if below threshold
    if return_delay_format in ["index", "sample"]:
        return_value = index
    elif return_delay_format == "ms":
        return_value = delay_in_seconds * 1000
    elif return_delay_format == "s":
        return_value = delay_in_seconds
    elif return_delay_format == "timedelta":
        return_value = time
    else:
        raise Exception(f"Wrong value for the parameter return_delay_format: {return_delay_format}. The value should " +
                        f"be either \"index\", \"sample\", \"ms\", \"s\" or \"timedelta\".")

    return (cross_corr_norm, cross_corr_norm_segment, return_value, max_corr_value, index_max_corr_value,
            t_cross_corr, t_cross_corr_min_idx)


def _create_figure(array_1, array_2, freq_array_1, freq_array_2, name_array_1, name_array_2, envelope_1, envelope_2,
                   y1, y2, compute_envelope, window_size_env, overlap_ratio_env, filter_below, filter_over,
                   resampling_rate, window_size_res, overlap_ratio_res, cross_correlation, cross_correlation_segment,
                   cross_correlation_start, threshold, number_of_plots, return_delay_format, return_value,
                   max_correlation_value, index_max_correlation_value, plot_figure, path_figure,
                   name_figure, plot_intermediate_steps, x_format_figure, dark_mode, verbosity, add_tabs):
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
        Added a second y-axis to the aligned arrays graph.

    .. versionchanged:: 2.3
        Corrected the figure saving to a file.
        If `plot_intermediate_steps` is `False`, the two graphs "cross-correlation" and "aligned arrays" are now on top
        of each other instead of side-by-side.

    .. versionchanged:: 2.4
        Added the new parameter `x_format_figure`, allowing to have HH:MM:SS times on the x-axis.
        Modified the scaling of the aligned arrays figure to be more accurate.

    .. versionchanged:: 2.17
        For negative values, invalid timedelta values wer displayed on the horizontal axes of the figures. This has been
        corrected. In order to gain space, hours on the horizontal axis are now shortened if the hour is equal to 0.
        Timestamps in the figure now appear rounded down to the sixth decimal, if ``return_delay_format`` is set on
        ``"ms"`` or ``"s"``.

    Parameters
    ----------
    array_1: :class:`numpy.ndarray`
        The first array involved in the cross-correlation.

    array_2: :class:`numpy.ndarray`
        The second array involved in the cross-correlation, being allegedly an excerpt from the first.

    freq_array_1: int | float
        The sampling rate of `array_1`.

    freq_array_2: int | float
        The sampling rate of `array_2`.

    name_array_1: str
        The name of the first array; will be "Array 1" for `find_delay` and "Array" for `find_delays`.

    name_array_2: str
        The name of the second array; will be "Array 2" for `find_delay` and "Excerpt n" for `find_delays`, with n
        being the index of the excerpt in the folder.

    envelope_1: :class:`numpy.ndarray`
        The envelope of `array_1` (if calculated).

    envelope_2: :class:`numpy.ndarray`
        The envelope of `array_2` (if calculated).

    y1: :class:`numpy.ndarray`
        The resampled `array_1` or `envelope_1` (if calculated).

    y2: :class:`numpy.ndarray`
        The resampled `array_2` or `envelope_2` (if calculated).

    compute_envelope: bool
        A boolean describing if the envelope has been computed or not.

    window_size_env: int
        The size of the windows in which the arrays were cut for the envelope calculation.

    overlap_ratio_env: float
        The ratio of overlapping between each envelope window with the previous and the next windows.

    filter_below: int | float | None
        The lower limit of the bandpass filter applied to the envelopes.

    filter_over: int | float | None
        The upper limit of the bandpass filter applied to the envelopes.

    resampling_rate: int | float | None
        The rate at which the arrays or the envelopes have been resampled.

    window_size_res: int
        The size of the windows in which the arrays or envelopes were cut for the resampling.

    overlap_ratio_res: float
        The ratio of overlapping between each resampling window with the previous and the next windows.

    cross-correlation: :class:`numpy.ndarray`
        The array containing the correlation values for each lag between the two arrays.

    threshold: float
        The threshold of the maximum correlation value between the two arrays, relative to the maximum
        correlation value between the excerpt and itself.

    return_delay_format: str
        Indicates the format of the displayed delay, either ``"index"``, ``"sample"``, ``"ms"``, ``"s"``,
        or ``"timedelta"``.

    return_value: int|float|timedelta
        The value of the delay in the format specified by the previous parameter.

    max_correlation_value: float
        The maximum correlation value from the cross-correlation.

    index_max_correlation_value: int
        The index at which the maximum correlation value can be found in the cross-correlation array.

    plot_figure: bool
        If set on `True`, plots the figure in a Matplotlib window.

    path_figure: str | None
        If set, saves the figure at the given path.

    name_figure: str | None
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

    dark_mode: bool
        If set on `True`, uses the `dark_background theme from matplotlib <https://matplotlib.org/stable/gallery/style_sheets/dark_background.html>`_
        (default: `False`).

    verbosity: int
        Sets how much feedback the code will provide in the console output:

        • *0: Silent mode.* The code won’t provide any feedback, apart from error messages.
        • *1: Normal mode* (default). The code will provide essential feedback such as progression markers and
          current steps.
        • *2: Chatty mode.* The code will provide all possible information on the events happening. Note that this
          may clutter the output and slow down the execution.

    add_tabs: int
        Adds the specified amount of tabulations to the verbosity outputs (default: 0). This parameter may be used by
        other functions to encapsulate the verbosity outputs by indenting them.

        .. versionadded:: 2.12
    """

    t = add_tabs * "\t"

    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

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

    # if x_format_figure == "time":
    #     t_array_1 = np.array(t_array_1 * 1000000, dtype="datetime64[us]")
    #     t_array_2 = np.array(t_array_2 * 1000000, dtype="datetime64[us]")
    #     t_res_1 = np.array(t_res_1 * 1000000, dtype="datetime64[us]")
    #     t_res_2 = np.array(t_res_2 * 1000000, dtype="datetime64[us]")
    #     t_res_2_aligned = np.array(t_res_2_aligned * 1000000, dtype="datetime64[us]")
    #     t_cc = np.array(t_cc * 1000000, dtype="datetime64[us]")

    # Formatting functions for the x-axis (MM:SS and HH:MM:SS)
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
        #td_value = mdates.num2timedelta(value)
        #seconds = td_value.total_seconds()

        seconds = value
        hh = str(int(seconds // 3600)).zfill(2)
        mm = str(int((seconds // 60) % 60)).zfill(2)
        ss = str(int(seconds % 60)).zfill(2)
        us = str(int((seconds % 1) * 1000000)).rstrip("0")

        label = ""
        if include_hour:
            label += hh
        label += ":" + mm + ":" + ss
        if include_us and us != "":
            label += "." + us

        if include_us and hh == "00":
            label = label[2:]

        if neg:
            label = "-" + label

        return label

    def get_label_hh_mm_ss_no_ms(value, pos=None):
        """Returns a label value as HH:MM:SS, without any ms value."""
        return get_label(value, True, False)

    def get_label_hh_mm_ss(value, pos=None):
        """Returns a label value as HH:MM:SS.ms, without any trailing zero."""
        return get_label(value, True, True)

    def set_label_time_figure(ax):
        """Sets the time-formatted labels on the x axes."""
        if x_format_figure == "time":
            formatter = mdates.AutoDateFormatter(ax.xaxis.get_major_locator())
            formatter.scaled[1 / mdates.MUSECONDS_PER_DAY] = get_label_hh_mm_ss
            formatter.scaled[1 / mdates.SEC_PER_DAY] = get_label_hh_mm_ss
            formatter.scaled[1 / mdates.MINUTES_PER_DAY] = get_label_hh_mm_ss
            formatter.scaled[1 / mdates.HOURS_PER_DAY] = get_label_hh_mm_ss
            formatter.scaled[1] = get_label_hh_mm_ss
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

    if cross_correlation_segment is not None:
        color_cc = "#aaaaaa"
    else:
        color_cc = "green"

    if plot_intermediate_steps:
        ax[i // 2][i % 2].set_title(title)
        ax[i // 2][i % 2].set_ylim(np.nanmin(cross_correlation), 1.5)
        ax[i // 2][i % 2].plot(t_cc, cross_correlation, color=color_cc)
        ax[i // 2][i % 2] = set_label_time_figure(ax[i // 2][i % 2])
        if cross_correlation_segment is not None:
            ax[i // 2][i % 2].plot(t_cc[cross_correlation_start:cross_correlation_start + len(
                cross_correlation_segment)], cross_correlation_segment, color="green")
    else:
        ax[0].set_title(title)
        ax[0].set_ylim(np.nanmin(cross_correlation), 1.5)
        ax[0].plot(t_cc, cross_correlation, color=color_cc)
        ax[0] = set_label_time_figure(ax[0])
        if cross_correlation_segment is not None:
            ax[0].plot(t_cc[cross_correlation_start:cross_correlation_start + len(
                cross_correlation_segment)], cross_correlation_segment, color="green")

    text = ""
    if return_delay_format in ["index", "sample"]:
        text = "Sample "

    if return_delay_format in ["ms", "s"]:
        text += str(np.round(return_value, 6))
        text += return_delay_format
    else:
        text += str(return_value)

    if max_correlation_value >= threshold:
        text += " · Correlation value: " + str(np.round(max_correlation_value, 3))
        if dark_mode:
            bbox_props = dict(boxstyle="square,pad=0.3", fc="green", ec="k", lw=0.72)
        else:
            bbox_props = dict(boxstyle="square,pad=0.3", fc="#99cc00", ec="k", lw=0.72)
    else:
        text += " · Correlation value (below threshold): " + str(np.round(max_correlation_value, 3))
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

    # Get the min and max values for overlaying the two arrays
    if resampling_rate is None:
        excerpt_in_original = array_1[max(0, index_max_correlation_value):
                                      min(index_max_correlation_value + len(array_2), len(array_1))]
    else:
        index = int(index_max_correlation_value * (freq_array_1 / resampling_rate))
        excerpt_in_original = array_1[max(0, index):
                                      min(index + int(len(array_2) * freq_array_1 / freq_array_2), len(array_1))]
    resampled_timestamps_array2 = t_res_2_aligned[:len(array_2)]

    min_excerpt_in_original = np.nanmin(excerpt_in_original)
    max_excerpt_in_original = np.nanmax(excerpt_in_original)

    index_start_array_2 = 0
    if index_max_correlation_value < 0:
        if resampling_rate is None:
            index_start_array_2 = int(-index_max_correlation_value)
        else:
            index_start_array_2 = int(-index_max_correlation_value * freq_array_2 / resampling_rate)

    index_end_array_2 = len(array_2)
    if resampling_rate is None and index_max_correlation_value + len(array_2) > len(array_1):
        index_end_array_2 = int(len(array_1) - index_max_correlation_value)
    elif resampling_rate is not None:
        if index_max_correlation_value * (freq_array_1 / resampling_rate) + \
            len(array_2) * freq_array_1 / freq_array_2 > len(array_1):
            index_end_array_2 = int(((len(array_1) - index_max_correlation_value * freq_array_1 / resampling_rate) *
                                     freq_array_2 / freq_array_1))

    if min_excerpt_in_original != 0:
        min_ratio = np.nanmin(array_2[index_start_array_2:index_end_array_2]) / min_excerpt_in_original
    else:
        min_ratio = 0

    if max_excerpt_in_original != 0:
        max_ratio = np.nanmax(array_2[index_start_array_2:index_end_array_2]) / max_excerpt_in_original
    else:
        max_ratio = 0

    ratio = np.nanmax([min_ratio, max_ratio])

    ax2.plot(resampled_timestamps_array2, array_2, color="#ffa500aa", linewidth=2)
    ax2.set_ylim((ylim[0] * ratio, ylim[1] * ratio))
    ax2.tick_params(axis='y', labelcolor="#ffa500")
    ax2.set_ylabel(name_array_2, color="#ffa500")

    if path_figure is not None:
        directory, _ = os.path.split(path_figure)
        if len(directory) != 0:
            os.makedirs(directory, exist_ok=True)
        if name_figure is not None:
            if verbosity > 0:
                print(f"\n{t}Saving the graph under {os.path.join(path_figure, name_figure)}...", end=" ")
            plt.savefig(str(path_figure) + "/" + str(name_figure))
        else:
            if verbosity > 0:
                print(f"\n{t}Saving the graph under {path_figure}...", end=" ")
            plt.savefig(str(path_figure))
        if verbosity > 0:
            print("Done.")

    if plot_figure:
        if verbosity > 0:
            print(f"\n{t}Showing the graph...")
        plt.show()

    plt.close()

