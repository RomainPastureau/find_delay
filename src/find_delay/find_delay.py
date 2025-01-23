"""Contains a series of functions directed at calculating the delay between two arrays.
* find_delay finds the delay between two time series using cross-correlation.
* find_delays does the same, but for multiple excerpts from one big time series.

Author: Romain Pastureau, BCBL (Basque Center on Cognition, Brain and Language)
Current version: 2.15 (2025-01-23)
"""
import datetime as dt
import os
import numpy as np
from scipy.io import wavfile

from .private_functions import _convert_to_mono, _get_envelope, _resample, _cross_correlation, _create_figure

def find_delay(array_1, array_2, freq_array_1=1, freq_array_2=1, compute_envelope=True, window_size_env=1e6,
               overlap_ratio_env=0.5, filter_below=None, filter_over=50, resampling_rate="auto", window_size_res=1e7,
               overlap_ratio_res=0.5, resampling_mode="cubic", remove_average_array_1=False,
               remove_average_array_2=False, return_delay_format="index", return_correlation_value=False, threshold=0.9,
               plot_figure=False, plot_intermediate_steps=False, x_format_figure="auto", path_figure=None,
               mono_channel=0, name_array_1="Array 1", name_array_2="Array 2", dark_mode=False, verbosity=1, add_tabs=0):
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
        Added the new parameter `x_format_figure`, allowing to have HH:MM:SS times on the x-axis.

    .. versionchanged:: 2.9
        array_1 and array_2 can now be strings containing paths to WAV files.
        Added the parameter mono_channel.

    .. versionchanged:: 2.13
        Added the parameters name_array_1 and name_array_2, allowing to customize the name of the arrays on the figure.

    .. versionchanged:: 2.15
        Added the parameters `remove_average_array_1` and `remove_average_array_2`, allowing to remove the average for
        the corresponding arrays.
        Added the parameter `dark_mode`.

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
    array_1: numpy.ndarray|list|str
        A first array of samples, or a string containing the path to a WAV file. In this case, the parameter
        `freq_array_1` will be ignored and extracted from the WAV file. Note that if the WAV file contains more than one
        channel, the function will turn the WAV to mono, using the method described by the parameter `mono_channel`.

        .. versionchanged:: 2.9

    array_2: numpy.ndarray|list|str
        An second array of samples, smaller than or of equal size to the first one, that is allegedly an excerpt
        from the first one. The amplitude, frequency or values do not have to match exactly the ones from the first
        array. The parameter can also be a string containing the path to a WAV file (see description of parameter
        array_1).

        .. _numpyarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
        .. versionchanged:: 2.9

    freq_array_1: int|float, optional
        The sampling frequency of the first array, in Hz (default: 1). This parameter is ignored if array_1 is a path to
        a WAV file.

    freq_array_2: int|float, optional
        The sampling frequency of the second array, in Hz (default: 1). This parameter is ignore is array_2 is a path to
        a WAV file.

    compute_envelope: bool, optional
        If `True` (default), calculates the envelope of the array values before performing the cross-correlation.

    window_size_env: int|None, optional
        The size of the windows in which to cut the arrays to calculate the envelope. Cutting long arrays
        in windows allows to speed up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 1 million.

        .. versionadded:: 2.0

    overlap_ratio_env: float|None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap. By default, this parameter is set on 0.5, meaning that each
        window will overlap for half of their values with the previous, and half of their values with the next.

    filter_below: int|None, optional
        If set, a high-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        0 Hz).

    filter_over: int|None, optional
        If set, a low-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        50 Hz).

    resampling_rate: int|float|str|None, optional
        The sampling rate at which to downsample the arrays for the cross-correlation. A larger value will result in
        longer computation times.

        • A recommended value for this parameter when working with audio files is 1000, as it will speed up the
          computation of the cross-correlation while still giving a millisecond-precision delay.
        • Setting the parameter on ``"auto"`` (default) will automatically downsample the array of higher frequency to
          the frequency of the other.
        • Setting the parameter on `None` will not downsample the arrays, which will result in an error if the two
          arrays are not the same frequency. If this parameter is `None`, the next parameters related to resampling can
          be ignored.

        .. versionchanged:: 2.12

    window_size_res: int|None, optional
        The size of the windows in which to cut the arrays. Cutting lo,g arrays in windows allows to speed up the
        computation. If this parameter is set on `None`, the window size will be set on the number of samples. A good
        value for this parameter is generally 1e7.

        .. versionadded:: 2.0

        .. versionchanged:: 2.1
            Decreased default `window_size_res` value from 1e8 to 1e7.

    overlap_ratio_res: float|None, optional
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

    remove_average_array_1: bool, optional
        If set on `True`, removes the average value from all the values in `array_1`. A typical use-case for this
        parameter is if an audio array is not centered around 0. Default: `False`.

        .. versionadded:: 2.15

    remove_average_array_2: bool, optional
        If set on `True`, removes the average value from all the values in `array_2`. A typical use-case for this
        parameter is if an audio array is not centered around 0. Default: `False`.

        .. versionadded:: 2.15

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
              Note that, in the case where the result is negative, the timedelta format may give unexpected display
              results (-1 second returns -1 days, 86399 seconds).

    return_correlation_value: bool, optional
        If `True`, the function returns a second value: the correlation value at the returned delay. This value will
        be None if it is below the specified threshold.

    threshold: float, optional
        The threshold of the minimum correlation value between the two arrays to accept a delay as a solution. If
        multiple delays are over threshold, the delay with the maximum correlation value will be returned. This value
        should be between 0 and 1; if the maximum found value is below the threshold, the function will return `None`
        instead of a timestamp.

    plot_figure: bool, optional
        If set on `True`, plots a graph showing the result of the cross-correlation using Matplotlib. Note that plotting
        the figure causes an interruption of the code execution.

    plot_intermediate_steps: bool, optional
        If set on `True`, plots the original arrays, the envelopes (if calculated) and the resampled arrays (if
        calculated) besides the cross-correlation.

    x_format_figure: str, optional
        If set on `"time"`, the values on the x axes of the output will take the HH:MM:SS format (or MM:SS if the time
        series are less than one hour long). If set on `"float"`, the values on the x axes will be displayed as float
        (unit: second). If set on `"auto"` (default), the format of the values on the x axes will be defined depending
        on the value of `return_delay_format`.

         .. versionadded:: 2.4

    path_figure: str|None, optional
        If set, saves the figure at the given path.

    mono_channel: int|str, optional
        Defines the method to use to convert multiple-channel WAV files to mono, if one of the parameters `array1` or
        `array2` is a path pointing to a WAV file. By default, this parameter value is ``0``: the channel with index 0
        in the WAV file is used as the array, while all the other channels are discarded. This value can be any
        of the channels indices (using ``1`` will preserve the channel with index 1, etc.). This parameter can also
        take the value ``"average"``: in that case, a new channel is created by averaging the values of all the
        channels of the WAV file. Note that this parameter applies to both arrays: in the case where you need to select
        different channels for each WAV file, open the files before calling the function and pass the samples and
        frequencies as parameters.

        .. versionadded:: 2.9

    name_array_1: str, optional
        The name of the first array, as it will appear on the figure (default: "Array 1").

        .. versionadded:: 2.13

    name_array_2: str, optional
        The name of the second array, as it will appear on the figure (default: "Array 2").

        .. versionadded:: 2.13

    dark_mode: bool, optional
        If set on `True`, uses the `dark_background theme from matplotlib <https://matplotlib.org/stable/gallery/style_sheets/dark_background.html>`_
        (default: `False`).

        .. versionadded:: 2.15

    verbosity: int, optional
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
    int|float|timedelta|None
        The sample index, timestamp or timedelta of array_1 at which array_2 can be found (defined by the parameter
        return_delay_format), or `None` if array1 is not contained in array_2.

    float|None, optional
        Optionally, if return_correlation_value is `True`, the correlation value at the corresponding index/timestamp.
    """

    time_before_function = dt.datetime.now()
    t = add_tabs * "\t"

    # Tries to open WAV files if they are paths
    if isinstance(array_1, str):
        if verbosity > 0:
            print(f"{t}Loading WAV file to array_1 from the path {array_1}...")
        if not os.path.exists(array_1):
            raise Exception(f"The file passed as parameter for array_1 doesn't exist: {array_1}")

        audio_wav = wavfile.read(array_1)
        freq_array_1 = audio_wav[0]
        array_1 = _convert_to_mono(audio_wav[1], mono_channel, verbosity)

    if isinstance(array_2, str):
        if verbosity > 0:
            print(f"{t}Loading WAV file to array_2 from the path {array_2}...")
        if not os.path.exists(array_2):
            raise Exception(f"The file passed as parameter for array_2 doesn't exist: {array_2}")

        audio_wav = wavfile.read(array_2)
        freq_array_2 = audio_wav[0]
        array_2 = _convert_to_mono(audio_wav[1], mono_channel, verbosity, add_tabs)

    # Turn lists into ndarray
    if isinstance(array_1, list):
        array_1 = np.array(array_1)
    if isinstance(array_2, list):
        array_2 = np.array(array_2)

    # Introduction
    if verbosity > 0:
        print(f"{t}Trying to find when {name_array_2} starts in {name_array_1}.")
        print(f"{t}\tThe first array {name_array_1} contains {np.size(array_1)} samples, at a rate of "
              f"{freq_array_1} Hz.")
        print(f"{t}\tThe second array {name_array_2} contains {np.size(array_2)} samples, at a rate of "
              f"{freq_array_2} Hz.\n")

    # Remove the average if needed
    if remove_average_array_1:
        mean = np.mean(array_1)
        array_1 = array_1 - mean
        if verbosity > 1:
            print(f"Removing the average {mean} off {name_array_1}.")
    if remove_average_array_2:
        mean = np.mean(array_2)
        array_2 = array_2 - mean
        if verbosity > 1:
            print(f"Removing the average {mean} off {name_array_2}.")

    number_of_plots = 2
    if plot_intermediate_steps:
        number_of_plots += 2

    if len(array_1.shape) > 1:
        raise Exception(f"{name_array_1} has more than one dimension: its shape is {array_1.shape}. To perform the " +
                        f"find_delay function, each array must be 1-dimensional. If you are trying to find the " +
                        f"delay between two audio files, make sure that your files are in mono, or select one of " +
                        f"the channels.")
    if len(array_2.shape) > 1:
        raise Exception(f"{name_array_2} has more than one dimension: its shape is {array_2.shape}. To perform the " +
                        f"find_delay function, each array must be 1-dimensional. If you are trying to find the " +
                        f"delay between two audio files, make sure that your files are in mono, or select one of " +
                        f"the channels.")

    # Envelope
    if compute_envelope:

        if plot_intermediate_steps:
            number_of_plots += 2

        if verbosity > 0:
            print(f"{t}Getting the envelope from {name_array_1}...")

        envelope_1 = _get_envelope(array_1, freq_array_1, window_size_env, overlap_ratio_env, filter_below, filter_over,
                                   verbosity, add_tabs)
        if verbosity > 0:
            print(f"{t}Getting the envelope from {name_array_2}...")
        envelope_2 = _get_envelope(array_2, freq_array_2, window_size_env, overlap_ratio_env, filter_below, filter_over,
                                   verbosity, add_tabs)
        if verbosity > 0:
            print(f"{t}Envelopes calculated.\n")
    else:
        envelope_1 = array_1
        envelope_2 = array_2

    # Resampling
    if resampling_rate is not None:

        if resampling_rate == "auto":
            freq_arrays = np.array([freq_array_1, freq_array_2])
            resampling_rate = np.min(freq_arrays)

        if plot_intermediate_steps:
            number_of_plots += 2

        rate = resampling_rate
        if verbosity > 0:
            print(f"{t}Resampling {name_array_1}...")
        y1 = _resample(envelope_1, freq_array_1, resampling_rate, window_size_res, overlap_ratio_res, resampling_mode,
                       verbosity, add_tabs)
        if verbosity > 0:
            print(f"{t}Resampling {name_array_2}...")
        y2 = _resample(envelope_2, freq_array_2, resampling_rate, window_size_res, overlap_ratio_res, resampling_mode,
                       verbosity, add_tabs)
        if verbosity > 0:
            print(f"{t}Resampling done.\n")
    else:
        rate = freq_array_1
        if freq_array_1 != freq_array_2:
            raise Exception(f"The rate of the two arrays you are trying to correlate are different ({freq_array_1} Hz" +
                            f" and {freq_array_2} Hz). You must indicate a resampling rate to perform the " +
                            f"cross-correlation.")
        y1 = envelope_1
        y2 = envelope_2

    values = _cross_correlation(y1, y2, rate, freq_array_1, threshold, return_delay_format, verbosity, add_tabs)
    cross_correlation_normalized = values[0]
    return_value = values[1]
    max_correlation_value = values[2]
    index_max_correlation_value = values[3]

    if verbosity > 0:
        print(f"\n{t}Complete delay finding function executed in: {dt.datetime.now() - time_before_function}")

    # Plot and/or save the figure
    if plot_figure is not None or path_figure is not None:
        _create_figure(array_1, array_2, freq_array_1, freq_array_2, name_array_1, name_array_2, envelope_1, envelope_2,
                       y1, y2, compute_envelope, window_size_env, overlap_ratio_env, filter_below, filter_over,
                       resampling_rate, window_size_res, overlap_ratio_res, cross_correlation_normalized, threshold,
                       number_of_plots, return_delay_format, return_value, max_correlation_value,
                       index_max_correlation_value, plot_figure, path_figure, None, plot_intermediate_steps,
                       x_format_figure, dark_mode, verbosity, add_tabs)

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
                resampling_rate="auto", window_size_res=1e7, overlap_ratio_res=0.5, resampling_mode="cubic",
                remove_average_array=False, remove_average_excerpts=False, return_delay_format="index",
                return_correlation_values=False, threshold=0.9, plot_figure=False, plot_intermediate_steps=False,
                x_format_figure="auto", path_figures=None, name_figures="figure", mono_channel=0, name_array="Array",
                name_excerpts="Excerpt", dark_mode=False, verbosity=1, add_tabs=0):
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
        Added the new parameter `x_format_figure`, allowing to have HH:MM:SS times on the x-axis.

    .. versionchanged:: 2.9
        array and excerpts can now be strings containing paths to WAV files.
        Added the parameter mono_channel.

    .. versionchanged:: 2.13
        Added the parameters name_array and name_excerpts, allowing to customize the name of the arrays on the figure.

    .. versionchanged:: 2.15
        Added the parameters `remove_average_array` and `remove_average_excerpts`, allowing to remove the average for
        the corresponding arrays.
        Added the parameter `dark_mode`.

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
    array: numpy.ndarray|list|str
        An array of samples, or a string containing the path to a WAV file. In this case, the parameter
        `freq_array` will be ignored and extracted from the WAV file. Note that if the WAV file contains more than one
        channel, the function will turn the WAV to mono, using the method described by the parameter `mono_channel`.

        .. versionchanged:: 2.9

    excerpts: list of numpy.ndarray|list|str
        A list of excerpts, each being an array of samples, a path to a WAV file, or a mix of both. Each excerpt should
        be smaller than or of equal size to the array in which to locate it. The amplitude, frequency or values do not
        have to match exactly the ones from the first array.

        .. versionchanged:: 2.9

    freq_array: int|float, optional
        The sampling frequency of the array, in Hz (default: 1). This parameter can be ignored if the parameter `array`
        is the path to a WAV file.

    freq_excerpts: int|float|list(int|float), optional
        The sampling frequency of the excerpts, in Hz (default: 1). This parameter accepts a single value that will be
        applied for each excerpt, or a list of values that has to be the same length as the number of excerpts, with
        each value corresponding to the frequency of the corresponding excerpt. This parameter can be ignored if all
        the values in the parameter `excerpts` are paths to a WAV file.

    compute_envelope: bool, optional
        If `True` (default), calculates the envelope of the array values before performing the cross-correlation.

    window_size_env: int|None, optional
        The size of the windows in which to cut the arrays to calculate the envelope. Cutting long arrays
        in windows allows to speed up the computation. If this parameter is set on `None`, the window size will be set
        on the number of samples. A good value for this parameter is generally 1 million.

        .. versionadded:: 2.0

    overlap_ratio_env: float|None, optional
        The ratio of samples overlapping between each window. If this parameter is not `None`, each window will
        overlap with the previous (and, logically, the next) for an amount of samples equal to the number of samples in
        a window times the overlap ratio. Then, only the central values of each window will be preserved and
        concatenated; this allows to discard any "edge" effect due to the windowing. If the parameter is set on `None`
        or 0, the windows will not overlap.

    filter_below: int|None, optional
        If set, a high-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        0 Hz).

    filter_over: int|None, optional
        If set, a low-pass filter will be applied on the envelopes before performing the cross-correlation (default:
        50 Hz).

    resampling_rate: int|float|str|None, optional
        The sampling rate at which to downsample the arrays for the cross-correlation. A larger value will result in
        longer computation times.

        • A recommended value for this parameter when working with audio files is 1000, as it will speed up the
          computation of the cross-correlation while still giving a millisecond-precision delay.
        • Setting the parameter on ``"auto"`` (default) will automatically downsample all the arrays to the lowest
          provided frequency, among the frequencies of the array and the excerpts.
        • Setting the parameter on `None` will not downsample the arrays, which will result in an error if the two
          arrays are not the same frequency. If this parameter is `None`, the next parameters related to resampling can
          be ignored.

        .. versionchanged:: 2.12

    window_size_res: int|None, optional
        The size of the windows in which to cut the arrays. Cutting lo,g arrays in windows allows to speed up the
        computation. If this parameter is set on `None`, the window size will be set on the number of samples. A good
        value for this parameter is generally 1e7.

        .. versionadded:: 2.0

        .. versionchanged:: 2.1
            Decreased default `window_size_res` value from 1e8 to 1e7.

    overlap_ratio_res: float|None, optional
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

    remove_average_array: bool, optional
        If set on `True`, removes the average value from all the values in `array`. A typical use-case for this
        parameter is if an audio array is not centered around 0. Default: `False`.

        .. versionadded:: 2.15

    remove_average_excerpts: bool, optional
        If set on `True`, removes the average value from all the values in all the excerpts. A typical use-case for this
        parameter is if an audio array is not centered around 0. Default: `False`.

        .. versionadded:: 2.15

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
              Note that, in the case where the result is negative, the timedelta format may give unexpected display
              results (-1 second returns -1 days, 86399 seconds).

    return_correlation_values: bool, optional
        If `True`, the function returns a second value: the correlation value at the returned delay. This value will
        be None if it is below the specified threshold.

    threshold: float, optional
        The threshold of the minimum correlation value between the two arrays to accept a delay as a solution. If
        multiple delays are over threshold, the delay with the maximum correlation value will be returned. This value
        should be between 0 and 1; if the maximum found value is below the threshold, the function will return `None`
        instead of a timestamp.

    plot_figure: bool, optional
        If set on `True`, plots a graph showing the result of the cross-correlation using Matplotlib. Note that plotting
        the figure causes an interruption of the code execution.

    plot_intermediate_steps: bool, optional
        If set on `True`, plots the original arrays, the envelopes (if calculated) and the resampled arrays (if
        calculated) besides the cross-correlation.

    path_figures: str|None, optional
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

    mono_channel: int|str, optional
        Defines the method to use to convert multiple-channel WAV files to mono, if one of the parameters `array` or
        `excerpts` contains a path pointing to a WAV file. By default, this parameter value is ``0``: the channel with
        index 0 in the WAV file is used as the array, while all the other channels are discarded. This value can be any
        of the channels indices (using ``1`` will preserve the channel with index 1, etc.). This parameter can also
        take the value ``"average"``: in that case, a new channel is created by averaging the values of all the
        channels of the WAV file. Note that this parameter applies to all arrays: in the case where you need to select
        different channels for each WAV file, open the files before calling the function and pass the samples and
        frequencies as parameters.

        .. versionadded:: 2.9

    name_array: str, optional
        The name of the array, as it will appear on the figure (default: "Array").

        .. versionadded:: 2.13

    name_excerpts: list(str)|str, optional
        The name of the excerpts, as it will appear on the figure. If it is a string (default: "Excerpt"), the same
        name will be applied to all the excerpts, followed by the excerpt index (plus one). If it is a list,
        each element of the list should name each excerpt.

        .. versionadded:: 2.13

    dark_mode: bool, optional
        If set on `True`, uses the `dark_background theme from matplotlib <https://matplotlib.org/stable/gallery/style_sheets/dark_background.html>`_
        (default: `False`).

        .. versionadded:: 2.15

    verbosity: int, optional
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
    int|float|timedelta|None
        The sample index, timestamp or timedelta of array1 at which array2 can be found (defined by the parameter
        return_delay_format), or `None` if array1 is not contained in array2.
    float|None, optional
        Optionally, if return_correlation_value is `True`, the correlation value at the corresponding index/timestamp.
    """

    time_before_function = dt.datetime.now()
    t = add_tabs * "\t"

    delays = []
    correlation_values = []

    # Tries to open a WAV file if it is a path
    if isinstance(array, str):
        if verbosity > 0:
            print(f"{t}Loading WAV file to array from the path {array}...")
        if not os.path.exists(array):
            raise Exception(f"The file passed as parameter for {name_array} doesn't exist: {array}")

        audio_wav = wavfile.read(array)
        freq_array = audio_wav[0]
        array = _convert_to_mono(audio_wav[1], mono_channel, verbosity, add_tabs)

    # Turn list into ndarray
    if isinstance(array, list):
        array = np.array(array)

    # name_excerpts size check
    if isinstance(name_excerpts, list):
        if len(name_excerpts) != len(excerpts):
            raise Exception(f"If it is a list, the length of the name_excerpts ({len(name_excerpts)}) must be the " +
                            f"same as the length of the excerpts ({len(excerpts)}.")

    # Introduction
    if verbosity > 0:
        print(f"{t}Trying to find when the excerpts starts in {name_array}.")
        print(f"{t}\tThe main array {name_array} contains {np.size(array)} samples, at a rate of {freq_array} Hz.")
        print(f"{t}\t{len(excerpts)} excerpts to find.")

    # Remove the average if needed
    if remove_average_array:
        mean = np.mean(array)
        array = array - mean
        if verbosity > 1:
            print(f"Removing the average {mean} off the array {name_array}.")

    # Check that the length of the excerpts equals the length of the frequencies
    if isinstance(freq_excerpts, list):
        if len(excerpts) != len(freq_excerpts):
            raise Exception(f"The number of frequencies given for the excerpts ({len(freq_excerpts)}) " +
                            f"is inconsistent with the number of excerpts ({len(excerpts)}).")

    number_of_plots = 2
    if plot_intermediate_steps:
        number_of_plots += 2

    if len(array.shape) > 1:
        raise Exception(f"The provided array has more than one dimension: its shape is {array.shape}. To perform the " +
                        f"find_delay function, each array must be 1-dimensional. If you are trying to find the " +
                        f"delay between two audio files, make sure that your files are in mono, or select one of " +
                        f"the channels.")

    # Envelope
    if compute_envelope:
        if plot_intermediate_steps:
            number_of_plots += 2
        if verbosity > 0:
            print(f"{t}Getting the envelope from the array {name_array}...")
        envelope_array = _get_envelope(array, freq_array, window_size_env, overlap_ratio_env, filter_below, filter_over,
                                       verbosity, add_tabs)
        if verbosity > 0:
            print(f"{t}Envelope calculated.\n")
    else:
        envelope_array = array

    # Resampling
    if resampling_rate is not None:

        if resampling_rate == "auto":
            all_frequencies = np.append(freq_array, freq_excerpts)
            resampling_rate = np.min(all_frequencies)

        if plot_intermediate_steps:
            number_of_plots += 2

        rate = resampling_rate
        if verbosity > 0:
            print(f"{t}Resampling array {name_array}...")
        y1 = _resample(envelope_array, freq_array, resampling_rate, window_size_res, overlap_ratio_res, resampling_mode,
                       verbosity, add_tabs)
        if verbosity > 0:
            print(f"{t}Resampling done.\n")
    else:
        rate = freq_array
        y1 = envelope_array

    for i in range(len(excerpts)):

        if isinstance(name_excerpts, str):
            name_excerpt = f"{name_excerpts} {str(i + 1)}"
        elif isinstance(name_excerpts, list):
            name_excerpt = name_excerpts[i]
        else:
            name_excerpt = None

        # Introduction
        if verbosity > 0:
            print(f"\n{t}Excerpt {i + 1}/{len(excerpts)}: {name_excerpt}")

        # Get the excerpt
        excerpt = excerpts[i]

        if remove_average_excerpts:
            mean = np.mean(excerpt)
            excerpt = excerpt - mean
            if verbosity > 1:
                print(f"Removing the average {mean} off the excerpt {name_excerpt}.")

        # Get the frequency
        if isinstance(freq_excerpts, list):
            freq_excerpt = freq_excerpts[i]
        else:
            freq_excerpt = freq_excerpts

        # Tries to open a WAV file if it is a path
        if isinstance(excerpt, str):
            if verbosity > 0:
                print(f"{t}Loading WAV file to excerpt from the path {excerpt}...")
            if not os.path.exists(excerpt):
                raise Exception(f"The file passed as parameter for excerpt {name_excerpt} doesn't exist: {excerpt}")

            audio_wav = wavfile.read(excerpt)
            freq_excerpt = audio_wav[0]
            excerpt = _convert_to_mono(audio_wav[1], mono_channel, verbosity, add_tabs + 1)

        # Turn list into ndarray
        if isinstance(excerpt, list):
            excerpt = np.array(excerpt)

        if len(excerpt.shape) > 1:
            raise Exception(f"The excerpt has more than one dimension: its shape is {excerpt.shape}. To perform the " +
                            f"find_delay function, each array must be 1-dimensional. If you are trying to find the " +
                            f"delay between two audio files, make sure that your files are in mono, or select one of " +
                            f"the channels.")

        if verbosity > 0:
            print(f"{t}\tThe excerpt {name_excerpt} contains {np.size(excerpt)} samples, at a rate of "
                  f"{freq_excerpt} Hz.\n")

        # Envelope
        if compute_envelope:
            if verbosity > 0:
                print(f"\t{t}Getting the envelope from the excerpt {name_excerpt}...")
            envelope_excerpt = _get_envelope(excerpt, freq_excerpt, window_size_env, overlap_ratio_env, filter_below,
                                             filter_over, verbosity, add_tabs+1)
            if verbosity > 0:
                print(f"\t{t}Envelope calculated.\n")
        else:
            envelope_excerpt = excerpt

        # Resampling
        if resampling_rate is not None:
            rate = resampling_rate
            if verbosity > 0:
                print(f"\t{t}Resampling excerpt {name_excerpt}...")
            y2 = _resample(envelope_excerpt, freq_excerpt, resampling_rate, window_size_res, overlap_ratio_res,
                           resampling_mode, verbosity, add_tabs + 1)
            if verbosity > 0:
                print(f"\t{t}Resampling done.\n")
        else:
            if freq_array != freq_excerpt:
                raise Exception(f"The rate of the two arrays you are trying to correlate are different ({freq_array} " +
                                f"Hz and {freq_excerpt} Hz). You must indicate a resampling rate to perform the " +
                                f"cross-correlation.")
            y2 = envelope_excerpt

        values = _cross_correlation(y1, y2, rate, freq_array, threshold,  return_delay_format, verbosity, add_tabs + 1)
        cross_correlation_normalized = values[0]
        return_value = values[1]
        max_correlation_value = values[2]
        index_max_correlation_value = values[3]

        # Plot and/or save the figure
        if plot_figure is not None or path_figures is not None:
            _create_figure(array, excerpt, freq_array, freq_excerpt, name_array, name_excerpt, envelope_array,
                           envelope_excerpt, y1, y2, compute_envelope, window_size_env, overlap_ratio_env,
                           filter_below, filter_over, resampling_rate, window_size_res, overlap_ratio_res,
                           cross_correlation_normalized, threshold, number_of_plots, return_delay_format, return_value,
                           max_correlation_value, index_max_correlation_value, plot_figure, path_figures, name_figures
                           + "_" + str(i + 1) + ".png", plot_intermediate_steps, x_format_figure, dark_mode, verbosity,
                           add_tabs+1)

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

    if verbosity > 0:
        print(f"\n{t}Complete delay finding function executed in: {dt.datetime.now() - time_before_function}")

    if return_correlation_values:
        return delays, correlation_values
    else:
        return delays
