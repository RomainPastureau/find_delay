This function tries to find the timestamp at which an excerpt of the current Audio instance begins.
The computation is performed through cross-correlation, by first turning the audio clips into downsampled and
filtered envelopes to accelerate the processing. The function returns the timestamp of the maximal correlation
value, or `None` if this value is below threshold. Optionally, it can also return a second element, the maximal
correlation value.

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
    The sample index, timestamp or timedelta of array1 at which array2 can be found (defined by the parameter
    return_delay_format), or `None` if array1 is not contained in array2.
float or None, optional
    Optionally, if return_correlation_value is `True`, the correlation value at the corresponding index/timestamp.
