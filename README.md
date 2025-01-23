# find_delay 2.15
[![Documentation Status](https://readthedocs.org/projects/find-delay/badge/?version=latest)](https://find-delay.readthedocs.io/en/latest/?badge=latest)

[PyPI page](https://pypi.org/project/find-delay/)

Author: Romain Pastureau

## What is find_delay?
**find_delay** is a **Python package** that tries to find the delay where a time series appears in another via 
cross-correlation. It can theoretically work with any time series (see the examples in the 
[demos folder](https://github.com/RomainPastureau/find_delay/tree/main/demos), but was created to try to align 
audio files.
**[Read the documentation here!](https://find-delay.readthedocs.io/en/latest/)**

## How to
The best way to use this function is to install the find_delay module for Python by running 
`py -m pip install find-delay`.

You can then import the function by writing `from find_delay import find_delay` (or `from find_delay import find_delays`
if you want to locate multiple excerpts in one big time series).

You can also run demos/demo.py to get four examples (in that case, you will need to download the .wav files present in 
the repository and place them in the same folder for examples 3 and 4).

## Quick use for audio files
To find when an excerpt starts in an audio file, use the `find_delay` function and fill only the first two parameters, 
by indicating the path to the corresponding WAV files; leave the other parameters default (just set `plot_figure = True`
if you want to visualize the output of the function).

## Specifics
The function accepts two arrays containing time series - the time series can be of different frequency or amplitude.

The function can then calculate the envelope of the time series (recommended for audio files) and apply a band-pass 
filter to the result.

The function can also resample the arrays (necessary when the two time series do not have the same frequency).

Finally, the function performs the cross-correlation between the two arrays.

The results can be then plotted if the corresponding parameters are activated, and the function returns the delay at 
which to find the second array in the first by selecting the delay with the maximum correlation value (optionally, the 
function can also return this correlation value).

## Dependencies
* **Matplotlib** for the plots
* **Numpy** for handling the numerical arrays
* **Scipy** for loading the WAV files, performing the resampling, calculating the envelope, and applying a band-pass 
  filter.

## Examples
### Delay between two numerical time series
```    
array_1 = [24, 70, 28, 59, 13, 97, 63, 30, 89, 4, 8, 15, 16, 23, 42, 37, 70, 18, 59, 48, 41, 83, 99, 6, 24, 86]
array_2 = [4, 8, 15, 16, 23, 42]

find_delay(array_1, array_2, compute_envelope=False, plot_figure=True, path_figure="figure_1.png")
```

![Delay between two numerical time series](https://raw.githubusercontent.com/RomainPastureau/find_delay/package/demos/figure_1.png)

### Delay between a sine function and a portion of it, different frequencies
```
timestamps_1 = np.linspace(0, np.pi * 2, 200001)
array_1 = np.sin(timestamps_1)
timestamps_2 = np.linspace(np.pi * 0.5, np.pi * 0.75, 6001)
array_2 = np.sin(timestamps_2)

find_delay(array_1, array_2, 100000 / np.pi, 6000 / (np.pi / 4),
           compute_envelope=False, resampling_rate=1000, window_size_res=20000, overlap_ratio_res=0.5,
           resampling_mode="cubic", plot_figure=True, path_figure="figure_2.png", plot_intermediate_steps=True,
           verbosity=1)
```

![Delay between a sine function and a portion of it, different frequencies](https://raw.githubusercontent.com/RomainPastureau/find_delay/package/demos/figure_2.png)

### Delay between an audio file and an excerpt from it
```
find_delay("i_have_a_dream_full_speech.wav", "i_have_a_dream_excerpt.wav",
           return_delay_format="timedelta",
           plot_figure=True, path_figure="figure_3.png", plot_intermediate_steps=True,
           verbosity=1)
```

![Delay between an audio file and an excerpt from it](https://raw.githubusercontent.com/RomainPastureau/find_delay/package/demos/figure_3.png)

[Find more examples here!](https://find-delay.readthedocs.io/en/latest/examples/delay_audio_files.html)

### Latest version
*2.15 (2025-01-23)*
-----------------
* Added the parameters `remove_average_array` in both `find_delay` and `find_delays`
* Added the parameter `dark_mode` in both `find_delay` and `find_delays`
* Incorporated the values of `name_array_1`, `name_array_2`, `name_array` and `name_excerpts` in the functions verbosity
* Removed a print in `_convert_mono()`
* Corrected a bug that prevented to save a figure if no directory was passed in the parameter
* Corrected a bug that prevented to see the proper time scale if `x_format_figure` was set on `"time"`
* Corrected a bug that displayed erroneous times on the x-axis of the cross-correlation subplot
* Corrected the execution time message that was appearing even if `verbosity` was set on 0
* Added two example pages in the documentation and linked them in the documentation index page
* Added a test for the parameter `remove_average`
* Added a test for the documentation examples
* Corrected erroneous documentation version number

[See version history](https://find-delay.readthedocs.io/en/latest/version_history.html)

If you detect any bug, please [open an issue](https://github.com/RomainPastureau/find_delay/issues/new).

Thanks! 🦆

<a href="https://www.buymeacoffee.com/romainpastureau" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-green.png" alt="Buy Me A Coffee" height="41" width="174"></a>
