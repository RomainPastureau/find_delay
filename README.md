# ** find_delay 1.1 **

Author: Romain Pastureau

## What is find_delay?
**find_delay** is a **Python function** that tries to find the delay where a time series appear in another via cross-correlation. It can theoretically work with any time series (see the examples in the ``__main__`` of the file), but was created to try to align audio files.

## Specifics
The function accepts to arrays containing time series - the time series can be of different frequency or amplitude - the only condition for the function to work is that the second array must be entirely contained into the first (note: if it is not the case, try to run the function with the beginning of the second array only).

The function can then calculate the envelope of the time series (recommended for audio files) and apply a band-pass filter to the result.

The function can also resample the arrays (necessary when the two time series do not have the same frequency).

Finally, the function performs the cross-correlation between the two arrays.

The results can be then plotted if the corresponding parameters are activated, and the function returns the delay at which to find the second array in the first by selecting the delay with the maximum correlation value (optionally, the function can also return this correlation value).

## How to
The best way to use this function is to download the find_delay.py file and place it in the same folder as your Python script. You can then import the function by writing `from find_delay import find_delay` (or `from find_delay import find_delays` if you want to locate multiple excerpt in one big time series).

You can also run find_delay.py to get four examples (in that case, you will need to download the .wav files present in the repository and place them in the same folder for examples 3 and 4).

## Dependencies
* **Matplotlib** for the plots
* **Numpy** for handling the numerical arrays
* **Scipy** for loading the wav files, performing the resampling, calculating the envelope and applying a band-pass filter.

## Examples
1. Delay between two numerical time series
```    
array_1 = [24, 70, 28, 59, 13, 97, 63, 30, 89, 4, 8, 15, 16, 23, 42, 37, 70, 18, 59, 48, 41, 83, 99, 6, 24, 86]
array_2 = [4, 8, 15, 16, 23, 42]

find_delay(array_1, array_2, 1, 1, compute_envelope=False, resampling_rate=None, plot_figure=True,plot_intermediate_steps=True)
```

![Delay between two numerical time series](https://github.com/RomainPastureau/find_delay/blob/main/figure_1.png?raw=true)

2. Delay between a sine function and a portion of it, different frequencies
```
timestamps_1 = np.linspace(0, np.pi * 2, 200001)
array_1 = np.sin(timestamps_1)
timestamps_2 = np.linspace(np.pi * 0.5, np.pi * 0.75, 6001)
array_2 = np.sin(timestamps_2)

find_delay(array_1, array_2, 100000 / np.pi, 6000 / (np.pi / 4),
           compute_envelope=False, resampling_rate=1000, number_of_windows_res=10, overlap_ratio_res=0.5,
           resampling_mode="cubic", plot_figure=True, plot_intermediate_steps=True, verbosity=1)
```

![Delay between a sine function and a portion of it, different frequencies](https://github.com/RomainPastureau/find_delay/blob/main/figure_2.png?raw=true)

3. Delay between an audio file and an excerpt from it
```
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
```

![Delay between an audio file and an excerpt from it](https://github.com/RomainPastureau/find_delay/blob/main/figure_3.png?raw=true)

### Version history
1.2 (2024-04-17) · Added transparency of the second (orange) array on the graph overlay
                 · Clarified README.md and added figures  
1.1 (2024-04-16) · Added `find_delays`
                 · Created `_create_figure` containing all the plotting-related code
                 · Modified the graph plot when the max correlation is below threshold
                 · Minor corrections in docstrings
1.0 (2024-04-12) · Initial release