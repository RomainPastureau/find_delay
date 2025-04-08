Version history
===============

2.17a (2025-04-??)
------------------
* The functions now accepts two new parameters, min_delay and max_delay, allowing to look for the delay with the
  maximum correlation in a range of possible delays.
* For negative values, invalid timedelta values wer displayed on the horizontal axes of the figures. This has been
  corrected.
* In order to gain space, hours on the horizontal axis are now shortened if the hour is equal to 0.

2.16 (2025-03-05)
-----------------
* Downgraded the requirements for Numpy (from 1.25.0 to 1.16.0), Scipy (from 1.11.0 to 1.5) and Matplotlib (from 3.7 to
  3.2) in order to ensure compatibility with older environments. Note: working with deprecated versions of modules is
  not recommended as they can include security issues.
* Removed the test files from the tar.gz, allowing to lower its size to 470 KB.

2.15 (2025-01-23)
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

2.14 (2024-12-17)
-----------------
* Corrected a critical bug that prevented to load the module
* Corrected a bug that prevented the name of the excerpts to properly appear in `find_delays` if they were set

2.13 (2024-12-15)
-----------------
*YANKED VERSION: Critical bug preventing to load the module*

* Corrected a bug that led to wrong behaviour when the parameters `window_size_res` and `window_size_env` are equal to
  `None`
* Separated the private functions from find_delay.py to their own file, `private_functions.py`
* When saving a figure, the function `_create_figure` now creates the subdirectories from `path_figure` that do not
  exist instead of returning an error
* Added two new parameters for `find_delay` and `find_delays` allowing to name the arrays on the figure
* Added one new test

2.12 (2024-11-16)
-----------------
* Modified the cross-correlation function to prevent numpy runtime warnings
* The parameter resampling_rate can now be set on `"auto"`, which is also the new default
* Corrected the linear resampling that had two parameters inverted
* Moved the cross-correlation to a function `_cross_correlation` to avoid repeating code
* Passing an array or an excerpt with more than one dimension now throws an exception
* Added one test to test the previous exception
* Prevented "Getting the Hilbert transform..." to appear when `verbosity=0`
* Added the parameter `add_tabs` for all functions with verbosity
* Removed the version history from the find_delay file to gain space
* Corrected the documentation

2.11 (2024-09-05)
-----------------
* Corrected bug that prevented figures to appear
* Added more WAV tests

2.10 (2024-09-05)
-----------------
*YANKED VERSION: Critical bug preventing the figures to appear*

* Corrected critical bug in stereo-to-mono conversion
* Added WAV tests

2.9 (2024-09-05)
----------------
*YANKED VERSION: Critical bug in stereo-to-mono conversion*

* Added the possibility to pass paths to WAV files as parameters of `find_delay` and `find_delays`
* Added the parameter `mono_channel` describing the method for converting multiple-channel audio to mono
* Added the function `_convert_to_mono` to perform the conversion to mono
* Corrected the display of negative delays when they are in timedelta format. A delay of -1 second
  will now print `-0:00:01` instead of `-1 day, 23:59:59`.
* Corrected a bug preventing the figure to display when the excerpt is found at the edges of the first
  array
* Closed the figure at the end of _create_figure to prevent warnings
* Added an FAQ page in the documentation
* Corrected typos and type errors in the documentation

2.8 (2024-06-19)
----------------
* Added tests with random numbers
* Corrected the link to the documentation on the PyPI page
* Replaced the strings by f-strings

2.7 (2024-05-09)
----------------
* Simplified `from find_delay.find_delay import find_delay` to `from find_delay import find_delay`
* Corrected scaling (again) on the aligned arrays graph
* Reestablished audio examples with downloadable WAV files when running the demo
* Added an example with randomly generated numbers

2.6 (2024-05-08)
----------------
* Removed demo audio files to lighten the Python package; they are still available on the main branch

2.5 (2024-05-08)
----------------
* **Turned find_delay into a Python package**, install with `py -m pip install find_delay`

2.4 (2024-05-08)
----------------
* The functions now look for correlation at the edges of the first array, in the case where the second array contains
  information that starts before the beginning, or ends after the end of the first
* Example 4 has been updated with one new audio file to demonstrate this change
* Adding a parameter x_format_figure that allows to display HH:MM:SS time on the x-axis
* Corrected a bug in the percentage progressions that prevented to display all the steps
* Added "Quick use for audio files" segment in the README file

2.3 (2024-05-02)
----------------
* Corrected a bug that prevented the figures to be saved as a file
* Plotting without intermediate steps now plots the graphs on top of each other, not side-by-side

2.2 (2024-05-02)
----------------
* "i_have_a_dream_excerpt2.wav" is now of lower amplitude to test the scaling on the graph overlay
* Arrays with different amplitudes now appear scaled on the graph overlay
* Excerpt numbers now start at 1 instead of 0 on the graphs in find_delays

2.1 (2024-04-25)
----------------
* Modified the overall functions so that they take a window size instead of a number of windows

2.0 (2024-04-24)
----------------
* Changed the parameter asking for a number of windows by a parameter asking for a window size instead
* Clarified the docstrings in the documentation of the functions
* Modified `find_delays` so that saving the figures would iterate the filenames instead of overwriting
* Modified `_get_envelope` and `_resample` so that a number of windows inferior to 1 would be set at 1
* Added documentation for `_create_figure` and simplified unused parameters
* Corrected broken figure saving
* Added figure saving for the 3 first examples

1.3 (2024-04-18)
----------------
* Removed unused function `_get_number_of_windows`

1.2 (2024-04-17)
----------------
* Added transparency of the second (orange) array on the graph overlay
* Clarified README.md and added figures

1.1 (2024-04-16)
----------------
* Added `find_delays`
* Created `_create_figure` containing all the plotting-related code
* Modified the graph plot when the max correlation is below threshold
* Minor corrections in docstrings

1.0 (2024-04-12)
----------------
* Initial release