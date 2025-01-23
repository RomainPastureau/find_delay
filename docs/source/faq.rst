FAQ
===

Why is the name of the package find-delay when installing, and find_delay when importing?
-----------------------------------------------------------------------------------------
The proper name of the package is ``find_delay``, with an **underscore**. Use it when importing the package in any of
your projects. However, PIP (the Package Installer for Python) automatically converts underscores to dashes when
creating packages, to uniformize with other programming languages. Hence, when installing the package in an environment,
you should run ```pip install find-delay``` (with a dash). Note that calling ```pip install find_delay``` (with an
underscore) should also work most of the time.

I want to use find_delay with audio files, but my audio files are not WAV...
----------------------------------------------------------------------------
find_delay only accepts WAV files as input. In the case where your files are in other formats, you first need to
convert your files to WAV, or to open the files with other Python libraries to access their arrays and frequency.

In the case where you opt for converting your files, you can use `ffmpeg <https://ffmpeg.org/>`_:
```ffmpeg -i original_file.mp3 converted_file.wav```. You can also use
`MediaHuman Audio Converter <https://www.mediahuman.com/audio-converter/>`_, which provides a graphical user interface
and allows to process files in batches. Note that with file format conversion generally comes data loss and quality
reduction.

To avoid this, you can also try to open your file directly in Python, as long as you manage to get an array of samples
and their frequency. You can look into `pydub <https://pypi.org/project/pydub/>`_ or
`audio2numpy <https://pypi.org/project/audio2numpy/>`_.

How does the function work for audio files?
-------------------------------------------
When trying to find the delay between two audio files, the function actually performs a cross-correlation of the
envelope of the two audio files. The envelope of the two audios is first calculated, using the absolute values of the
Hilbert transform (from scipy) over multiple overlapping windows. Then, the center values of each window is concatenated
to obtain an envelope that is, in most cases, over 99% correlated to an envelope calculated from the full audio; the
advantage is that this method is way much faster.

Tests performed on audio excerpts with an overlap of 0.5 and windows 50 000 samples long resulted in a 99.84% precision
and a 99.99998 % correlation with the envelope calculated on the full duration of the audio, for a very significant
time and memory usage gain - especially for very long files.

Once the envelope is calculated for both files, the function performs a cross-correlation; essentially, a correlation is
performed for each delay of an audio compared to the other. The delay matching the highest correlation value is then
returned.

Please note that the parameters of find_delay allow to manually set the envelope window size and overlap and the cross-
correlation threshold value.

The delay found by the function is erroneous...
-----------------------------------------------
This can happen for a number of reasons:
* You are working with audio files and did not set `compute_envelope` on `True`.
* The average value for one array is not the same as the other: you need to normalize your files. Try to set the
  parameters `remove_average_array_1` and `remove_average_array_2` on `True`.
* The sampling rate of the data is too low. Especially if you are working with audio files or neuro-imaging data, arrays
  down-sampled lower than 1000 Hz will be significantly harder to correlate.

I found an issue! How can I contact the developers?
---------------------------------------------------
If you found a bug, or if the package is not functioning as expected,
`please open an issue on GitHub <https://github.com/RomainPastureau/find_delay/issues/new?assignees=RomainPastureau&labels=bug&projects=&template=bug_report.md&title=>`_,
explaining your problem. Thanks in advance for your help :)

I have an idea for improvement!
-------------------------------
If you have an idea for improving the package, `please open an issue on Github too! <https://github.com/RomainPastureau/find_delay/issues/new?assignees=RomainPastureau&labels=enhancement&projects=&template=feature_request.md&title=>`_
