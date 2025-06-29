"""Tests both functions."""

import unittest
from find_delay import find_delay, find_delays
import random
from scipy.io import wavfile
import numpy as np
import datetime as dt


class Tests(unittest.TestCase):

    def test_find_delay_random(self):
        """Performs test for the find_delay function, involving an array of randomly generated numbers."""

        amount_of_numbers = 1000
        numbers = [i for i in range(-amount_of_numbers // 2, amount_of_numbers // 2)]
        random.shuffle(numbers)
        array_1_len = random.randint(len(numbers) // 2, len(numbers))
        array_1 = numbers[:array_1_len]
        print(f"Creating array with length {array_1_len}.")

        array_2_start = random.randint(0, array_1_len - 5)
        array_2_len = random.randint(5, array_1_len - array_2_start - 5)
        array_2 = array_1[array_2_start:array_2_start + array_2_len]
        print(f"Creating excerpt with length {array_2_len}, starting at {array_2_start}.")
        delay = find_delay(array_1, array_2, compute_envelope=False)

        assert (delay == array_2_start)

    def test_find_delays_random(self):
        """Performs test for the find_delays function, involving an array of randomly generated numbers."""

        amount_of_numbers = 1000
        numbers = [i for i in range(-amount_of_numbers // 2, amount_of_numbers // 2)]
        random.shuffle(numbers)
        array_len = random.randint(len(numbers) // 2, len(numbers))
        array = numbers[:array_len]
        print(f"Creating array with length {array_len}.")

        number_of_excerpts = 10
        excerpts = []
        excerpts_start = []
        for i in range(number_of_excerpts):
            excerpt_start = random.randint(0, array_len - 5)
            excerpts_start.append(excerpt_start)
            excerpt_len = random.randint(5, array_len - excerpt_start)
            excerpts.append(array[excerpt_start:excerpt_start + excerpt_len])
            print(f"Creating excerpt with length {excerpt_len}, starting at {excerpt_start}.")
        delays = find_delays(array, excerpts, compute_envelope=False)

        for i in range(number_of_excerpts):
            assert (delays[i] == excerpts_start[i])

    def test_wav(self):
        """Performs tests with WAV files (with opened WAVs)"""

        wav_full = wavfile.read("test_wav/test_full_2ch_48000Hz.wav")
        freq_full = wav_full[0]
        array_full = wav_full[1][:, 0]

        # Excerpt at 2000 ms
        wav_excerpt1 = wavfile.read("test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav")
        freq_excerpt1 = wav_excerpt1[0]
        array_excerpt1 = wav_excerpt1[1][:, 0]

        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, return_correlation_value=True,
                                 path_figure="figures/arrays/figure_1.png")
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

        # Excerpt at 2000 ms, 6 ch
        wav_excerpt2 = wavfile.read("test_wav/test_excerpt_2000ms_inside_6ch_48000Hz.wav")
        freq_excerpt2 = wav_excerpt2[0]
        array_excerpt2 = wav_excerpt2[1][:, 0]

        delay, corr = find_delay(array_full, array_excerpt2, freq_full, freq_excerpt2, return_correlation_value=True,
                                 mono_channel="average", path_figure="figures/arrays/figure_2.png")
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

        # Excerpt at 2000 ms, 44100 Hz
        wav_excerpt3 = wavfile.read("test_wav/test_excerpt_2000ms_inside_2ch_44100Hz.wav")
        freq_excerpt3 = wav_excerpt3[0]
        array_excerpt3 = wav_excerpt3[1][:, 0]

        delay, corr = find_delay(array_full, array_excerpt3, freq_full, freq_excerpt3, resampling_rate=1000,
                                 return_correlation_value=True, path_figure="figures/arrays/figure_3.png")
        assert (delay == 96000)
        assert (round(corr, 3) == 0.982)

        # Excerpt starting 500 ms before the onset
        wav_excerpt4 = wavfile.read("test_wav/test_excerpt_2000ms_onset_-500_2ch_48000Hz.wav")
        freq_excerpt4 = wav_excerpt4[0]
        array_excerpt4 = wav_excerpt4[1][:, 0]

        delay, corr = find_delay(array_full, array_excerpt4, freq_full, freq_excerpt4, return_correlation_value=True,
                                 path_figure="figures/arrays/figure_4.png")
        assert (delay == -24001)
        assert (round(corr, 3) == 0.977)

        # Excerpt ending 500 ms before the offset, mono
        wav_excerpt5 = wavfile.read("test_wav/test_excerpt_2000ms_onset_-500_1ch_48000Hz.wav")
        freq_excerpt5 = wav_excerpt5[0]
        array_excerpt5 = wav_excerpt5[1]

        delay, corr = find_delay(array_full, array_excerpt5, freq_full, freq_excerpt5, return_correlation_value=True,
                                 path_figure="figures/arrays/figure_5.png")
        assert (delay == -24001)
        assert (round(corr, 3) == 0.977)

        # Excerpt ending 500 ms after the offset
        wav_excerpt6 = wavfile.read("test_wav/test_excerpt_2000ms_offset_+500_2ch_48000Hz.wav")
        freq_excerpt6 = wav_excerpt6[0]
        array_excerpt6 = wav_excerpt6[1][:, 0]

        delay, corr = find_delay(array_full, array_excerpt6, freq_full, freq_excerpt6, return_correlation_value=True,
                                 path_figure="figures/arrays/figure_6.png")
        assert (delay == 216002)
        assert (round(corr, 3) == 0.966)

        # Test with find_delays
        arrays_excerpts = [array_excerpt1, array_excerpt2, array_excerpt3,
                           array_excerpt4, array_excerpt5, array_excerpt6]
        freqs_excerpts = [freq_excerpt1, freq_excerpt2, freq_excerpt3, freq_excerpt4, freq_excerpt5, freq_excerpt6]
        delays, corrs = find_delays(array_full, arrays_excerpts, freq_full, freqs_excerpts,
                                    resampling_rate=1000, return_correlation_values=True,
                                    path_figures="figures/arrays/", name_figures="figure_delays")

        assert (delays == [96000, 96000, 96000, -24000, -24000, 216000])
        assert (np.all(np.round(corrs, 3) == [0.982, 0.982, 0.982, 0.977, 0.977, 0.965]))

    def test_wav_paths(self):
        """Performs tests with WAV files (with WAV as paths)"""

        # Excerpt at 2000 ms
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                                 return_correlation_value=True, path_figure="figures/wav/figure_1.png")
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

        # Excerpt at 2000 ms, 6 ch
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_inside_6ch_48000Hz.wav",
                                 return_correlation_value=True, mono_channel="average",
                                 path_figure="figures/wav/figure_2.png")
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

        # Excerpt at 2000 ms, 44100 Hz
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_inside_2ch_44100Hz.wav",
                                 resampling_rate=1000, return_correlation_value=True,
                                 path_figure="figures/wav/figure_3.png")
        assert (delay == 96000)
        assert (round(corr, 3) == 0.982)

        # Excerpt starting 500 ms before the onset
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_onset_-500_2ch_48000Hz.wav",
                                 return_correlation_value=True, path_figure="figures/wav/figure_4.png")
        assert (delay == -24001)
        assert (round(corr, 3) == 0.977)

        # Excerpt ending 500 ms before the offset, mono
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_onset_-500_1ch_48000Hz.wav",
                                 return_correlation_value=True, path_figure="figures/wav/figure_5.png")
        assert (delay == -24001)
        assert (round(corr, 3) == 0.977)

        # Excerpt ending 500 ms after the offset
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_offset_+500_2ch_48000Hz.wav",
                                 return_correlation_value=True, path_figure="figures/wav/figure_6.png")
        assert (delay == 216002)
        assert (round(corr, 3) == 0.966)

        # Find delays
        delays, corrs = find_delays("test_wav/test_full_2ch_48000Hz.wav",
                                    ["test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                                     "test_wav/test_excerpt_2000ms_inside_6ch_48000Hz.wav",
                                     "test_wav/test_excerpt_2000ms_inside_2ch_44100Hz.wav",
                                     "test_wav/test_excerpt_2000ms_onset_-500_2ch_48000Hz.wav",
                                     "test_wav/test_excerpt_2000ms_onset_-500_1ch_48000Hz.wav",
                                     "test_wav/test_excerpt_2000ms_offset_+500_2ch_48000Hz.wav"],
                                    resampling_rate=1000, return_correlation_values=True,
                                    path_figures="figures/wav/", name_figures="figure_delays", verbosity=2)

        assert (delays == [96000, 96000, 96000, -24000, -24000, 216000])
        assert (np.all(np.round(corrs, 3) == [0.982, 0.982, 0.982, 0.977, 0.977, 0.965]))

    def test_wav_paths_examples(self):
        """Tests the examples from the documentation."""
        delay = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                           "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav")
        assert (delay == 96002)

        delay = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                           "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav", mono_channel="average")
        assert (delay == 96002)

        delay = (find_delay("test_wav/test_full_2ch_48000Hz.wav",
                            "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                            return_delay_format="timedelta",
                            plot_figure=False, plot_intermediate_steps=True,
                            verbosity=1))

        assert (delay.total_seconds() == 2.000042)

        delay = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                           "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                           return_delay_format="timedelta",
                           plot_figure=False,
                           plot_intermediate_steps=True,
                           path_figure="../docs/source/images/figure_example.png",
                           x_format_figure="time",
                           name_array_1="Original",
                           name_array_2="Excerpt")

    def test_stereo_exception(self):
        """With both arrays being stereo, checks that the function find_delay returns an exception."""
        wav_full = wavfile.read("test_wav/test_full_2ch_48000Hz.wav")
        freq_full = wav_full[0]
        array_full = wav_full[1]

        # Excerpt at 2000 ms - we keep the two channels
        wav_excerpt1 = wavfile.read("test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav")
        freq_excerpt1 = wav_excerpt1[0]
        array_excerpt1 = wav_excerpt1[1]

        self.assertRaises(Exception, find_delay, array_full, array_excerpt1, freq_full, freq_excerpt1)

    def test_resample_window_none(self):
        """Tests that when the parameter `window_res` is set on `None`, the expected behaviour is triggered:
        the whole array is selected."""

        # Excerpt at 2000 ms
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                                 resampling_rate=1000, window_size_res=None, return_correlation_value=True,
                                 path_figure="figures/wav/figure_1.png", verbosity=0)
        assert (delay == 96000)
        assert (round(corr, 3) == 0.982)

    def test_remove_average(self):
        """Tests that when the parameters `remove_average` is set on `True`, it doesn't break the normal
        flow and works for data that doesn't have an average around 0."""

        # Normal files
        wav_full = wavfile.read("test_wav/test_full_2ch_48000Hz.wav")
        freq_full = wav_full[0]
        array_full = wav_full[1][:, 0]
        wav_excerpt1 = wavfile.read("test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav")
        freq_excerpt1 = wav_excerpt1[0]
        array_excerpt1 = wav_excerpt1[1][:, 0]

        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, remove_average_array_1=True,
                                 remove_average_array_2=True, return_correlation_value=True, verbosity=2)
        assert (delay == 95999)
        assert (round(corr, 3) == 0.981)

        # Modified file
        array_excerpt1 = array_excerpt1 - 10000

        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, remove_average_array_1=True,
                                 remove_average_array_2=True, return_correlation_value=True, verbosity=2)
        assert (delay == 95999)
        assert (round(corr, 3) == 0.981)

    def test_dark_mode(self):
        # Excerpt at 2000 ms
        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                                 return_correlation_value=True, plot_figure=True, dark_mode=True)
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

    def test_return_values_types(self):
        # Sample
        wav_full = wavfile.read("test_wav/test_full_2ch_48000Hz.wav")
        freq_full = wav_full[0]
        array_full = wav_full[1][:, 0]

        # Excerpt at 2000 ms
        wav_excerpt1 = wavfile.read("test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav")
        freq_excerpt1 = wav_excerpt1[0]
        array_excerpt1 = wav_excerpt1[1][:, 0]

        # No parameter
        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, return_correlation_value=True,
                                 path_figure="figures/arrays/figure_no_param.png")
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

        # Index
        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, return_correlation_value=True,
                                 return_delay_format="index", path_figure="figures/arrays/figure_index.png")
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

        # Second
        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, return_correlation_value=True,
                                 return_delay_format="s", path_figure="figures/arrays/figure_sec.png")
        assert np.isclose(delay, 2 + 2/48000)
        assert (round(corr, 3) == 0.984)

        # Millisecond
        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, return_correlation_value=True,
                                 return_delay_format="ms", path_figure="figures/arrays/figure_ms.png")
        assert np.isclose(delay, 2000 + 2/48)
        assert (round(corr, 3) == 0.984)

        # Datetime
        delay, corr = find_delay(array_full, array_excerpt1, freq_full, freq_excerpt1, return_correlation_value=True,
                                 return_delay_format="timedelta", path_figure="figures/arrays/figure_datetime.png")
        assert delay == dt.timedelta(seconds=2 + 2/48000)
        assert (round(corr, 3) == 0.984)

    def test_min_max_delay(self):
        array = np.array([10, 6, 3, 4, 9, 8, 16, 23, 12, 1, 0, 9, 6, 3, 7, 6, 5, 1, 0, 4, 9, 24, 42])
        excerpt = array[5:12]

        delay = find_delay(array, excerpt, compute_envelope=False, plot_figure=True)
        assert delay == 5

        delay = find_delay(array, excerpt, compute_envelope=False, plot_figure=True, min_delay=10, max_delay=20)

        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                                 return_correlation_value=True, plot_figure=True, plot_intermediate_steps=True)
        assert (delay == 96002)
        assert (round(corr, 3) == 0.984)

        delay, corr = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                                 "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav",
                                 return_correlation_value=True, plot_figure=True, plot_intermediate_steps=True,
                                 min_delay=100000, max_delay=200000)
        assert delay is None
        assert corr is None

        array_1 = np.array([10,  98,  56, -39,  50, -96, -48, -13,  97, -59, -82,  16,  15,
                            14,  65,  84, -32, -65,  52, -45,  98, -38, -50, -96,  38,  52,
                            49, -33, -71,  34, -52,  99,  66, -92, -82,  90, -91,  77, -30,
                           -24, -91,  83,   5, -57, -12,  62,  17,  46,  17, -64,  98, -38,
                           -50, -96,  38,  52,  49,   7, -23,  42,   0,  88,  -9,  -9,  18])

        array_2 = np.array([98, -38, -50, -96,  38,  52,  49])

        delay, corr = find_delay(array_1, array_2, compute_envelope=False, return_correlation_value=True,
                                 plot_figure=True, plot_intermediate_steps=True)

        assert delay == 20
        assert np.isclose(corr, 1.0)

        delay, corr = find_delay(array_1, array_2, min_delay=25, compute_envelope=False,
                                 return_correlation_value=True,
                                 plot_figure=True, plot_intermediate_steps=True)

        assert delay == 50
        assert np.isclose(corr, 1.0)

        array_1 = np.array([10,  98,  56, -39,  50, -96, -48, -13,  97, -59, -82,  16,  15,
                            14,  65,  84, -32, -65,  52, -45,  98, -38, -48, -96,  38,  52,
                            49, -33, -71,  34, -52,  99,  66, -92, -82,  90, -91,  77, -30,
                           -24, -91,  83,   5, -57, -12,  62,  17,  46,  17, -64,  98, -38,
                           -50, -96,  38,  52,  49,   7, -23,  42,   0,  88,  -9,  -9,  18])

        delay, corr = find_delay(array_1, array_2, compute_envelope=False, return_correlation_value=True,
                                 plot_figure=False, plot_intermediate_steps=True)
        assert delay == 50
        assert np.isclose(corr, 1.0)

        delay, corr = find_delay(array_1, array_2, max_delay=25, compute_envelope=False,
                                 return_correlation_value=True,
                                 plot_figure=False, plot_intermediate_steps=True)
        assert delay == 20
        assert np.isclose(corr, 0.9999482290589539)


    def test_big_chunks(self):
        find_delay("../demos/i_have_a_dream_full_without_end.wav",
                   "../demos/i_have_a_dream_full_without_end_+800ms.wav",
                   plot_figure=True, return_delay_format="ms")

    def test_sample(self):
        delay_index = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                           "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav")
        delay_sample = find_delay("test_wav/test_full_2ch_48000Hz.wav",
                           "test_wav/test_excerpt_2000ms_inside_2ch_48000Hz.wav", return_delay_format="sample")
        assert (delay_index == delay_sample)

    def test_return_none_if_below_threshold(self):
        array = np.array([10, 6, 3, 4, 9, 8, 16, 23, 12, 1, 0, 9, 6, 3, 7, 6, 5, 1, 0, 4, 9, 24, 42])
        excerpt = array[5:12]

        delay = find_delay(array, excerpt, return_none_if_below_threshold=True, compute_envelope=False,
                           plot_figure=False)
        assert delay == 5

        delay = find_delay(array, excerpt, return_none_if_below_threshold=False, compute_envelope=False,
                           plot_figure=False)
        assert delay == 5

        excerpt = np.array([42, 23, 16, 15, 108, 8, 4])
        delay = find_delay(array, excerpt, return_none_if_below_threshold=True, compute_envelope=False,
                           plot_figure=False)
        assert delay is None

        excerpt = np.array([42, 23, 16, 15, 108, 8, 4])
        delay = find_delay(array, excerpt, return_none_if_below_threshold=False, compute_envelope=False,
                           plot_figure=False)
        assert delay == 18

    def test_return_corr_array(self):
        array = np.array([10, 6, 3, 4, 9, 8, 16, 23, 12, 1, 0, 9, 6, 3, 7, 6, 5, 1, 0, 4, 9, 24, 42])
        excerpt = array[5:12]

        expected_corr = [[-0.04659149, -0.52343527, -0.78245225, -0.3523535, 0.36301162, 0.08987192,
                          -0.49961267, -0.48278441, -0.65668238, -0.37573296, 0.44536903, 1.,
                          0.44365251, -0.25002696, -0.4038325, 0.17737077, 0.10685918, -0.22450604,
                          0.44002936, 0.91937351, 0.28552471, -0.28467871, -0.46145835, -0.71754499,
                          -0.22749815, 0.58026659, 0.85429219, 0.25915351, -0.10094823],
                         [-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                          12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.]]

        delay, corr = find_delay(array, excerpt, return_none_if_below_threshold=True, return_correlation_value="array",
        compute_envelope=False, plot_figure=False)
        assert delay == 5
        assert np.allclose(corr, expected_corr)

        delay, corr = find_delay(array, excerpt, return_none_if_below_threshold=False, return_correlation_value="array",
                           compute_envelope=False, plot_figure=False)
        assert delay == 5
        assert np.allclose(corr, expected_corr)

        expected_corr = None

        excerpt = np.array([42, 23, 16, 15, 108, 8, 4])
        delay, corr = find_delay(array, excerpt, return_none_if_below_threshold=True, return_correlation_value="array",
                           compute_envelope=False,  plot_figure=False)
        assert delay is None
        assert corr is None

        expected_corr = [[-0.32735528, -0.44103888, 0.64644566, 0.15662565, -0.39843367,
                          -0.41972282, 0.04016218, -0.33611136, 0.08384929, 0.66772067,
                          0.19186655, -0.41634635, -0.3156566, 0.35944265, 0.20045834,
                          -0.42795734, 0.0967511, 0.38445588, 0.352152, -0.3958831,
                          -0.56370764, -0.30150797, -0.29768461, 0.20248751, 0.77340207,
                          -0.3010363, -0.1998609, -0.01703675, 0.13581762],
                         [-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.,
                          7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
                          20., 21., 22.]]

        excerpt = np.array([42, 23, 16, 15, 108, 8, 4])
        delay, corr = find_delay(array, excerpt, return_none_if_below_threshold=False, return_correlation_value="array",
                           compute_envelope=False, plot_figure=False)
        assert delay == 18
        assert np.allclose(corr, expected_corr)

        delay, corr = find_delay(array, excerpt, return_none_if_below_threshold=False, return_correlation_value="array",
        min_delay=2, max_delay=8, compute_envelope=False, plot_figure=False)
        assert delay == 3
        assert np.allclose(corr, [[0.08384929, 0.66772067, 0.19186655, -0.41634635,
                                  -0.3156566, 0.35944265, 0.20045834], [2., 3., 4., 5., 6., 7., 8.]])

        array = np.array([10, 6, 3, 4, 9, 8, 16, 23, 12, 1, 0])
        excerpt1 = np.array([3, 4, 9, 8, 16])
        excerpt2 = np.array([16, 8, 9, 4, 3])

        delays, corrs = find_delays(array, [excerpt1, excerpt2], return_none_if_below_threshold=True,
                                    return_correlation_values="array", compute_envelope=False, plot_figure=False)

        assert delays[0] == 2
        assert delays[1] is None

        assert np.allclose(corrs[0], [[0.86874449, 0.50627954, 0.38704396, -0.02614614, 0.01592486,
                             0.49526056, 1., 0.87868205, 0.27831121, -0.42834593,
                             -0.78773589, -0.77257866, -0.58948756, -0.5429653, 0.],
                            [-4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
        assert corrs[1] is None

if __name__ == "__main__":
    unittest.main()

