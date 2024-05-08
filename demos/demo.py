from find_delay import *
import numpy as np
from scipy.io import wavfile

if __name__ == "__main__":

    # Example 1: random numbers
    array_1 = [24, 70, 28, 59, 13, 97, 63, 30, 89, 4, 8, 15, 16, 23, 42, 37, 70, 18, 59, 48, 41, 83, 99, 6, 24, 86]
    array_2 = [4, 8, 15, 16, 23, 42]

    find_delay(array_1, array_2, 1, 1, compute_envelope=False, resampling_rate=None, plot_figure=True,
               path_figure="figure_1.png", plot_intermediate_steps=False)

    # Example 2: sine function, different frequencies
    timestamps_1 = np.linspace(0, np.pi * 2, 200001)
    array_1 = np.sin(timestamps_1)
    timestamps_2 = np.linspace(np.pi * 0.5, np.pi * 0.75, 6001)
    array_2 = np.sin(timestamps_2)

    find_delay(array_1, array_2, 100000 / np.pi, 6000 / (np.pi / 4),
               compute_envelope=False, resampling_rate=1000, window_size_res=20000, overlap_ratio_res=0.5,
               resampling_mode="cubic", plot_figure=True, path_figure="figure_2.png", plot_intermediate_steps=True,
               verbosity=1)

    # Example 3: audio files
    audio_path = "i_have_a_dream_full_without_end.wav"
    audio_wav = wavfile.read(audio_path)
    audio_frequency = audio_wav[0]
    audio_array = audio_wav[1][:, 0]  # Turn to mono

    excerpt_path = "i_have_a_dream_excerpt.wav"
    excerpt_wav = wavfile.read(excerpt_path)
    excerpt_frequency = excerpt_wav[0]
    excerpt_array = excerpt_wav[1][:, 0]  # Turn to mono

    find_delay(audio_array, excerpt_array, audio_frequency, excerpt_frequency,
               compute_envelope=True, window_size_env=1e6, overlap_ratio_env=0.5,
               resampling_rate=1000, window_size_res=1e7, overlap_ratio_res=0.5, return_delay_format="timedelta",
               resampling_mode="cubic", plot_figure=True, path_figure="figure_3.png", plot_intermediate_steps=True,
               verbosity=1)

    # Example 4: multiple audio files
    excerpt2_path = "i_have_a_dream_excerpt2.wav"
    excerpt2_wav = wavfile.read(excerpt2_path)
    excerpt2_frequency = excerpt2_wav[0]
    excerpt2_array = excerpt2_wav[1][:, 0]  # Turn to mono

    excerpt3_path = "i_have_a_dream_excerpt_end.wav"
    excerpt3_wav = wavfile.read(excerpt3_path)
    excerpt3_frequency = excerpt3_wav[0]
    excerpt3_array = excerpt3_wav[1][:, 0]  # Turn to mono

    excerpt_not_present_path = "au_revoir.wav"
    excerpt_not_present_wav = wavfile.read(excerpt_not_present_path)
    excerpt_not_present_frequency = excerpt_not_present_wav[0]
    excerpt_not_present_array = excerpt_not_present_wav[1][:, 0]  # Turn to mono

    find_delays(audio_array,
                [excerpt_array, excerpt2_array, excerpt3_array, excerpt_not_present_array],
                audio_frequency,
                [excerpt_frequency, excerpt2_frequency, excerpt3_frequency, excerpt_not_present_frequency],
                compute_envelope=True, window_size_env=1e6, overlap_ratio_env=0.5,
                resampling_rate=1000, window_size_res=1e7, overlap_ratio_res=0.5, return_delay_format="timedelta",
                resampling_mode="cubic", threshold=0.8, plot_figure=True, plot_intermediate_steps=True, verbosity=1)
