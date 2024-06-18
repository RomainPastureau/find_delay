"""Tests both functions."""

import unittest
from find_delay import find_delay, find_delays
import random


class Tests(unittest.TestCase):

    def test_find_delay_random(self):
        """Performs test for the find_delay function, involving an array of randomly generated numbers."""

        amount_of_numbers = 1000
        numbers = [i for i in range(-amount_of_numbers//2, amount_of_numbers//2)]
        random.shuffle(numbers)
        array_1_len = random.randint(len(numbers)//2, len(numbers))
        array_1 = numbers[:array_1_len]
        print(f"Creating array with length {array_1_len}.")

        array_2_start = random.randint(0, array_1_len - 5)
        array_2_len = random.randint(5, array_1_len - array_2_start - 5)
        array_2 = array_1[array_2_start:array_2_start + array_2_len]
        print(f"Creating excerpt with length {array_2_len}, starting at {array_2_start}.")
        delay = find_delay(array_1, array_2, compute_envelope=False)

        assert(delay == array_2_start)

    def test_find_delays_random(self):
        """Performs test for the find_delays function, involving an array of randomly generated numbers."""

        amount_of_numbers = 1000
        numbers = [i for i in range(-amount_of_numbers//2, amount_of_numbers//2)]
        random.shuffle(numbers)
        array_len = random.randint(len(numbers)//2, len(numbers))
        array = numbers[:array_len]
        print(f"Creating array with length {array_len}.")

        number_of_excerpts = 10
        excerpts = []
        excerpts_start = []
        for i in range(number_of_excerpts):
            excerpt_start = random.randint(0, array_len - 5)
            excerpts_start.append(excerpt_start)
            excerpt_len = random.randint(5, array_len - excerpt_start - 5)
            excerpts.append(array[excerpt_start:excerpt_start + excerpt_len])
            print(f"Creating excerpt with length {excerpt_len}, starting at {excerpt_start}.")
        delays = find_delays(array, excerpts, compute_envelope=False)

        for i in range(number_of_excerpts):
            assert(delays[i] == excerpts_start[i])


if __name__ == "__main__":
    unittest.main()
