from setuptools import setup, find_packages

setup(name='find_delay',
      version='2.16',
      packages=find_packages(exclude=["demos"]),
      include_package_data=True)
      # exclude_package_data={"demos": ["au_revoir.wav",
      #                                 "i_have_a_dream_excerpt.wav",
      #                                 "i_have_a_dream_excerpt2.wav",
      #                                 "i_have_a_dream_excerpt_end.wav",
      #                                 "i_have_a_dream_full_without_end.wav",
      #                                 "i_have_a_dream_full_without_end_+800ms.wav"]})
