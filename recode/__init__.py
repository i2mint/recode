"""
Make codecs for fixed size structured chunks serialization and deserialization of
sequences, tabular data, and time-series.


"""

from recode.util import spy, get_struct, list_of_dicts
from recode.base import *

from recode.base import mk_encoder_and_decoder  # main interface function
from recode.audio import mk_pcm_audio_codec
