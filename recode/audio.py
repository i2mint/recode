r"""Encoding audio

This module illustrates how one can use recode to make audio codecs.

Make pcm encoders and decoders:

>>> encode, decode = mk_pcm_audio_codec('int16')
>>> encoded = encode([1, 2, 3])
>>> encoded
b'\x01\x00\x02\x00\x03\x00'
>>> decode(encoded)
[1, 2, 3]

Or just encode directly:

>>> encode_pcm([1, 2, 3])
b'\x01\x00\x02\x00\x03\x00'


Or decode directly:

>>> encode_pcm([1, 2, 3])
b'\x01\x00\x02\x00\x03\x00'

TODO: Use builtin wave module to handle wav format as well.

"""

from typing import Union, Iterable
from numbers import Number
from recode.base import mk_encoder_and_decoder

Width = Union[str, int]
Waveform = Iterable[Number]


def mk_pcm_audio_codec(width: Width = 16, n_channels: int = 1):
    r"""Make a (encoder, decoder) pair for PCM data with given width and n_channels.

    PCM data is what's used in the uncompressed raw WAVE formats (such as used in CDs).
    See https://en.wikipedia.org/wiki/Pulse-code_modulation.

    :param width: The width of a sample (in bits, bytes, numpy dtype, pyaudio ...)
        (Will try to figure it out)
    :param n_channels: Number of channels
    :return: A (encoder, decoder) pair of functions that are inverse of each other

    >>> encode, decode = mk_pcm_audio_codec('int16')
    >>> encoded = encode([1, 2, 3])
    >>> encoded
    b'\x01\x00\x02\x00\x03\x00'
    >>> decode(encoded)
    [1, 2, 3]

    Let's check over more combinations of width and n_channels that we can decode
    what we encode to get back the same thing:

    >>> wf = [-3, -2, -1, 0, 1, 2, 3]
    >>> for width in [16, 2, 'int16', 'paInt16', 'PCM_16', 32, 4, 'int32']:
    ...     for channel in wf:
    ...         encode, decode = mk_pcm_audio_codec('int16')
    ...         encoded = encode(wf)
    ...         assert isinstance(encoded, bytes)
    ...         assert decode(encoded) == wf
    """
    struct_char = num_find_num_type_for(width)
    return mk_encoder_and_decoder(struct_char * n_channels, n_channels=n_channels)


def encode_pcm(wf: Waveform, width: Width = 16, n_channels: int = 1):
    r"""Encode waveform (e.g. list of numbers) into PCM bytes.

    :param wf: Waveform to encode
    :param width: The width of a sample (in bits, bytes, numpy dtype, pyaudio ...)
        (will try to figure it out by itself)
    :param n_channels: Number of channels
    :return: The pcm-bytes-encoded waveform

    >>> encode_pcm([1, 2, 3])
    b'\x01\x00\x02\x00\x03\x00'

    """
    encode, _ = mk_pcm_audio_codec(width, n_channels)
    return encode(wf)


def decode_pcm(pcm_bytes: bytes, width: Width = 16, n_channels: int = 1):
    r"""

    :param width: The width of a sample (in bits, bytes, numpy dtype, pyaudio ...)
        (Will try to figure it out)
    :param n_channels: Number of channels
    :return: The decoded waveform

    >>> decode_pcm(b'\x01\x00\x02\x00\x03\x00')
    [1, 2, 3]
    """
    _, decode = mk_pcm_audio_codec(width, n_channels)
    return decode(pcm_bytes)


# TODO: Can optimize (index) the data below to make search functions faster
num_type_synonyms = [
    {
        'dtype': 'int16',
        'soundfile': 'PCM_16',
        'pyaudio': 'paInt16',
        'n_bits': 16,
        'n_bytes': 2,
        'struct': 'h',
    },
    {
        'dtype': 'int8',
        'soundfile': 'PCM_S8',
        'pyaudio': 'paInt8',
        'n_bits': 8,
        'n_bytes': 1,
        'struct': 'b',
    },
    {
        'dtype': 'int24',
        'soundfile': 'PCM_24',
        'pyaudio': 'paInt24',
        'n_bits': 24,
        'n_bytes': 3,
        'struct': None,
    },
    {
        'dtype': 'int32',
        'soundfile': 'PCM_32',
        'pyaudio': 'paInt32',
        'n_bits': 32,
        'n_bytes': 4,
        'struct': 'i',
    },
    {
        'dtype': 'uint8',
        'soundfile': 'PCM_U8',
        'pyaudio': 'paUInt8',
        'n_bits': 8,
        'n_bytes': 1,
        'struct': 'B',
    },
    {
        'dtype': 'float32',
        'soundfile': 'FLOAT',
        'pyaudio': 'paFloat32',
        'n_bits': 32,
        'n_bytes': 4,
        'struct': 'f',
    },
    {
        'dtype': 'float64',
        'soundfile': 'DOUBLE',
        'pyaudio': None,
        'n_bits': 64,
        'n_bytes': 8,
        'struct': 'd',
    },
]


def num_find_num_type_for(
    num,
    target_num_sys='struct',
    num_sys_search_order=('n_bits', 'n_bytes', 'dtype', 'pyaudio', 'soundfile'),
):
    """Find the target_num_sys equivalent of input num checking multiple unit options"""
    for num_sys in num_sys_search_order:
        try:
            return num_type_for(num, num_sys, target_num_sys)
        except ValueError:
            'Just try the next num_sys...'


def num_type_for(num, num_sys='n_bits', target_num_sys='struct'):
    """Translate from one (sample width) number type to another.

    :param num:
    :param num_sys:
    :param target_num_sys:
    :return:

    >>> num_type_for(16, "n_bits", "soundfile")
    'PCM_16'
    >>> num_type_for(3, "n_bytes", "soundfile")
    'PCM_24'

    Tip: Use with `functools.partial` when you have some fix translation endpoints.

    >>> from functools import partial
    >>> get_dtype_from_n_bytes = partial(
    ...     num_type_for, num_sys="n_bytes", target_num_sys="dtype"
    ... )
    >>> get_dtype_from_n_bytes(8)
    'float64'
    """
    for d in num_type_synonyms:
        if num == d[num_sys]:
            if target_num_sys in d:
                return d[target_num_sys]
            else:
                raise ValueError(
                    f'Did not find any {target_num_sys} entry for {num_sys}={num}'
                )
    raise ValueError(f'Did not find any entry for {num_sys}={num}')
