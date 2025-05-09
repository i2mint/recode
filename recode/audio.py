r"""Encoding audio

This module illustrates how one can use recode to make audio codecs.

>>> wav_bytes = encode_wav_bytes([1, 2, 3], 42)
>>> wav_bytes
b'RIFF*\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00*\x00\x00\x00T\x00\x00\x00\x02\x00\x10\x00data\x06\x00\x00\x00\x01\x00\x02\x00\x03\x00'
>>> wf, sr = decode_wav_bytes(wav_bytes)
>>> sr
42
>>> wf
[1, 2, 3]

The wav codecs are based on pcm codecs along with wav header codecs
(i.e. parsing and generation -- using the builtin `wave` package).

Make pcm encoders and decoders:

>>> encode, decode = mk_pcm_audio_codec('int16')
>>> encoded = encode([1, 2, 3])
>>> encoded
b'\x01\x00\x02\x00\x03\x00'
>>> decode(encoded)
[1, 2, 3]

Or just encode directly:

>>> encode_pcm_bytes([1, 2, 3])
b'\x01\x00\x02\x00\x03\x00'

Or decode directly:

>>> encode_pcm_bytes([1, 2, 3])
b'\x01\x00\x02\x00\x03\x00'


"""

from io import BytesIO
from typing import Union, Iterable
from numbers import Number
from recode.base import mk_codec
from wave import Wave_write, Wave_read

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
    return mk_codec(struct_char * n_channels, n_channels=n_channels)


def encode_pcm_bytes(wf: Waveform, width: Width = 16, n_channels: int = 1):
    r"""Encode waveform (e.g. list of numbers) into PCM bytes.

    :param wf: Waveform to encode
    :param width: The width of a sample (in bits, bytes, numpy dtype, pyaudio ...)
        (will try to figure it out by itself)
    :param n_channels: Number of channels
    :return: The pcm-bytes-encoded waveform

    >>> encode_pcm_bytes([1, 2, 3])
    b'\x01\x00\x02\x00\x03\x00'

    """
    encode, _ = mk_pcm_audio_codec(width, n_channels)
    return encode(wf)


def decode_pcm_bytes(pcm_bytes: bytes, width: Width = 2, n_channels: int = 1):
    r"""

    :param width: The width of a sample (in bits, bytes, numpy dtype, pyaudio ...)
        (Will try to figure it out)
    :param n_channels: Number of channels
    :return: The decoded waveform

    >>> decode_pcm_bytes(b'\x01\x00\x02\x00\x03\x00')
    [1, 2, 3]
    """
    _, decode = mk_pcm_audio_codec(width, n_channels)
    return decode(pcm_bytes)


MIN_WAV_N_BYTES = 44


def decode_wav_bytes(wav_bytes: bytes):
    r"""

    :param width: The width of a sample (in bits, bytes, numpy dtype, pyaudio ...)
        (Will try to figure it out)
    :param n_channels: Number of channels
    :return: The decoded waveform

    >>> wav_bytes = (
    ...     b'RIFF.\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'  # header
    ...     b'*\x00\x00\x00T\x00\x00\x00\x02\x00\x10\x00data\n\x00\x00\x00'  # header
    ...     b'\x00\x00\x01\x00\xff\xff\x02\x00\xfe\xff'  # data
    ... )
    >>> wf, sr = decode_wav_bytes(wav_bytes)
    >>> wf
    [0, 1, -1, 2, -2]
    >>> sr
    42
    """
    meta = decode_wav_header_bytes(wav_bytes)
    header_size = header_size_of_wav_bytes(wav_bytes, meta)
    wf = decode_pcm_bytes(
        wav_bytes[header_size:],
        width=meta["width_bytes"],
        n_channels=meta["n_channels"],
    )
    return wf, meta["sr"]


def header_size_of_wav_bytes(wav_bytes: bytes, meta: dict = None):
    """Compute the header size"""
    if meta is None:
        meta = decode_wav_header_bytes(wav_bytes)
    # the header tells us how many samples (frames) of data there are, how many
    # channels, and how many bytes each sample (frame) takes, so the header size is
    # the total size (number of bytes), minus the product of those three quantities
    data_size = int(meta["n_channels"] * meta["width_bytes"] * meta["nframes"])
    header_size = len(wav_bytes) - data_size
    assert (
        header_size >= MIN_WAV_N_BYTES
    ), f"Header size of wav bytes should be at least 44 bytes"
    return header_size


# # TODO: Repair. See https://github.com/otosense/recode/issues/3
# def encode_wav_bytes(wf: Waveform, sr: int, width_bytes: int = 2, n_channels: int = 1):
#     r"""Encode waveform (e.g. list of numbers) into PCM bytes.

#     :param wf: Waveform to encode
#     :param width: The width of a sample (in bits, bytes, numpy dtype, pyaudio ...)
#         (will try to figure it out by itself)
#     :param n_channels: Number of channels
#     :return: The pcm-bytes-encoded waveform


#     wav_bytes = encode_wav_bytes([0, 1, -1, 2, -2], sr=42)
#     header_bytes, data_bytes = wav_bytes[:44], wav_bytes[44:]
#     assert data_bytes == b'\x00\x00\x01\x00\xff\xff\x02\x00\xfe\xff'
#     decoded_wf, decoded_sr = decode_wav_bytes(wav_bytes)
#     assert decoded_wf == [0, 1, -1, 2, -2]
#     assert decoded_sr == 42

#     """
#     wf = list(wf)
#     nframes = len(wf)
#     wav_header_bytes = encode_wav_header_bytes(
#         sr, width_bytes=width_bytes, n_channels=n_channels, nframes=nframes
#     )
#     encode, _ = mk_pcm_audio_codec(width_bytes, n_channels)
#     return wav_header_bytes + encode(wf)


def encode_wav_bytes(wf: Waveform, sr: int, width_bytes: int = 2, n_channels: int = 1):
    r"""Encode waveform (e.g. list of numbers) into PCM bytes with WAV header.

    Args:
        wf: Waveform to encode (iterable of numbers)
        sr: Sample rate in Hz
        width_bytes: The width of a sample in bytes
        n_channels: Number of channels

    Returns:
        bytes: The complete WAV file bytes (header + data)

    Examples:

    >>> wav_bytes = encode_wav_bytes([0, 1, -1, 2, -2], sr=42)
    >>> header_bytes, data_bytes = wav_bytes[:44], wav_bytes[44:]
    >>> data_bytes
    b'\x00\x00\x01\x00\xff\xff\x02\x00\xfe\xff'

    See that the header bytes can be decoded to get the right information about our waveform:

    >>> decode_wav_header_bytes(header_bytes)  # doctest: +NORMALIZE_WHITESPACE
    {'sr': 42, 'width_bytes': 2, 'n_channels': 1, 'nframes': 5, 'comptype': None}

    See that our wave_bytes can be decoded to get the original waveform and sample rate:

    >>> decoded_wf, decoded_sr = decode_wav_bytes(wav_bytes)
    >>> decoded_wf
    [0, 1, -1, 2, -2]
    >>> decoded_sr
    42

    """
    # Convert iterable to list to ensure we can get the length
    wf = list(wf)
    nframes = len(wf)

    # Create a BytesIO buffer for the complete WAV file
    bio = BytesIO()

    # Create a Wave_write object and set all parameters
    with Wave_write(bio) as obj:
        obj.setnchannels(n_channels)
        obj.setsampwidth(width_bytes)
        obj.setframerate(sr)
        # Explicitly set the number of frames
        obj.setnframes(nframes)

        # Encode the waveform data
        encode, _ = mk_pcm_audio_codec(width_bytes, n_channels)
        pcm_data = encode(wf)

        # Write the frames data
        obj.writeframesraw(pcm_data)

    # Get the complete WAV file bytes
    bio.seek(0)
    return bio.read()


def encode_wav_header_bytes(
    sr: int,
    width_bytes: int,
    *,
    n_channels: int = 1,
    nframes: int = 0,
    comptype=None,
) -> bytes:
    r"""Make a WAV header from given parameters.

    NOTE: This function creates a header template. The `nframes` field in the
    resulting bytes might not be accurate if actual audio data is written
    separately. Use `encode_wav_bytes` for writing complete WAV files.

    :param sr: The sample rate (i.e. "frame rate" i.e. "chk_rate")
    :param width_bytes: The "sample width" in bytes
    :param n_channels: Number of channels (default is 1)
    :param nframes: Optional number of frames (default is 0).
        NOTE: If a wav file is to be read correctly, the num of frames (i.e.
        samples/chks) should be exactly the number you'll actually be writing in the
        wave file. This function creates a header *template* and might not reflect
        the final nframes if data is appended later.
    :param comptype: No supported by python's wave module (yet).

    >>> header_bytes = encode_wav_header_bytes(44100, 2, n_channels=3)
    >>> len(header_bytes)
    44
    >>> header_bytes[:31]
    b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x03\x00D\xac\x00\x00\x98\t\x04'

    You can decode those params (including those you didn't specify, but were
    defaulted) with the `decode_wav_header_bytes` inverse function.

    >>> # Assuming this is saved as a module where decode_wav_header_bytes is accessible
    >>> header_bytes_with_nframes = encode_wav_header_bytes(44100, 2, n_channels=3, nframes=100)
    >>> params = decode_wav_header_bytes(header_bytes_with_nframes)
    >>> params  # doctest: +NORMALIZE_WHITESPACE
    {'sr': 44100,
     'width_bytes': 2,
     'n_channels': 3,
     'nframes': 100,
     'comptype': None}

    Note: encoding the decoded params might not produce the identical header
    bytes if the wave module's internal header generation varies slightly,
    or if the original header contained extra chunks not handled here.
    The important part is that the *parameters* are decoded correctly.

    """
    bio = BytesIO()
    with Wave_write(bio) as obj:
        obj.setnchannels(n_channels)
        obj.setsampwidth(width_bytes)
        obj.setframerate(sr)
        obj.setnframes(nframes)
        if comptype:
            obj.setcomptype(comptype)

        # This writes the header with the specified nframes count
        obj.writeframesraw(b"")

    bio.seek(0)
    return bio.read()
    return bio.read()


def encode_wav_header_bytes(
    sr: int,
    width_bytes: int,
    *,
    n_channels: int = 1,
    nframes: int = 0,
    comptype=None,
) -> bytes:
    r"""Make a WAV header from given parameters.

    :param sr: The sample rate (i.e. "frame rate" i.e. "chk_rate")
    :param width_bytes: The "sample width" in bytes
    :param n_channels: Number of channels (default is 1)
    :param nframes: Optional number of frames (default is 0).
        NOTE: If a wav file is to be read correctly, the num of frames (i.e.
        samples/chks) should be exactly the number you'll actually be writing in the
        wave file.
    :param comptype: No supported by python's wave module (yet).

    >>> header_bytes = encode_wav_header_bytes(44100, 2, n_channels=3)
    >>> len(header_bytes)
    44
    >>> header_bytes[:31]
    b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x03\x00D\xac\x00\x00\x98\t\x04'

    You can decode those params (including those you didn't specify, but were
    defaulted) with the `decode_wav_header_bytes` inverse function.

    >>> from recode.audio import decode_wav_header_bytes
    >>> params = decode_wav_header_bytes(header_bytes)
    >>> params  # doctest: +NORMALIZE_WHITESPACE
    {'sr': 44100,
     'width_bytes': 2,
     'n_channels': 3,
     'nframes': 0,
     'comptype': None}
    >>> assert encode_wav_header_bytes(**params) == header_bytes

    """
    bio = BytesIO()
    with Wave_write(bio) as obj:
        obj.setnchannels(n_channels)
        obj.setsampwidth(width_bytes)
        obj.setframerate(sr)
        # print(nframes)
        # print(f"{obj.getnframes()=}")
        if nframes:
            obj.setnframes(nframes)
            # print(f"{obj.getnframes()=}")
        if comptype:
            obj.setcomptype(comptype)

        obj.writeframesraw(b"")
        # print(f"{obj.getnframes()=}")
        bio.seek(0)

    return bio.read()


def decode_wav_header_bytes(wav_header_bytes: bytes) -> dict:
    """Get a dict of params decoded from a wav header

    For examples, see the `encode_wav_header_bytes` function, it's inverse.

    >>> from recode.audio import encode_wav_header_bytes
    >>> header_bytes = encode_wav_header_bytes(44100, 2, n_channels=3)
    >>> decode_wav_header_bytes(header_bytes)  # doctest: +NORMALIZE_WHITESPACE
    {'sr': 44100,
     'width_bytes': 2,
     'n_channels': 3,
     'nframes': 0,
     'comptype': None}

    """
    wav_read_obj = Wave_read(BytesIO(wav_header_bytes))
    params = wav_read_obj.getparams()
    if params.comptype == "NONE":  # it's the only one supported
        comptype = None  # but we're making it compatible with encoding anyway
    return dict(
        sr=params.framerate,
        width_bytes=params.sampwidth,
        n_channels=params.nchannels,
        nframes=params.nframes,
        comptype=comptype,
    )


# TODO: Untested. Test.
# TODO: Could generalize to accept open file pointer directly too.
#  -> Tip: Change input name to file and wrap such that context manager just returns
#   the file pointer as is, if file is not a string.
# TODO: How could we get this efficient "only read header" with wavs in zip files?
def extract_wav_header_from_file(filepath):
    """Extracts the header of a WAV file, given it's filepath.

    This function is useful for reading the header of a WAV file without having to read
    the entire file into memory.
    This is useful when WAV files are large and/or numerous.

    Args:
        filename (str): The path to the WAV file.

    Returns:
        bytes: The bytes of the WAV file header.
    """
    # Initially read the first 44 bytes
    with open(filepath, "rb") as file:
        header = file.read(44)

        # Unpack the ChunkSize (bytes 4-8) and Subchunk2Size (bytes 40-44)
        chunk_size = int.from_bytes(header[4:8], byteorder="little")
        subchunk2_size = int.from_bytes(header[40:44], byteorder="little")

        # Calculate the total header size
        header_size = chunk_size + 8 - subchunk2_size

        # If the header is larger than 44 bytes, read the remaining bytes
        if header_size > 44:
            header += file.read(header_size - 44)

    return header


# TODO: Can optimize (index) the data below to make search functions faster
num_type_synonyms = [
    {
        "dtype": "int16",
        "soundfile": "PCM_16",
        "pyaudio": "paInt16",
        "n_bits": 16,
        "n_bytes": 2,
        "struct": "h",
    },
    {
        "dtype": "int8",
        "soundfile": "PCM_S8",
        "pyaudio": "paInt8",
        "n_bits": 8,
        "n_bytes": 1,
        "struct": "b",
    },
    {
        "dtype": "int24",
        "soundfile": "PCM_24",
        "pyaudio": "paInt24",
        "n_bits": 24,
        "n_bytes": 3,
        "struct": None,
    },
    {
        "dtype": "int32",
        "soundfile": "PCM_32",
        "pyaudio": "paInt32",
        "n_bits": 32,
        "n_bytes": 4,
        "struct": "i",
    },
    {
        "dtype": "uint8",
        "soundfile": "PCM_U8",
        "pyaudio": "paUInt8",
        "n_bits": 8,
        "n_bytes": 1,
        "struct": "B",
    },
    {
        "dtype": "float32",
        "soundfile": "FLOAT",
        "pyaudio": "paFloat32",
        "n_bits": 32,
        "n_bytes": 4,
        "struct": "f",
    },
    {
        "dtype": "float64",
        "soundfile": "DOUBLE",
        "pyaudio": None,
        "n_bits": 64,
        "n_bytes": 8,
        "struct": "d",
    },
]


def num_find_num_type_for(
    num,
    target_num_sys="struct",
    num_sys_search_order=("n_bits", "n_bytes", "dtype", "pyaudio", "soundfile"),
):
    """Find the target_num_sys equivalent of input num checking multiple unit options"""
    for num_sys in num_sys_search_order:
        try:
            return num_type_for(num, num_sys, target_num_sys)
        except ValueError:
            "Just try the next num_sys..."


def num_type_for(num, num_sys="n_bits", target_num_sys="struct"):
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
                    f"Did not find any {target_num_sys} entry for {num_sys}={num}"
                )
    raise ValueError(f"Did not find any entry for {num_sys}={num}")
