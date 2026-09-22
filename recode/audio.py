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

import struct
import warnings
import wave
from io import BytesIO
from typing import Union
from collections.abc import Iterable
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


_RIFF_HEADER_SIZE = 12  # 'RIFF' + form size + 'WAVE'
_CHUNK_HEADER_SIZE = 8  # chunk id + chunk size
_CHUNK_ID_SIZE = 4
_PCM_FMT_CHUNK_SIZE = 16  # the plain (non-extensible) `fmt ` chunk contents

# The smallest a WAV file can be: the RIFF/WAVE preamble, a plain PCM `fmt ` chunk, and
# an empty `data` chunk -- 44 bytes. A long-standing public name of this module.
MIN_WAV_N_BYTES = (
    _RIFF_HEADER_SIZE + _CHUNK_HEADER_SIZE + _PCM_FMT_CHUNK_SIZE + _CHUNK_HEADER_SIZE
)

# How much of a file `extract_wav_header_from_file` reads at a time while looking for
# the audio. Comfortably more than any header in practice, so it is normally one read.
DFLT_HEADER_READ_SIZE = 64 * 1024

# WAV stores 8-bit PCM *unsigned* (0..255, silence at 128) and every wider width signed.
_UINT8_BIAS = 128
_UINT8_WIDTH_NAME = "uint8"

_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_EXTENSIBLE_FMT_CHUNK_SIZE = 40
_SUBFORMAT_GUID_OFFSET = 24  # within the `fmt ` chunk contents
# The SubFormat GUID that says an EXTENSIBLE chunk really does carry plain PCM.
_PCM_SUBFORMAT_GUID = bytes.fromhex("0100000000001000800000aa00389b71")


class ShortWavData(UserWarning):
    """The `data` chunk carries fewer bytes than its own header declares.

    Raised as a warning rather than an error because the audio that *is* present is
    still worth decoding -- a partially downloaded file, or one written to a stream
    whose length was never patched back into the header. What must not happen is for
    the shortfall to pass unmentioned, since the caller cannot otherwise tell a
    truncated file from a complete one.
    """


def _shift_samples(frames, by: int):
    """Add `by` to every sample, keeping the mono (flat) or multichannel (grouped) shape.

    8-bit is the one WAV width where the bytes on disk (unsigned, 0..255, silence at
    128) and the samples a caller works with (signed) differ by a constant, so the shift
    is applied around the PCM codec rather than inside it -- leaving `mk_pcm_audio_codec`
    and the generic `encode_pcm_bytes`/`decode_pcm_bytes` pair untouched.

    >>> _shift_samples([0, 128, 255], -128)
    [-128, 0, 127]
    >>> _shift_samples([(0, 255), (128, 64)], -128)
    [(-128, 127), (0, -64)]
    """
    return [
        (
            tuple(sample + by for sample in frame)
            if isinstance(frame, Iterable)
            else frame + by
        )
        for frame in frames
    ]


def decode_wav_bytes(wav_bytes: bytes, *, eight_bit_unsigned: bool = True):
    r"""Decode WAV bytes into a ``(waveform, sample_rate)`` pair.

    :param wav_bytes: The bytes of a RIFF/WAVE container holding uncompressed PCM
    :param eight_bit_unsigned: Read 8-bit audio as the unsigned PCM the WAV spec
        mandates (0..255 on disk, biased to -128..127 here). Pass `False` for the
        pre-recode#12 behaviour, which read those bytes as signed -- silence came back
        as -128 -- and so round-tripped with `encode_wav_bytes` but with nothing else.
        Widths of 9 bits and up are signed in the spec and are unaffected either way.
    :return: ``(wf, sr)`` -- the decoded waveform and its sample rate

    :raises ValueError: if `wav_bytes` is not a RIFF/WAVE container with a `data`
        chunk. (Before recode#4 the same inputs raised `AssertionError`, `wave.Error`
        or `EOFError` depending on how they were malformed; they are unified here.)
    :raises ShortWavData: *warning*, not an exception -- see below.

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

    The `data` chunk is located by walking the RIFF structure, so chunks that sit
    *after* the audio -- `LIST`/`INFO` metadata, which ffmpeg, Audacity and iTunes
    all append -- do not shift the waveform:

    >>> import struct
    >>> info = b'INFOISFT' + struct.pack('<I', 6) + b'Lavf58'
    >>> with_trailing_metadata = wav_bytes + b'LIST' + struct.pack('<I', len(info)) + info
    >>> decode_wav_bytes(with_trailing_metadata)[0]
    [0, 1, -1, 2, -2]

    A file carrying less audio than its header declares decodes to the whole frames
    that are actually there, and says so:

    >>> truncated = wav_bytes[:-4]
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     _ = warnings.simplefilter('always')
    ...     wf, sr = decode_wav_bytes(truncated)
    >>> wf
    [0, 1, -1]
    >>> caught[0].category.__name__
    'ShortWavData'

    8-bit WAV audio is stored *unsigned*, so a byte of 128 is silence rather than full
    negative (recode#12). Pass `eight_bit_unsigned=False` to get the old signed reading:

    >>> eight_bit = encode_wav_bytes([-128, 0, 127], sr=42, width_bytes=1)
    >>> eight_bit[44:]
    b'\x00\x80\xff'
    >>> decode_wav_bytes(eight_bit)[0]
    [-128, 0, 127]
    >>> decode_wav_bytes(eight_bit, eight_bit_unsigned=False)[0]
    [0, -128, -1]
    """
    offset, size = _wav_data_chunk(wav_bytes)
    meta = decode_wav_header_bytes(wav_bytes)
    # Decoding is defined on whole frames, and a frame is every channel's sample: a
    # file cut mid-frame drops the remainder rather than raising a struct-size error.
    frame_size = int(meta["n_channels"] * meta["width_bytes"])
    if frame_size:
        size -= size % frame_size
        if size // frame_size < meta["nframes"]:
            warnings.warn(
                f"WAV `data` chunk is short: the header declares {meta['nframes']} "
                f"frames, {size // frame_size} are present. Decoding what is there.",
                ShortWavData,
                stacklevel=2,
            )
    if size == 0:
        # No whole frame survived. Answering with an empty waveform is the consistent
        # reading of "decode the frames that are present"; letting it through would
        # surface as an IndexError from inside the chunked decoder instead.
        return [], meta["sr"]
    payload = wav_bytes[offset : offset + size]
    if eight_bit_unsigned and meta["width_bytes"] == 1:
        wf = _shift_samples(
            decode_pcm_bytes(
                payload, width=_UINT8_WIDTH_NAME, n_channels=meta["n_channels"]
            ),
            -_UINT8_BIAS,
        )
    else:
        wf = decode_pcm_bytes(
            payload, width=meta["width_bytes"], n_channels=meta["n_channels"]
        )
    return wf, meta["sr"]


def _wav_data_chunk(wav_bytes: bytes) -> tuple:
    r"""Locate the audio payload: ``(offset, size)`` of the `data` chunk's contents.

    Walks the RIFF chunk list rather than inferring the position arithmetically, which
    is what makes it robust to the shapes real-world WAV files take that a
    size-subtraction cannot survive (recode#4):

    - **chunks after `data`.** `LIST`/`INFO` tags are routinely appended by encoders.
      Deriving the header size as ``len(wav_bytes) - n_channels * width * nframes``
      silently counts those trailing bytes as header, so the waveform is read from too
      far in -- returning audio of the right *length* and the wrong *content*, with no
      error raised.
    - **an over-declared `data` size.** Files written to a stream (length unknown at
      write time, patched afterwards -- or never) declare more samples than they carry,
      commonly with the sentinel ``0xFFFFFFFF``. The size is clamped to what is really
      there instead of asserting.

    Those two interact, and naively clamping an over-declared size to end-of-file
    would re-create the very bug this function exists to kill -- a trailing `LIST`
    would be handed back as audio. So when the declared size overruns, the payload is
    bounded by the next chunk that the remainder of the file parses cleanly from,
    rather than by EOF. Only when no such boundary exists does it fall back to EOF.

    The walk itself is :func:`_wav_chunk`; what this adds is the reconciliation of the
    declared size with the bytes actually there.

    >>> import struct, wave, io
    >>> b = io.BytesIO()
    >>> with wave.open(b, 'wb') as w:
    ...     _ = w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000)
    ...     w.writeframes(struct.pack('<3h', 1, 2, 3))
    >>> raw = b.getvalue()
    >>> offset, size = _wav_data_chunk(raw)
    >>> size
    6
    >>> raw[offset:offset + size] == struct.pack('<3h', 1, 2, 3)
    True

    An over-declared size does not swallow what follows the audio:

    >>> broken = bytearray(raw)
    >>> at = broken.find(b'data')
    >>> broken[at + 4:at + 8] = struct.pack('<I', 0xFFFFFFFF)  # stream sentinel
    >>> trailing = b'LIST' + struct.pack('<I', 4) + b'INFO'
    >>> _wav_data_chunk(bytes(broken) + trailing)
    (44, 6)
    """
    contents, declared = _wav_chunk(wav_bytes, b"data")
    if declared <= len(wav_bytes) - contents:
        return contents, declared
    return contents, _payload_end_of_overrunning_data(wav_bytes, contents)


def _check_riff_wave(wav_bytes: bytes) -> None:
    """Raise `ValueError` unless the buffer opens with a RIFF/WAVE preamble."""
    if (
        len(wav_bytes) < _RIFF_HEADER_SIZE
        or wav_bytes[:4] != b"RIFF"
        or wav_bytes[8:12] != b"WAVE"
    ):
        raise ValueError(
            "Not WAV bytes: expected a RIFF/WAVE container, got "
            f"{bytes(wav_bytes[:4])!r}...{bytes(wav_bytes[8:12])!r}"
        )


def _wav_chunk(wav_bytes: bytes, wanted: bytes = b"data") -> tuple:
    r"""Walk the RIFF chunk list to `wanted`: ``(contents_offset, declared_size)``.

    The size is the one the chunk's own header declares, unexamined -- `data` chunks lie
    about it often enough that :func:`_wav_data_chunk` exists to reconcile the claim
    with the buffer, and :func:`_extensible_wav_header_params` wants the declared number
    precisely because that is what stdlib `wave` counts frames from.

    Sizes are read as unsigned little-endian, per the RIFF spec.

    >>> raw = (
    ...     b'RIFF.\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'
    ...     b'*\x00\x00\x00T\x00\x00\x00\x02\x00\x10\x00data\n\x00\x00\x00'
    ...     b'\x00\x00\x01\x00\xff\xff\x02\x00\xfe\xff'
    ... )
    >>> _wav_chunk(raw, b'fmt ')
    (20, 16)
    >>> _wav_chunk(raw, b'data')
    (44, 10)
    """
    _check_riff_wave(wav_bytes)
    label = wanted.decode("ascii", "replace")
    pos = _RIFF_HEADER_SIZE
    while pos + _CHUNK_HEADER_SIZE <= len(wav_bytes):
        chunk_id = bytes(wav_bytes[pos : pos + _CHUNK_ID_SIZE])
        (declared,) = struct.unpack(
            "<I", wav_bytes[pos + _CHUNK_ID_SIZE : pos + _CHUNK_HEADER_SIZE]
        )
        contents = pos + _CHUNK_HEADER_SIZE
        if chunk_id == wanted:
            return contents, declared
        # RIFF chunks are word-aligned: an odd-sized chunk carries a pad byte.
        pos = contents + declared + (declared % 2)
        if declared > len(wav_bytes) - contents:
            raise ValueError(
                f"Not WAV bytes: chunk walk desynced at offset {pos - declared - 8} "
                f"({chunk_id!r} declares {declared} bytes but only "
                f"{len(wav_bytes) - contents} remain); no `{label}` chunk reachable"
            )
    raise ValueError(f"Not WAV bytes: no `{label}` chunk found")


def _payload_end_of_overrunning_data(wav_bytes: bytes, contents: int) -> int:
    """How many bytes of audio a `data` chunk whose declared size overruns really has.

    The declared size is unusable, so the extent has to come from the file itself: the
    audio runs until the next thing that is demonstrably a chunk, meaning a position
    (word-aligned, as RIFF requires) from which the rest of the buffer parses as a
    well-formed chunk list ending exactly at EOF. Requiring the parse to reach EOF is
    what keeps this from firing on audio that merely happens to contain four
    plausible-looking bytes.

    Falls back to end-of-file when no such position exists -- a genuinely truncated
    file, where reading to the end is right.
    """
    end = len(wav_bytes)
    start = contents + (contents % 2)
    for candidate in range(start, end - _CHUNK_HEADER_SIZE + 1, 2):
        if _parses_as_chunk_list_to_eof(wav_bytes, candidate):
            return candidate - contents
    return end - contents


def _parses_as_chunk_list_to_eof(wav_bytes: bytes, pos: int) -> bool:
    """Does the buffer from `pos` read as a chunk list that lands exactly on EOF?"""
    end = len(wav_bytes)
    while pos < end:
        if pos + _CHUNK_HEADER_SIZE > end:
            return False
        chunk_id = wav_bytes[pos : pos + _CHUNK_ID_SIZE]
        # A chunk id is four printable ASCII characters; anything else is audio.
        if not all(0x20 <= b < 0x7F for b in chunk_id):
            return False
        (declared,) = struct.unpack(
            "<I", wav_bytes[pos + _CHUNK_ID_SIZE : pos + _CHUNK_HEADER_SIZE]
        )
        pos += _CHUNK_HEADER_SIZE + declared + (declared % 2)
    return pos == end


def header_size_of_wav_bytes(wav_bytes: bytes, meta: dict = None) -> int:
    r"""Size, in bytes, of everything preceding the audio payload.

    That is the offset of the `data` chunk's contents, found by walking the RIFF
    structure (see :func:`_wav_data_chunk`). For a well-formed file with nothing after
    the audio this is the same number the old size-subtraction produced; unlike it, it
    stays correct when the file carries trailing metadata or an over-declared `data`
    size.

    `meta` -- an already-decoded header, once passed to save re-parsing it -- is
    accepted for backwards compatibility and no longer used.

    >>> header_size_of_wav_bytes(
    ...     b'RIFF.\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'
    ...     b'*\x00\x00\x00T\x00\x00\x00\x02\x00\x10\x00data\n\x00\x00\x00'
    ...     b'\x00\x00\x01\x00\xff\xff\x02\x00\xfe\xff'
    ... )
    44
    """
    offset, _ = _wav_data_chunk(wav_bytes)
    return offset


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


def encode_wav_bytes(
    wf: Waveform,
    sr: int,
    width_bytes: int = 2,
    n_channels: int = 1,
    *,
    eight_bit_unsigned: bool = True,
):
    r"""Encode waveform (e.g. list of numbers) into PCM bytes with WAV header.

    Args:
        wf: Waveform to encode (iterable of numbers)
        sr: Sample rate in Hz
        width_bytes: The width of a sample in bytes
        n_channels: Number of channels
        eight_bit_unsigned: Write 8-bit audio as the unsigned PCM the WAV spec
            mandates (samples biased by 128 into 0..255 on disk). Pass `False` for the
            pre-recode#12 behaviour, which wrote them signed -- files that recode read
            back correctly and every other tool did not. Widths of 9 bits and up are
            signed in the spec and are unaffected either way.

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

        # Encode the waveform data, biasing 8-bit samples into the unsigned range the
        # WAV spec reserves for that one width (see `_shift_samples`).
        unsigned8 = eight_bit_unsigned and width_bytes == 1
        encode, _ = mk_pcm_audio_codec(
            _UINT8_WIDTH_NAME if unsigned8 else width_bytes, n_channels
        )
        pcm_data = encode(_shift_samples(wf, _UINT8_BIAS) if unsigned8 else wf)

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


def _extensible_wav_header_params(wav_bytes: bytes):
    """Params of a `WAVE_FORMAT_EXTENSIBLE` header, or `None` if it is not one.

    Stdlib `wave` only learned to read the 0xFFFE (`WAVE_FORMAT_EXTENSIBLE`) format tag
    in Python 3.12, so on 3.10 and 3.11 -- 3.10 being the version CI runs -- an ordinary
    ffmpeg file raises `wave.Error: unknown format: 65534` (recode#13). Anything above
    two channels or sixteen bits tends to carry the tag. The chunk is the plain PCM
    `fmt ` layout with `cbSize`, `wValidBitsPerSample`, `dwChannelMask` and a SubFormat
    GUID appended, so every field `getparams()` would have reported is right there, and
    the frame count comes from the `data` chunk's *declared* size, exactly as `Wave_read`
    computes it.

    Returns `None` -- rather than raising -- for anything it does not recognise, so the
    caller can re-raise what `wave` itself said. That matters: `wave.Error` also covers
    "not a WAVE file" and "fmt chunk missing", and those must keep failing as before.

    Only the PCM SubFormat GUID is accepted -- which is exactly what 3.12's own
    `wave._read_fmt_chunk` does (`KSDATAFORMAT_SUBTYPE_PCM`, or `Error('unknown extended
    format: ...')`). So a compressed EXTENSIBLE payload keeps being refused on every
    version, and the versions agree on what they accept as well as on what they read.
    """
    try:
        fmt_at, fmt_size = _wav_chunk(wav_bytes, b"fmt ")
        _, data_size = _wav_chunk(wav_bytes, b"data")
    except ValueError:
        return None
    fmt = wav_bytes[fmt_at : fmt_at + _EXTENSIBLE_FMT_CHUNK_SIZE]
    if fmt_size < _EXTENSIBLE_FMT_CHUNK_SIZE or len(fmt) < _EXTENSIBLE_FMT_CHUNK_SIZE:
        return None
    fmt_tag, n_channels, sr, _byte_rate, _block_align, n_bits = struct.unpack(
        "<HHIIHH", fmt[:_PCM_FMT_CHUNK_SIZE]
    )
    if (
        fmt_tag != _WAVE_FORMAT_EXTENSIBLE
        or fmt[_SUBFORMAT_GUID_OFFSET:] != _PCM_SUBFORMAT_GUID
    ):
        return None
    width_bytes = (n_bits + 7) // 8  # how `wave` itself rounds a bit depth to bytes
    frame_size = n_channels * width_bytes
    if not frame_size:
        return None
    return dict(
        sr=sr,
        width_bytes=width_bytes,
        n_channels=n_channels,
        nframes=data_size // frame_size,
        comptype=None,
    )


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

    Stdlib `wave` is the primary reader. A `WAVE_FORMAT_EXTENSIBLE` header, which it
    refuses before Python 3.12, is parsed directly instead so the same file decodes on
    every supported version -- see :func:`_extensible_wav_header_params`.
    """
    try:
        params = Wave_read(BytesIO(wav_header_bytes)).getparams()
    except wave.Error:
        extensible = _extensible_wav_header_params(wav_header_bytes)
        if extensible is None:
            raise  # not the one case we can read; `wave`'s own complaint stands
        return extensible
    # Normalized to None so the value round-trips with `encode_wav_header_bytes`.
    # Unconditional on purpose: `Wave_read` rejects any non-PCM fmt tag with
    # `wave.Error` before `getparams()` returns, so comptype is always 'NONE' here.
    # It used to be assigned inside `if params.comptype == "NONE"`, an unreachable
    # guard that would have left the name unbound if that ever changed.
    comptype = None
    return dict(
        sr=params.framerate,
        width_bytes=params.sampwidth,
        n_channels=params.nchannels,
        nframes=params.nframes,
        comptype=comptype,
    )


# TODO: Could generalize to accept open file pointer directly too.
#  -> Tip: Change input name to file and wrap such that context manager just returns
#   the file pointer as is, if file is not a string.
# TODO: How could we get this efficient "only read header" with wavs in zip files?
def extract_wav_header_from_file(filepath, *, read_size: int = DFLT_HEADER_READ_SIZE):
    """Extract the header of a WAV file -- everything before the audio -- from its path.

    Useful for reading the header of a WAV file without having to read the entire file
    into memory, which is what you want when WAV files are large and/or numerous: only
    as much of the file as the header occupies is read.

    The answer is the same one :func:`header_size_of_wav_bytes` gives for the same
    bytes, because both locate the `data` chunk by walking the RIFF structure rather
    than inferring where it must be. This used to compute
    ``chunk_size + 8 - subchunk2_size``, reading bytes 40-44 as the size of the audio --
    true only of a bare 44-byte header. With a `LIST`/`INFO` chunk after the audio (what
    ffmpeg, Audacity and iTunes write) it returned audio bytes as header; with one
    before, a truncated prefix that parses as nothing at all (recode#4).

    :param filepath: The path to the WAV file
    :param read_size: How many bytes to read at a time while looking for the audio
    :return: The bytes of the WAV file header, i.e. everything preceding the `data`
        chunk's contents
    :raises ValueError: if the file is not a RIFF/WAVE container with a reachable `data`
        chunk. (It used to answer such files with arbitrary bytes and no complaint.)
    """
    with open(filepath, "rb") as file:
        # At least the preamble, so a non-WAV is refused without reading it to its end.
        prefix = file.read(max(read_size, _RIFF_HEADER_SIZE))
        _check_riff_wave(prefix)
        while True:
            try:
                offset, _ = _wav_chunk(prefix, b"data")
            except ValueError:
                more = file.read(read_size)
                if not more:
                    raise  # the whole file is in hand; the complaint is the final one
                prefix += more
            else:
                return prefix[:offset]


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
