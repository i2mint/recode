from recode import (
    ChunkedEncoder,
    MetaEncoder,
    ChunkedDecoder,
    IterativeDecoder,
    MetaDecoder,
    frame_to_meta,
    meta_to_frame,
    StructCodecSpecs,
    specs_from_frames,
    decode_wav_bytes,
)
import pytest
from collections.abc import Iterator


@pytest.mark.parametrize(
    "chk_format,frame,n_channels",
    [
        ("h", [1, 2, 3, 4, 5], None),
        ("h", [1, 2, 3, 4, 5], 1),
        ("h", [1], None),
        ("h", [1], 1),
        ("d", [1.1, 2.2, 3.3], None),
        ("d", [1.1, 2.2, 3.3], 1),
        ("d", [1.1], None),
        ("d", [1.1], 1),
        ("=h", [1, 2, 3, 4, 5], None),
        ("<h", [1, 2, 3, 4, 5], None),
        (">h", [1, 2, 3, 4, 5], None),
        ("!h", [1, 2, 3, 4, 5], None),
    ],
)
def test_single_channel_chunk(chk_format, frame, n_channels):
    n_channels = n_channels or 1
    specs = StructCodecSpecs(chk_format=chk_format * n_channels)
    encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    b = encoder(frame)
    assert isinstance(b, bytes)
    decoded_frames = decoder(b)
    assert decoded_frames == frame


@pytest.mark.parametrize(
    "chk_format,frame,n_channels",
    [
        ("hh", [[1, 1]], None),
        ("h", [[1, 1]], 2),
        ("hh", [[1, 1], [2, 2]], None),
        ("h", [[1, 1], [2, 2]], 2),
        ("dd", [[1.1, 1.1]], None),
        ("d", [[1.1, 1.1]], 2),
        ("dd", [[1.1, 1.1], [2.2, 2.2]], None),
        ("d", [[1.1, 1.1], [2.2, 2.2]], 2),
        (
            "hhhhhhhhh",
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            None,
        ),
        (
            "h",
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            9,
        ),
        ("=hh", [[1, 2]], None),
        (">hh", [[1, 2]], None),
        ("<hh", [[1, 2]], None),
        ("!hh", [[1, 2]], None),
    ],
)
def test_multi_channel_chunk(chk_format, frame, n_channels):
    n_channels = n_channels or 1
    specs = StructCodecSpecs(chk_format=chk_format * n_channels)
    encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    b = encoder(frame)
    assert isinstance(b, bytes)
    decoded_frames = list(map(list, decoder(b)))
    assert decoded_frames == frame


@pytest.mark.parametrize(
    "chk_format,frame,n_channels",
    [
        ("h", [1, 2, 3, 4, 5], None),
        ("h", [1, 2, 3, 4, 5], 1),
        ("h", [1], None),
        ("h", [1], 1),
        ("d", [1.1, 2.2, 3.3], None),
        ("d", [1.1, 2.2, 3.3], 1),
        ("d", [1.1], None),
        ("d", [1.1], 1),
        ("=h", [1, 2, 3, 4, 5], None),
        ("<h", [1, 2, 3, 4, 5], None),
        (">h", [1, 2, 3, 4, 5], None),
        ("!h", [1, 2, 3, 4, 5], None),
    ],
)
def test_single_channel_iter(chk_format, frame, n_channels):
    specs = StructCodecSpecs(chk_format=chk_format, n_channels=n_channels)
    encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    decoder = IterativeDecoder(chk_to_frame=specs.chk_to_frame)
    b = encoder(frame)
    assert isinstance(b, bytes)
    decoded_frames = decoder(b)
    assert isinstance(decoded_frames, Iterator)
    assert next(decoded_frames)[0] == frame[0]


@pytest.mark.parametrize(
    "chk_format,frame,n_channels",
    [
        ("hh", [[1, 1]], None),
        ("h", [[1, 1]], 2),
        ("hh", [[1, 1], [2, 2]], None),
        ("h", [[1, 1], [2, 2]], 2),
        ("dd", [[1.1, 1.1]], None),
        ("d", [[1.1, 1.1]], 2),
        ("dd", [[1.1, 1.1], [2.2, 2.2]], None),
        ("d", [[1.1, 1.1], [2.2, 2.2]], 2),
        (
            "hhhhhhhhh",
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            None,
        ),
        (
            "h",
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            9,
        ),
        ("=hh", [[1, 2]], None),
        (">hh", [[1, 2]], None),
        ("<hh", [[1, 2]], None),
        ("!hh", [[1, 2]], None),
    ],
)
def test_multi_channel_iter(chk_format, frame, n_channels):
    n_channels = n_channels or 1
    specs = StructCodecSpecs(chk_format=chk_format * n_channels)
    encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    decoder = IterativeDecoder(chk_to_frame=specs.chk_to_frame)
    b = encoder(frame)
    assert isinstance(b, bytes)
    decoded_frames = decoder(b)
    assert isinstance(decoded_frames, Iterator)
    assert list(next(decoded_frames)) == frame[0]


@pytest.mark.parametrize(
    "chk_format,table,n_channels",
    [
        ("h", [{"a": 1}], 1),
        ("h", [{"a": 1}, {"a": 2}], 1),
        ("d", [{"a": 1.1}], 1),
        ("d", [{"a": 1.1}, {"a": 2.2}], 1),
        ("hh", [{"a": 1, "b": 2}], None),
        ("hh", [{"a": 1, "b": 2}, {"a": 3, "b": 4}], None),
        ("dd", [{"a": 1.1, "b": 2.2}], None),
        ("dd", [{"a": 1.1, "b": 2.2}, {"a": 3.3, "b": 4.4}], None),
    ],
)
def test_tabular(chk_format, table, n_channels):
    n_channels = n_channels or 1
    specs = StructCodecSpecs(chk_format=chk_format * n_channels)
    encoder = MetaEncoder(frame_to_chk=specs.frame_to_chk, frame_to_meta=frame_to_meta)
    decoder = MetaDecoder(chk_to_frame=specs.chk_to_frame, meta_to_frame=meta_to_frame)
    b = encoder(table)
    assert isinstance(b, bytes)
    decoded_frames = decoder(b)
    assert decoded_frames == table


@pytest.mark.parametrize(
    "chk_format,frame,n_channels",
    [
        ("h", [1, 2], 1),
        ("hh", [[1, 1], [2, 2]], 2),
        ("h", [1], 1),
        ("hh", [[1, 1]], 2),
        ("d", [1.1, 2.2], 1),
        ("dd", [[1.1, 1.1], [2.2, 2.2]], 2),
        ("d", [1.1], 1),
        ("dd", [[1.1, 1.1]], 2),
        ("h", iter([1, 2]), 1),
        ("hh", iter([[1, 1], [2, 2]]), 2),
        ("h", iter([1]), 1),
        ("hh", iter([[1, 1]]), 2),
        ("d", iter([1.1, 2.2]), 1),
        ("dd", iter([[1.1, 1.1], [2.2, 2.2]]), 2),
        ("d", iter([1.1]), 1),
        ("dd", iter([[1.1, 1.1]]), 2),
        ("h", [{"a": 1}], 1),
        ("h", [{"a": 1}, {"a": 2}], 1),
        ("d", [{"a": 1.1}], 1),
        ("d", [{"a": 1.1}, {"a": 2.2}], 1),
        ("hh", [{"a": 1, "b": 2}], 2),
        ("hh", [{"a": 1, "b": 2}, {"a": 3, "b": 4}], 2),
        ("dd", [{"a": 1.1, "b": 2.2}], 2),
        ("dd", [{"a": 1.1, "b": 2.2}, {"a": 3.3, "b": 4.4}], 2),
    ],
)
def test_implicit_specs(chk_format, frame, n_channels):
    _, specs = specs_from_frames(frame)
    assert specs.chk_format == chk_format
    assert specs.n_channels == n_channels


# ------ testing wav related things -----------------------------------------------------


def wf_to_wav_bytes_with_soundfile(wf, sr=44100, dtype="int16", subtype="PCM_16"):
    """Just used to MAKE some test data (uses non-builtins)"""
    import soundfile as sf
    import io
    import numpy as np

    b = io.BytesIO()
    sf.write(
        b, np.array(wf).astype(dtype), samplerate=sr, format="wav", subtype=subtype
    )
    b.seek(0)
    return b.read()


# Was used to make a little case for decode_wav_bytes doctest
# little_wf = [0, 1, -1, 2, -2]
# little_wf_wav_bytes = wf_to_wav_bytes_with_soundfile(little_wf)
from itertools import chain

big_wf = [0] + list(chain.from_iterable([x, -x] for x in range(32768)))
big_wf_bytes_file = "test_wf.wav"


def mk_test_wav_file():
    """To make the test_wf.wav test file (uses non-builtins)"""
    big_wf_wav_bytes = wf_to_wav_bytes_with_soundfile(big_wf)

    with open(big_wf_bytes_file, "wb") as fp:
        fp.write(big_wf_wav_bytes)


def test_decode_wav_bytes():
    with open(big_wf_bytes_file, "rb") as fp:
        b = fp.read()

    wf, sr = decode_wav_bytes(b)
    assert list(wf) == list(big_wf)
    assert sr == 44100


# mk_test_wav_file()


# ------ recode#4: the `data` chunk must be located, not inferred -----------------------
#
# `decode_wav_bytes` used to derive the header size by subtraction:
#
#     header_size = len(wav_bytes) - n_channels * width_bytes * nframes
#
# which reads any bytes that follow the audio as though they were header. The failure is
# silent -- the waveform comes back the right LENGTH and the wrong CONTENT -- so these
# tests assert decoded samples, never just a length or a "did not raise".
#
# Each case below is a real shape a WAV file takes in the wild, and each one is read
# correctly by `soundfile`, which is how the original report noticed the discrepancy.

import io
import struct
import warnings
import wave as _wave

from recode.audio import ShortWavData

_WF = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5]
_SR = 44100


_STRUCT_CODE_FOR_WIDTH = {1: "b", 2: "h", 4: "i"}


def _wav(wf=_WF, sr=_SR, n_channels=1, width=2):
    """A minimal, well-formed PCM WAV, built with the stdlib only.

    `width` drives the packing as well as the header, so a non-default width yields a
    valid file rather than a header that disagrees with its own payload.
    """
    code = _STRUCT_CODE_FOR_WIDTH[width]
    b = io.BytesIO()
    with _wave.open(b, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<" + code, x) for x in wf))
    return b.getvalue()


def _with_riff_size_fixed(raw):
    raw = bytearray(raw)
    raw[4:8] = struct.pack("<I", len(raw) - 8)
    return bytes(raw)


def _insert_chunk_before_data(raw, chunk):
    at = raw.find(b"data")
    return _with_riff_size_fixed(raw[:at] + chunk + raw[at:])


def test_decode_wav_bytes_ignores_metadata_after_the_audio():
    """A `LIST`/`INFO` chunk after `data` -- what ffmpeg, Audacity and iTunes append.

    This is the regression the issue was filed for. Before the fix this returned
    `[0, 20041, 20294, ...]`: same number of samples, different audio, no exception.
    """
    info = b"INFOISFT" + struct.pack("<I", 6) + b"Lavf58"
    raw = _with_riff_size_fixed(_wav() + b"LIST" + struct.pack("<I", len(info)) + info)
    assert decode_wav_bytes(raw) == (_WF, _SR)


@pytest.mark.parametrize("over_declare_by", [200, 0xFFFFFFFF - 20])
def test_decode_wav_bytes_survives_an_over_declared_data_size(over_declare_by):
    """Files written to a stream declare a length they may never go back and patch.

    Before the fix this tripped `assert header_size >= 44` and raised. `0xFFFFFFFF` is
    the sentinel a stream writer leaves behind when it never learns the length.
    """
    raw = bytearray(_wav())
    at = raw.find(b"data")
    (declared,) = struct.unpack("<I", bytes(raw[at + 4 : at + 8]))
    raw[at + 4 : at + 8] = struct.pack(
        "<I", min(declared + over_declare_by, 0xFFFFFFFF)
    )
    with pytest.warns(ShortWavData):
        assert decode_wav_bytes(bytes(raw)) == (_WF, _SR)


def test_over_declared_data_does_not_swallow_a_trailing_chunk():
    """The two malformations combined -- where a naive clamp-to-EOF re-creates the bug.

    An over-declared `data` size says "read to the end"; a trailing `LIST` says "the
    end is not audio". Getting this wrong returns the metadata bytes as extra samples,
    silently, which is the exact failure class this whole change exists to remove.
    """
    info = b"INFOISFT" + struct.pack("<I", 6) + b"Lavf58"
    trailing = b"LIST" + struct.pack("<I", len(info)) + info
    raw = bytearray(_wav())
    at = raw.find(b"data")
    raw[at + 4 : at + 8] = struct.pack("<I", 0xFFFFFFFF)  # stream sentinel
    raw = _with_riff_size_fixed(bytes(raw) + trailing)

    from recode.audio import _wav_data_chunk

    assert _wav_data_chunk(raw) == (44, len(_WF) * 2)
    with pytest.warns(ShortWavData):
        assert decode_wav_bytes(raw) == (_WF, _SR)


def test_over_declared_data_falls_back_to_eof_when_nothing_follows():
    """No trailing chunk to find, so end-of-file is the right boundary."""
    raw = bytearray(_wav())
    at = raw.find(b"data")
    raw[at + 4 : at + 8] = struct.pack("<I", 0xFFFFFFFF)
    with pytest.warns(ShortWavData):
        assert decode_wav_bytes(bytes(raw)) == (_WF, _SR)


def test_decode_wav_bytes_skips_chunks_before_the_audio():
    """`fact` is legal and common; an odd-sized chunk also carries a RIFF pad byte."""
    fact = b"fact" + struct.pack("<I", 4) + struct.pack("<I", len(_WF))
    assert decode_wav_bytes(_insert_chunk_before_data(_wav(), fact)) == (_WF, _SR)

    odd = b"note" + struct.pack("<I", 3) + b"abc" + b"\x00"  # 3 bytes + pad
    assert decode_wav_bytes(_insert_chunk_before_data(_wav(), odd)) == (_WF, _SR)


@pytest.mark.parametrize("n_channels", [1, 2, 3])
@pytest.mark.parametrize("cut", range(0, 8))
def test_decode_wav_bytes_drops_a_partial_trailing_frame(n_channels, cut):
    """Truncation yields whole FRAMES, and a frame is width * n_channels bytes.

    Parametrized over channel count on purpose: with mono-only coverage, dropping the
    `* n_channels` from the frame size leaves the whole suite green while breaking
    truncated stereo with a struct error.
    """
    flat = [(-1) ** i * (i + 1) for i in range(12 * n_channels)]
    raw = _wav(flat, n_channels=n_channels)
    kept_frames = (len(flat) * 2 - cut) // (2 * n_channels)
    frames = [tuple(flat[i : i + n_channels]) for i in range(0, len(flat), n_channels)]
    expected = frames[:kept_frames]
    if n_channels == 1:
        expected = [f[0] for f in expected]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShortWavData)
        assert decode_wav_bytes(raw[: len(raw) - cut] if cut else raw) == (
            expected,
            _SR,
        )


def test_decode_wav_bytes_returns_empty_when_no_whole_frame_survives():
    """A stream cut before any audio arrived: an empty waveform, not an IndexError."""
    with pytest.warns(ShortWavData):
        assert decode_wav_bytes(_wav()[:44]) == ([], _SR)


def test_decode_wav_bytes_warns_when_the_data_chunk_is_short():
    """The caller must be able to tell a half-downloaded file from a complete one."""
    with pytest.warns(ShortWavData, match="declares 10 frames, 8 are present"):
        decode_wav_bytes(_wav()[:-4])

    # ... and stays quiet on a complete file
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShortWavData)
        assert decode_wav_bytes(_wav()) == (_WF, _SR)


def test_decode_wav_bytes_still_reads_a_plain_file():
    """The unremarkable case, stereo included -- the fix must not move the happy path."""
    assert decode_wav_bytes(_wav()) == (_WF, _SR)

    stereo = [1, -1, 2, -2, 3, -3]
    assert decode_wav_bytes(_wav(stereo, n_channels=2)) == (
        [(1, -1), (2, -2), (3, -3)],
        _SR,
    )


@pytest.mark.parametrize(
    "not_wav",
    [
        b"",
        b"short",
        b"not a wav file at all, really",
        b"RIFF" + struct.pack("<I", 4) + b"AVI ",  # RIFF, but not a WAVE form
    ],
)
def test_decode_wav_bytes_rejects_non_wav_input_clearly(not_wav):
    """Say what is wrong, rather than failing somewhere inside the codec.

    These inputs used to raise `wave.Error` or `EOFError` from the stdlib; unifying on
    `ValueError` is a deliberate new contract, not pre-existing behaviour.
    """
    with pytest.raises(ValueError, match="RIFF/WAVE"):
        decode_wav_bytes(not_wav)


def test_decode_wav_bytes_says_when_the_chunk_walk_desyncs():
    """A file that plainly has a `data` chunk, reached through a lying earlier one."""
    liar = b"junk" + struct.pack("<I", 10**6)
    with pytest.raises(ValueError, match="desynced"):
        decode_wav_bytes(_insert_chunk_before_data(_wav(), liar))


def test_decode_wav_bytes_rejects_a_riff_with_no_data_chunk():
    raw = _wav()
    assert decode_wav_bytes(raw) == (_WF, _SR)  # guard: the fixture is a real WAV
    with pytest.raises(ValueError, match="no `data` chunk"):
        decode_wav_bytes(raw.replace(b"data", b"xxxx", 1))


def test_header_size_of_wav_bytes_is_the_data_offset():
    from recode.audio import header_size_of_wav_bytes

    raw = _wav()
    assert header_size_of_wav_bytes(raw) == 44  # unchanged for a well-formed file
    # ... and stays put when metadata is appended, where subtraction would have grown it
    info = b"INFOISFT" + struct.pack("<I", 6) + b"Lavf58"
    padded = _with_riff_size_fixed(raw + b"LIST" + struct.pack("<I", len(info)) + info)
    assert header_size_of_wav_bytes(padded) == 44


def test_decode_wav_header_bytes_reports_comptype_for_pcm():
    """`comptype` is normalized to None so it round-trips with the encoder.

    It used to be bound only inside `if params.comptype == "NONE"`. That branch is
    unreachable -- stdlib `wave` rejects non-PCM formats itself -- so the guard bought
    nothing while leaving the name unbound if `wave` ever widened. Pinning the PCM
    value here keeps the round-trip honest.
    """
    from recode.audio import decode_wav_header_bytes, encode_wav_header_bytes

    meta = decode_wav_header_bytes(_wav())
    assert meta["comptype"] is None
    assert meta["sr"] == _SR and meta["width_bytes"] == 2 and meta["n_channels"] == 1
    # and it is accepted straight back by its inverse
    assert (
        decode_wav_header_bytes(
            encode_wav_header_bytes(_SR, 2, n_channels=1, comptype=meta["comptype"])
        )["comptype"]
        is None
    )


# ------ recode#12 / #13 and the last of recode#4: the wav header, read for real --------


_UNSIGNED8_BYTES = bytes([0, 64, 128, 192, 255])
_UNSIGNED8_AS_SAMPLES = [-128, -64, 0, 64, 127]


def _wav_8bit(payload=_UNSIGNED8_BYTES, sr=_SR, n_channels=1):
    """An 8-bit PCM WAV whose payload is written as the spec's UNSIGNED bytes.

    Deliberately not built through `_wav`: 8-bit is the one width where the file format
    and the in-memory sample range disagree, so the fixture states the bytes on disk
    literally rather than routing them through the packing under test.
    """
    b = io.BytesIO()
    with _wave.open(b, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(1)
        w.setframerate(sr)
        w.writeframes(payload)
    return b.getvalue()


# The SubFormat GUID that says "this EXTENSIBLE chunk really does carry plain PCM".
_PCM_SUBFORMAT_GUID = bytes.fromhex("0100000000001000800000aa00389b71")


def _extensible_wav(wf=_WF, sr=_SR, n_channels=1, width=2):
    """A `WAVE_FORMAT_EXTENSIBLE` file -- what ffmpeg writes above 2 channels or 16 bits.

    Hand-built because stdlib `wave` cannot write the tag; the layout is the plain PCM
    `fmt ` chunk with `cbSize`, `wValidBitsPerSample`, `dwChannelMask` and the SubFormat
    GUID appended, for 40 bytes in all.
    """
    code = _STRUCT_CODE_FOR_WIDTH[width]
    payload = b"".join(struct.pack("<" + code, x) for x in wf)
    block_align = n_channels * width
    fmt = (
        struct.pack(
            "<HHIIHH",
            0xFFFE,
            n_channels,
            sr,
            sr * block_align,
            block_align,
            width * 8,
        )
        + struct.pack("<HHI", 22, width * 8, 0)
        + _PCM_SUBFORMAT_GUID
    )
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


# --- the `meta` argument and `MIN_WAV_N_BYTES` that recode#4's fix dropped ---


def test_header_size_of_wav_bytes_still_takes_the_legacy_meta_argument():
    """Its own docstring says `meta` is still accepted; until now it was not.

    Passing the already-decoded header was the documented way to avoid re-parsing it,
    so a caller written against the pre-#4 signature is a plausible thing to exist.
    """
    from recode.audio import decode_wav_header_bytes, header_size_of_wav_bytes

    raw = _wav()
    assert header_size_of_wav_bytes(raw, decode_wav_header_bytes(raw)) == 44
    assert header_size_of_wav_bytes(raw, None) == header_size_of_wav_bytes(raw)


def test_min_wav_n_bytes_is_still_exported():
    """A module-level constant that shipped for years and vanished without a note."""
    from recode.audio import MIN_WAV_N_BYTES

    assert MIN_WAV_N_BYTES == 44


# --- extract_wav_header_from_file: the last size-subtraction in the module ---


@pytest.mark.parametrize(
    "shape", ["canonical", "metadata_after_data", "metadata_before_data"]
)
def test_extract_wav_header_from_file_stops_where_the_audio_starts(tmp_path, shape):
    """The file-reading twin of `header_size_of_wav_bytes` must agree with it.

    It inferred the header size as `chunk_size + 8 - subchunk2_size`, reading
    `header[40:44]` as the data size -- true only of a bare 44-byte header. With a
    `LIST` after the audio it handed back audio bytes as header; with one before, a
    truncated prefix that parses as nothing at all.
    """
    from recode.audio import extract_wav_header_from_file, header_size_of_wav_bytes

    info = b"INFOISFT" + struct.pack("<I", 6) + b"Lavf58"
    listed = b"LIST" + struct.pack("<I", len(info)) + info
    raw = {
        "canonical": lambda: _wav(),
        "metadata_after_data": lambda: _with_riff_size_fixed(_wav() + listed),
        "metadata_before_data": lambda: _insert_chunk_before_data(_wav(), listed),
    }[shape]()

    path = tmp_path / "sample.wav"
    path.write_bytes(raw)
    header = extract_wav_header_from_file(str(path))
    assert header == raw[: header_size_of_wav_bytes(raw)]
    # ... which is to say it ends on the `data` chunk header, with no audio in it
    assert header[-8:-4] == b"data"


def test_extract_wav_header_from_file_grows_its_read_until_the_header_fits(tmp_path):
    """A header longer than the first read must not truncate the answer.

    Driven with a deliberately tiny `read_size` so the growth loop is exercised by the
    test rather than only by files nobody has.
    """
    from recode.audio import extract_wav_header_from_file, header_size_of_wav_bytes

    info = b"INFOISFT" + struct.pack("<I", 6) + b"Lavf58"
    raw = _insert_chunk_before_data(
        _wav(), b"LIST" + struct.pack("<I", len(info)) + info
    )
    path = tmp_path / "sample.wav"
    path.write_bytes(raw)
    assert (
        extract_wav_header_from_file(str(path), read_size=8)
        == raw[: header_size_of_wav_bytes(raw)]
    )


def test_extract_wav_header_from_file_rejects_a_non_wav_file(tmp_path):
    """It used to answer arbitrary junk -- it never raised, whatever it was given."""
    from recode.audio import extract_wav_header_from_file

    path = tmp_path / "not.wav"
    path.write_bytes(b"this is not a wav file, not even close")
    with pytest.raises(ValueError, match="RIFF/WAVE"):
        extract_wav_header_from_file(str(path))


# --- recode#12: 8-bit WAV PCM is unsigned ---


def test_decode_wav_bytes_reads_8_bit_pcm_as_unsigned():
    """Byte 128 is digital silence in an 8-bit WAV; it used to decode as -128."""
    assert decode_wav_bytes(_wav_8bit()) == (_UNSIGNED8_AS_SAMPLES, _SR)


def test_encode_wav_bytes_writes_8_bit_pcm_as_unsigned():
    """The stdlib is the arbiter: what we write must be what `wave` reads back."""
    from recode.audio import encode_wav_bytes, header_size_of_wav_bytes

    raw = encode_wav_bytes(_UNSIGNED8_AS_SAMPLES, _SR, width_bytes=1)
    assert raw[header_size_of_wav_bytes(raw) :] == _UNSIGNED8_BYTES
    with _wave.open(io.BytesIO(raw)) as w:
        assert w.getsampwidth() == 1
        assert w.readframes(w.getnframes()) == _UNSIGNED8_BYTES


def test_8_bit_wav_round_trips_through_recode():
    from recode.audio import encode_wav_bytes

    raw = encode_wav_bytes(_UNSIGNED8_AS_SAMPLES, _SR, width_bytes=1)
    assert decode_wav_bytes(raw) == (_UNSIGNED8_AS_SAMPLES, _SR)


def test_8_bit_wav_handles_multiple_channels():
    """The bias applies per sample, not per frame -- multichannel frames are tuples."""
    raw = _wav_8bit(bytes([0, 128, 255, 64]), n_channels=2)
    assert decode_wav_bytes(raw) == ([(-128, 0), (127, -64)], _SR)


def test_8_bit_signed_stays_reachable_as_an_opt_out():
    """Anyone who paired recode's encoder with its decoder gets their bytes back."""
    from recode.audio import encode_wav_bytes, header_size_of_wav_bytes

    assert decode_wav_bytes(_wav_8bit(), eight_bit_unsigned=False) == (
        [0, 64, -128, -64, -1],
        _SR,
    )
    legacy = encode_wav_bytes(
        _UNSIGNED8_AS_SAMPLES, _SR, width_bytes=1, eight_bit_unsigned=False
    )
    assert legacy[header_size_of_wav_bytes(legacy) :] == bytes([128, 192, 0, 64, 127])


def test_the_8_bit_fix_leaves_every_other_width_alone():
    """16-bit is signed and must stay so; only width 1 changes."""
    from recode.audio import encode_wav_bytes

    assert decode_wav_bytes(_wav()) == (_WF, _SR)
    assert encode_wav_bytes(_WF, _SR) == _wav()
    assert encode_wav_bytes(_WF, _SR, width_bytes=4) == _wav(width=4)


# --- recode#13: WAVE_FORMAT_EXTENSIBLE ---


def test_decode_wav_bytes_reads_wave_format_extensible():
    """The same file decodes on 3.12 and raised `unknown format: 65534` on 3.10.

    Stdlib `wave` only learned the 0xFFFE tag in 3.12, and 3.10 is what CI runs -- so a
    plain 6-channel ffmpeg file was undecodable on the only version that gates a merge.
    """
    assert decode_wav_bytes(_extensible_wav()) == (_WF, _SR)


def test_extensible_and_plain_wav_headers_report_the_same_thing():
    from recode.audio import decode_wav_header_bytes

    assert decode_wav_header_bytes(_extensible_wav()) == decode_wav_header_bytes(_wav())


def test_extensible_wav_decodes_multichannel():
    """Multichannel is the reason encoders reach for EXTENSIBLE in the first place."""
    stereo = [1, -1, 2, -2, 3, -3]
    assert decode_wav_bytes(_extensible_wav(stereo, n_channels=2)) == (
        [(1, -1), (2, -2), (3, -3)],
        _SR,
    )


def test_a_genuinely_unsupported_format_tag_still_raises():
    """The fallback must widen what is read, not silence what should be refused."""
    raw = bytearray(_extensible_wav())
    at = raw.find(b"fmt ") + 8
    raw[at : at + 2] = struct.pack("<H", 0x0011)  # IMA ADPCM
    with pytest.raises(_wave.Error):
        decode_wav_bytes(bytes(raw))


def test_a_non_wave_riff_still_raises_the_way_the_stdlib_says():
    """`decode_wav_header_bytes` must not swap `wave`'s complaint for another one."""
    from recode.audio import decode_wav_header_bytes

    with pytest.raises(_wave.Error):
        decode_wav_header_bytes(b"RIFF" + struct.pack("<I", 4) + b"AVI ")


def test_a_non_pcm_extensible_subformat_still_raises():
    """The versions must agree on what they refuse, not only on what they read.

    3.12's own `wave` insists on `KSDATAFORMAT_SUBTYPE_PCM` for an EXTENSIBLE chunk;
    so does the fallback, so a compressed payload is turned away on every version.
    """
    raw = _extensible_wav()
    ieee_float_guid = bytes.fromhex("0300000000001000800000aa00389b71")
    non_pcm = raw.replace(_PCM_SUBFORMAT_GUID, ieee_float_guid)
    assert non_pcm != raw  # guard: the fixture really did carry the PCM GUID
    with pytest.raises(_wave.Error):
        decode_wav_bytes(non_pcm)


def test_8_bit_encode_accepts_a_numpy_int8_waveform():
    """int8 is the natural dtype of 8-bit audio; biasing it by 128 must not overflow.

    Before the unsigned-8-bit fix, an int8 array went straight to the struct codec and
    encoded fine; the bias then added a Python 128 to each np.int8 sample, which numpy 2
    refuses (OverflowError) and numpy 1 silently wraps.
    """
    np = pytest.importorskip("numpy")
    from recode.audio import encode_wav_bytes

    mono = np.array([-128, 0, 127], dtype=np.int8)
    raw = encode_wav_bytes(mono, _SR, width_bytes=1)
    assert decode_wav_bytes(raw) == ([-128, 0, 127], _SR)

    stereo = np.array([[-128, 1], [0, 127]], dtype=np.int8)
    raw = encode_wav_bytes(stereo, _SR, width_bytes=1, n_channels=2)
    assert decode_wav_bytes(raw) == ([(-128, 1), (0, 127)], _SR)
