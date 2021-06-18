"""
Make codecs for fixed size structured chunks serialization and deserialization of
sequences, tabular data, and time-series.
"""
from dataclasses import dataclass
from typing import Iterable, Callable, Sequence, Union, Any
from struct import pack, unpack
import struct
from operator import itemgetter

Chunk = Sequence[bytes]
Chunks = Iterable[Chunk]
ByteChunker = Callable[[bytes], Chunks]
Sample = Any  # but usually a number
Frame = Union[Sample, Sequence[Sample]]
Frames = Iterable[Frame]

Encoder = Callable[[Frames], bytes]
Decoder = Callable[[bytes], Iterable[Frame]]

ChunkToFrame = Callable[[Chunk], Frame]
FrameToChunk = Callable[[Frame], Chunk]


@dataclass
class ChunkedEncoder(Encoder):
    frame_to_chk: FrameToChunk

    def __call__(self, frames: Frames):
        return b''.join(map(self.frame_to_chk, frames))


first_element = itemgetter(0)  # "equivalent" to lambda x: x[0]


@dataclass
class ChunkedDecoder(Decoder):
    chk_to_frame: ChunkToFrame
    n_channels: int
    chk_size_bytes: int

    # ByteChunker
    def chunker(self, b: bytes) -> Chunks:
        # TODO: Check efficiency against other byte-specific chunkers
        return map(bytes, zip(*([iter(b)] * self.chk_size_bytes)))

    def __call__(self, b: bytes):
        frames = map(self.chk_to_frame, self.chunker(b))
        if self.n_channels == 1:
            return map(first_element, frames)
        return frames


def _split_chk_format(chk_format):
    """
    >>> assert _split_chk_format('hq') == ('', 'hq')
    >>> assert _split_chk_format('@hq') == ('@', 'hq')
    """
    if chk_format[0] in '@=<>!':
        return chk_format[0], chk_format[1:]
    return '', chk_format


def _format_chars_part_of_chk_format(chk_format):
    """
    >>> assert _format_chars_part_of_chk_format('!hh') == 'hh'
    >>> _format_chars_part_of_chk_format('q')
    'q'
    """
    byte_order, format_chars = _split_chk_format(chk_format)
    return format_chars


def _chk_format_to_n_channels(chk_format):
    """
    >>> assert _chk_format_to_n_channels('hq') == 2
    >>> _chk_format_to_n_channels('@hqt')
    3
    """
    return len(_format_chars_part_of_chk_format(chk_format))


def _chk_format_is_for_single_channel(chk_format):
    """
    >>> assert _chk_format_is_for_single_channel('h') == True
    >>> assert _chk_format_is_for_single_channel('@hq') == False
    """
    return _chk_format_to_n_channels(chk_format) == 1


@dataclass
class StructCodecSpecs:
    r"""Enable the definition of codec specs based on format characters of the
    python struct module
    (https://docs.python.org/3/library/struct.html#format-characters)

    :param chk_format: The format of a chunk, as specified by the struct module
        See https://docs.python.org/3/library/struct.html#format-characters
    :param n_channels: Only n_channels = 1 serves a purpose; to indicate that

    To utilise recode, first define your codec specs. If your frame is only one channel (ex. a list) then your format
    string will include two characters, a byte character (@, =, <, >, !) and a format character. The format character
    should match the data type of the samples in the frame so they are properly encoded/decoded. This can be seen in the
    following example.

    >>> specs = StructCodecSpecs(chk_format='h')
    >>> print(specs)
    StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)
    >>>
    >>> encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    >>> decoder = ChunkedDecoder(
    ...     chk_size_bytes=specs.chk_size_bytes,
    ...     chk_to_frame=specs.chk_to_frame,
    ...     n_channels=specs.n_channels
    ... )
    >>>
    >>> frames = [1, 2, 3]
    >>> b = encoder(frames)
    >>> assert b == b'\x01\x00\x02\x00\x03\x00'
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == frames

    If your data has more than one channel, then there are two options. If all of the samples are of the same data type,
    say integer, then your format string needs only one format character and then you can specify the number of channels
    in your data with the n_channels argument. This can be seen in the following example.

    >>> specs = StructCodecSpecs(chk_format='@h', n_channels=2)
    >>> print(specs)
    StructCodecSpecs(chk_format='@hh', n_channels=2, chk_size_bytes=4)
    >>>
    >>> encoder = ChunkedEncoder(
    ...     frame_to_chk=specs.frame_to_chk
    ... )
    >>> decoder = ChunkedDecoder(
    ...     chk_size_bytes=specs.chk_size_bytes,
    ...     chk_to_frame=specs.chk_to_frame,
    ...     n_channels=specs.n_channels
    ... )
    >>>
    >>> frames = [(1, 2), (3, 4), (5, 6)]
    >>>
    >>> b = encoder(frames)
    >>> assert b == b'\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00'
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == frames

    On the other hand, if each channel has a different data type, say (int, float, int), then your format string needs
    a format character for each of your channels. This can be seen in the following example, which also shows the use
    of a different byte character (=).

    >>> specs = StructCodecSpecs(chk_format = '=hdh')
    >>> print(specs)
    StructCodecSpecs(chk_format='=hdh', n_channels=3, chk_size_bytes=12)
    >>>
    >>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
    >>> decoder = ChunkedDecoder(
    ...     chk_size_bytes = specs.chk_size_bytes,
    ...     chk_to_frame = specs.chk_to_frame,
    ...     n_channels = specs.n_channels
    ... )
    >>> frames = [(1, 2.45, 1), (3, 4.321, 3)]
    >>> b = encoder(frames)
    >>> assert b == b'\x01\x00\x9a\x99\x99\x99\x99\x99\x03@\x01\x00\x03\x00b\x10X9\xb4H\x11@\x03\x00'
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == frames
    """
    chk_format: str
    n_channels: int = None
    chk_size_bytes: int = None

    def __post_init__(self):
        inferred_n_channels = _chk_format_to_n_channels(self.chk_format)
        if self.n_channels is None:
            self.n_channels = inferred_n_channels
        else:
            assert (
                inferred_n_channels == 1
            ), 'if n_channels is given, chk_format needs to be for a single channel'
            byte_order, format_chars = _split_chk_format(self.chk_format)
            self.chk_format = byte_order + format_chars * self.n_channels

        chk_size_bytes = struct.calcsize(self.chk_format)
        if self.chk_size_bytes is None:
            self.chk_size_bytes = chk_size_bytes
        else:
            assert self.chk_size_bytes == chk_size_bytes, (
                f'The given chk_size_bytes {self.chk_size_bytes} did not match the '
                f'inferred (from chk_format) {chk_size_bytes}'
            )

    def frame_to_chk(self, frame):
        # Note: Removed the assertions because
        #  (1) It lead to errors (len(frame) not defined when n_channels==1
        #  (2) It might be a bit heavy to do this (calls to frame_to_chk frequent)
        if self.n_channels == 1:
            # assert len(frame) == _chk_format_to_n_channels(self.chk_format), (
            #     f'The inferred n_channels {_chk_format_to_n_channels(self.chk_format)} did not match the '
            #     f'inferred (from frames) {len(frame)}'
            # )
            return pack(self.chk_format, frame)
        else:
            # assert len(frame) == self.n_channels, (
            #     f'The given n_channels {self.n_channels} did not match the '
            #     f'inferred (from frames) {len(frame)}'
            # )
            return pack(self.chk_format, *frame)

    def chk_to_frame(self, chk):
        return unpack(self.chk_format, chk)
