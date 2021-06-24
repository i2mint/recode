"""
Make codecs for fixed size structured chunks serialization and deserialization of
sequences, tabular data, and time-series.
"""
from dataclasses import dataclass
from typing import Iterable, Callable, Sequence, Union, Any
from struct import pack, unpack, iter_unpack
import struct
from operator import itemgetter
from recode.util import spy, get_struct, list_of_dicts


Meta = Sequence[bytes]
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

MetaToFrame = Callable[[Meta], Frame]
FrameToMeta = Callable[[Frame], Meta]


@dataclass
class ChunkedEncoder(Encoder):
    """
    Serializes numerical streams and sequences
    """

    frame_to_chk: FrameToChunk

    def __call__(self, frames: Frames):
        return b''.join(map(self.frame_to_chk, frames))


@dataclass
class MetaEncoder(Encoder):
    """
    Serializes tabular data (must be formatted as list of dicts)
    """

    frame_to_chk: FrameToChunk
    frame_to_meta: FrameToMeta

    def __call__(self, frames: Frames):
        meta = self.frame_to_meta(frames)
        vals = list(map(list, (d.values() for d in frames)))
        if len(vals[0]) == 1:
            vals = [item for sublist in vals for item in sublist]
        return meta + b''.join(map(self.frame_to_chk, vals))


first_element = itemgetter(0)  # "equivalent" to lambda x: x[0]


@dataclass
class ChunkedDecoder(Decoder):
    """
    Deserializes numerical streams and sequences serialized by ChunkedEncoder
    """

    chk_to_frame: ChunkToFrame

    def __call__(self, b: bytes):
        iterator = self.chk_to_frame(b)
        frame = list(iterator)
        if len(frame[0]) == 1:
            frame = [item for tup in frame for item in tup]
        return frame


@dataclass
class IterativeDecoder(Decoder):
    """
    Creates an iterator of deserialized chunks of numerical streams and sequences serialized
    by ChunkedEncoder
    """

    chk_to_frame: ChunkToFrame

    def __call__(self, b: bytes):
        iterator = self.chk_to_frame(b)
        return iterator


@dataclass
class MetaDecoder(Decoder):
    """
    Deserializes tabular data serialized by MetaEncoder
    """

    chk_to_frame: ChunkToFrame
    meta_to_frame: MetaToFrame

    def __call__(self, b: bytes):
        cols, split = self.meta_to_frame(b)
        b = b[split:]
        vals = self.chk_to_frame(b)
        return list_of_dicts(cols, vals)


def _split_chk_format(chk_format):
    """
    Splits a struct format string into the byte order character and format characters
    >>> assert _split_chk_format('hq') == ('', 'hq')
    >>> assert _split_chk_format('@hq') == ('@', 'hq')
    """
    if chk_format[0] in '@=<>!':
        return chk_format[0], chk_format[1:]
    return '', chk_format


def _format_chars_part_of_chk_format(chk_format):
    """
    Returns the format character part of a struct format string
    >>> assert _format_chars_part_of_chk_format('!hh') == 'hh'
    >>> _format_chars_part_of_chk_format('q')
    'q'
    """
    byte_order, format_chars = _split_chk_format(chk_format)
    return format_chars


def _chk_format_to_n_channels(chk_format):
    """
    Returns the number of channels defined a struct format string
    >>> assert _chk_format_to_n_channels('hq') == 2
    >>> _chk_format_to_n_channels('@hqt')
    3
    """
    return len(_format_chars_part_of_chk_format(chk_format))


def _chk_format_is_for_single_channel(chk_format):
    """
    Returns if a struct format string is designated for a single channel of data
    >>> assert _chk_format_is_for_single_channel('h') == True
    >>> assert _chk_format_is_for_single_channel('@hq') == False
    """
    return _chk_format_to_n_channels(chk_format) == 1


def frame_to_meta(frame):
    r"""
    Defines header for serialization of tabluar data
    >>> rows = [{'customer': 1}, {'customer': 2}, {'customer': 3}]
    >>> assert frame_to_meta(rows) == b'\x08\x00customer'
    """
    cols = list(frame[0].keys())
    s = '.'.join(cols)
    return b'' + pack('h', len(s)) + s.encode()


def meta_to_frame(meta):
    r"""
    Deserializes header for deserialization of tabular data
    >>> meta = b'\x1c\x00customer.apple.banana.tomato\x01\x00\x01\x00\x02\x00\x03\x00\x02\x00\
    ... x03\x00\x02\x00\x05\x00\x01\x00\x03\x00\x04\x00\t\x00'
    >>> assert meta_to_frame(meta)[0] == ['customer', 'apple', 'banana', 'tomato']
    """
    length = unpack('h', meta[:2])[0] + 2
    cols = (meta[2:length]).decode().split('.')
    return cols, length


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
    >>> encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    >>> decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
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
    >>> encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    >>> decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    >>> frames = [(1, 2), (3, 4), (5, 6)]
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
    >>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
    >>> decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    >>> frames = [(1, 2.45, 1), (3, 4.321, 3)]
    >>> b = encoder(frames)
    >>> assert b == b'\x01\x00\x9a\x99\x99\x99\x99\x99\x03@\x01\x00\x03\x00b\x10X9\xb4H\x11@\x03\x00'
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == frames

    Along with a ChunkedDecoder, you can also use an IterativeDecoder which implements the struct.iter_unpack function.
    IterativeDecoder only requires one argument, a chk_to_frame_iter function, and returns an iterator of each chunk.
    An example of an IterativeDecorator can be seen below.

    >>> specs = StructCodecSpecs(chk_format = 'hdhd')
    >>> print(specs)
    StructCodecSpecs(chk_format='hdhd', n_channels=4, chk_size_bytes=32)
    >>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
    >>> decoder = IterativeDecoder(chk_to_frame = specs.chk_to_frame)
    >>> frames = [(1,1.1,1,1.1),(2,2.2,2,2.2),(3,3.3,3,3.3)]
    >>> b = encoder(frames)
    >>> iter_frames = decoder(b)
    >>> assert next(iter_frames) == frames[0]
    >>> next(iter_frames)
    (2, 2.2, 2, 2.2)

    Along with using recode for the kinds of data we have looked at so far, it can also be applied to DataFrames when
    they have been converted to a list of dicts using MetaEncoder and MetaDecoder. An example of this can be seen below.

    >>> data = [{'foo': 1.1, 'bar': 2.2}, {'foo': 513.23, 'bar': 456.1}, {'foo': 32.0, 'bar': 6.7}]
    >>> specs = StructCodecSpecs(chk_format='d', n_channels = 2)
    >>> print(specs)
    StructCodecSpecs(chk_format='dd', n_channels=2, chk_size_bytes=16)
    >>> encoder = MetaEncoder(frame_to_chk = specs.frame_to_chk, frame_to_meta = frame_to_meta)
    >>> decoder = MetaDecoder(chk_to_frame = specs.chk_to_frame, meta_to_frame = meta_to_frame)
    >>> b = encoder(data)
    >>> assert decoder(b) == data
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
        if self.n_channels == 1:
            return pack(self.chk_format, frame)
        else:
            return pack(self.chk_format, *frame)

    def chk_to_frame(self, chk):
        return iter_unpack(self.chk_format, chk)


def specs_from_frames(frames: Frames):
    r"""
    Implicitly defines the codec specs based on the frames to encode/decode.
    specs_from_frames returns a tuple of an iterator of frames and the defined StructCodecSpecs. If frames is an
    iterable, then the iterator can be ignored like the following example.
    >>> frames = [1,2,3]
    >>> _, specs = specs_from_frames(frames)
    >>> print(specs)
    StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)
    >>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
    >>> decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    >>> b = encoder(frames)
    >>> assert b == b'\x01\x00\x02\x00\x03\x00'
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == frames

    If frames is an iterator, then we can still use specs_from_frames as long as we redefine frames from the output like
    in the following example.

    >>> frames = iter([[1.1,2.2],[3.3,4.4]])
    >>> frames, specs = specs_from_frames(frames)
    >>> print(specs)
    StructCodecSpecs(chk_format='dd', n_channels=2, chk_size_bytes=16)
    >>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
    >>> decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    >>> b = encoder(frames)
    >>> decoded_frames = list(decoder(b))
    >>> assert decoded_frames == [(1.1,2.2),(3.3,4.4)]
    """
    head, frames = spy(frames)

    if isinstance(head[0], (int, float)):
        format_char = get_struct(type(head[0]))
        n_channels = 1
    elif isinstance(head[0], (list, tuple)):
        format_char = get_struct(type(head[0][0]))
        n_channels = len(head[0])
    elif isinstance(head[0], dict):
        format_char = get_struct(type(list(head[0].values())[0]))
        n_channels = len(head[0].keys())
    else:
        raise AttributeError('Unknown data format')

    return frames, StructCodecSpecs(format_char, n_channels=n_channels)
