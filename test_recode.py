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
)
import pytest
from typing import Iterator


@pytest.mark.parametrize(
    'chk_format,frame,n_channels',
    [
        ('h', [1, 2, 3, 4, 5], None),
        ('h', [1, 2, 3, 4, 5], 1),
        ('h', [1], None),
        ('h', [1], 1),
        ('d', [1.1, 2.2, 3.3], None),
        ('d', [1.1, 2.2, 3.3], 1),
        ('d', [1.1], None),
        ('d', [1.1], 1),
        ('=h', [1, 2, 3, 4, 5], None),
        ('<h', [1, 2, 3, 4, 5], None),
        ('>h', [1, 2, 3, 4, 5], None),
        ('!h', [1, 2, 3, 4, 5], None),
    ],
)
def test_single_channel_chunk(chk_format, frame, n_channels):
    specs = StructCodecSpecs(chk_format=chk_format, n_channels=n_channels)
    encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    b = encoder(frame)
    assert isinstance(b, bytes)
    decoded_frames = decoder(b)
    assert decoded_frames == frame


@pytest.mark.parametrize(
    'chk_format,frame,n_channels',
    [
        ('hh', [[1, 1]], None),
        ('h', [[1, 1]], 2),
        ('hh', [[1, 1], [2, 2]], None),
        ('h', [[1, 1], [2, 2]], 2),
        ('dd', [[1.1, 1.1]], None),
        ('d', [[1.1, 1.1]], 2),
        ('dd', [[1.1, 1.1], [2.2, 2.2]], None),
        ('d', [[1.1, 1.1], [2.2, 2.2]], 2),
        (
            'hhhhhhhhh',
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            None,
        ),
        (
            'h',
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            9,
        ),
        ('=hh', [[1, 2]], None),
        ('>hh', [[1, 2]], None),
        ('<hh', [[1, 2]], None),
        ('!hh', [[1, 2]], None),
    ],
)
def test_multi_channel_chunk(chk_format, frame, n_channels):
    specs = StructCodecSpecs(chk_format=chk_format, n_channels=n_channels)
    encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
    b = encoder(frame)
    assert isinstance(b, bytes)
    decoded_frames = list(map(list, decoder(b)))
    assert decoded_frames == frame


@pytest.mark.parametrize(
    'chk_format,frame,n_channels',
    [
        ('h', [1, 2, 3, 4, 5], None),
        ('h', [1, 2, 3, 4, 5], 1),
        ('h', [1], None),
        ('h', [1], 1),
        ('d', [1.1, 2.2, 3.3], None),
        ('d', [1.1, 2.2, 3.3], 1),
        ('d', [1.1], None),
        ('d', [1.1], 1),
        ('=h', [1, 2, 3, 4, 5], None),
        ('<h', [1, 2, 3, 4, 5], None),
        ('>h', [1, 2, 3, 4, 5], None),
        ('!h', [1, 2, 3, 4, 5], None),
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
    'chk_format,frame,n_channels',
    [
        ('hh', [[1, 1]], None),
        ('h', [[1, 1]], 2),
        ('hh', [[1, 1], [2, 2]], None),
        ('h', [[1, 1], [2, 2]], 2),
        ('dd', [[1.1, 1.1]], None),
        ('d', [[1.1, 1.1]], 2),
        ('dd', [[1.1, 1.1], [2.2, 2.2]], None),
        ('d', [[1.1, 1.1], [2.2, 2.2]], 2),
        (
            'hhhhhhhhh',
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            None,
        ),
        (
            'h',
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            9,
        ),
        ('=hh', [[1, 2]], None),
        ('>hh', [[1, 2]], None),
        ('<hh', [[1, 2]], None),
        ('!hh', [[1, 2]], None),
    ],
)
def test_multi_channel_iter(chk_format, frame, n_channels):
    specs = StructCodecSpecs(chk_format=chk_format, n_channels=n_channels)
    encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
    decoder = IterativeDecoder(chk_to_frame=specs.chk_to_frame)
    b = encoder(frame)
    assert isinstance(b, bytes)
    decoded_frames = decoder(b)
    assert isinstance(decoded_frames, Iterator)
    assert list(next(decoded_frames)) == frame[0]


@pytest.mark.parametrize(
    'chk_format,table,n_channels',
    [
        ('h', [{'a': 1}], 1),
        ('h', [{'a': 1}, {'a': 2}], 1),
        ('d', [{'a': 1.1}], 1),
        ('d', [{'a': 1.1}, {'a': 2.2}], 1),
        ('hh', [{'a': 1, 'b': 2}], None),
        ('hh', [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], None),
        ('dd', [{'a': 1.1, 'b': 2.2}], None),
        ('dd', [{'a': 1.1, 'b': 2.2}, {'a': 3.3, 'b': 4.4}], None),
    ],
)
def test_tabular(chk_format, table, n_channels):
    specs = StructCodecSpecs(chk_format=chk_format, n_channels=n_channels)
    encoder = MetaEncoder(frame_to_chk=specs.frame_to_chk, frame_to_meta=frame_to_meta)
    decoder = MetaDecoder(chk_to_frame=specs.chk_to_frame, meta_to_frame=meta_to_frame)
    b = encoder(table)
    assert isinstance(b, bytes)
    decoded_frames = decoder(b)
    assert decoded_frames == table


@pytest.mark.parametrize(
    'chk_format,frame,n_channels',
    [
        ('h', [1, 2], 1),
        ('hh', [[1, 1], [2, 2]], 2),
        ('h', [1], 1),
        ('hh', [[1, 1]], 2),
        ('d', [1.1, 2.2], 1),
        ('dd', [[1.1, 1.1], [2.2, 2.2]], 2),
        ('d', [1.1], 1),
        ('dd', [[1.1, 1.1]], 2),
        ('h', iter([1, 2]), 1),
        ('hh', iter([[1, 1], [2, 2]]), 2),
        ('h', iter([1]), 1),
        ('hh', iter([[1, 1]]), 2),
        ('d', iter([1.1, 2.2]), 1),
        ('dd', iter([[1.1, 1.1], [2.2, 2.2]]), 2),
        ('d', iter([1.1]), 1),
        ('dd', iter([[1.1, 1.1]]), 2),
        ('h', [{'a': 1}], 1),
        ('h', [{'a': 1}, {'a': 2}], 1),
        ('d', [{'a': 1.1}], 1),
        ('d', [{'a': 1.1}, {'a': 2.2}], 1),
        ('hh', [{'a': 1, 'b': 2}], 2),
        ('hh', [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], 2),
        ('dd', [{'a': 1.1, 'b': 2.2}], 2),
        ('dd', [{'a': 1.1, 'b': 2.2}, {'a': 3.3, 'b': 4.4}], 2),
    ],
)
def test_implicit_specs(chk_format, frame, n_channels):
    _, specs = specs_from_frames(frame)
    assert specs.chk_format == chk_format
    assert specs.n_channels == n_channels
