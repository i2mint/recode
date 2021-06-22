
# recode
Make codecs for fixed size structured chunks serialization and deserialization of
sequences, tabular data, and time-series.

To install:	```pip install recode```

# Examples

## Single channel numerical stream

```python
>>> from recode import StructCodecSpecs
>>> specs = StructCodecSpecs(
...     chk_format='h',
... )
>>> print(specs)
StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)
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
>>> frames = [1, 2, 3]
>>> b = encoder(frames)
>>> assert b == b'\x01\x00\x02\x00\x03\x00'
>>> decoded_frames = list(decoder(b))
>>> assert decoded_frames == frames
```


## Multi-channel numerical stream
```python
>>> from recode import StructCodecSpecs
>>> specs = StructCodecSpecs(
...     chk_format='@h',
...     n_channels = 2
... )
>>> print(specs)
StructCodecSpecs(chk_format='@hh', n_channels=2, chk_size_bytes=4)
>>>
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
```


## Iterative decoder
```python
>>> specs = StructCodecSpecs(chk_format = 'hdhd')
>>> print(specs)
StructCodecSpecs(chk_format='hdhd', n_channels=4, chk_size_bytes=32)
>>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
>>> decoder = IterativeDecoder(chk_to_frame = specs.chk_to_frame_iter)
>>> frames = [(1,1.1,1,1.1),(2,2.2,2,2.2),(3,3.3,3,3.3)]
>>> b = encoder(frames)
>>> iter_frames = decoder(b)
>>> assert next(iter_frames) == frames[0]
>>> next(iter_frames)
(2, 2.2, 2, 2.2)
```    


## DataFrame (as list of dicts) using MetaEncoder/MetaDecoder
```python
>>> data = [{'foo': 1.1, 'bar': 2.2}, {'foo': 513.23, 'bar': 456.1}, {'foo': 32.0, 'bar': 6.7}]
>>> specs = StructCodecSpecs(chk_format='d', n_channels = 2)
>>> print(specs)
StructCodecSpecs(chk_format='dd', n_channels=2, chk_size_bytes=16)
>>> encoder = MetaEncoder(frame_to_chk = specs.frame_to_chk, frame_to_meta = frames_to_meta)
>>> decoder = MetaDecoder(chk_to_frame = specs.chk_to_frame, 
...                       n_channels = specs.n_channels,
...                       chk_size_bytes = specs.chk_size_bytes,
...                       meta_to_frame = meta_to_frames)
>>> b = encoder(data)
>>> assert decoder(b) == data
```


## Implicitly define codec specs based on frames

`specs_from_frames` allows for the implicit definition of StructCodecSpecs based on the frames that are being encoded. `specs_from_frames` returns a tuple of an iterator of the frames being encoded and the defined StructCodecSpecs. The first element of the tuple can be ignored unless frames is an iterator.

```python
>>> frames = [1,2,3]
>>> _, specs = specs_from_frames(frames)
>>> print(specs)
StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)
>>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
>>> decoder = ChunkedDecoder(
...     chk_to_frame = specs.chk_to_frame,
...     n_channels = specs.n_channels,
...     chk_size_bytes = specs.chk_size_bytes
... )
>>> b = encoder(frames)
>>> assert b == b'\x01\x00\x02\x00\x03\x00'
>>> decoded_frames = list(decoder(b))
>>> decoded_frames
[1, 2, 3]
```

## Implicit definition of iterator
```python
>>> specs = StructCodecSpecs(chk_format = 'hdhd')
>>> print(specs)
StructCodecSpecs(chk_format='hdhd', n_channels=4, chk_size_bytes=32)
>>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
>>> decoder = IterativeDecoder(chk_to_frame = specs.chk_to_frame_iter)
>>> frames = [(1,1.1,1,1.1),(2,2.2,2,2.2),(3,3.3,3,3.3)]
>>> b = encoder(frames)
>>> iter_frames = decoder(b)
>>> assert next(iter_frames) == frames[0]
>>> next(iter_frames)
(2, 2.2, 2, 2.2)
```
