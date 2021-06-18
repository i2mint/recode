
# recode
Make codecs for fixed size structured chunks serialization and deserialization of
sequences, tabular data, and time-series.

To install:	```pip install recode```

# Examples

## single channel numerical stream

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

