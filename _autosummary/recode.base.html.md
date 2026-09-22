# recode.base

Base recode objects

### Functions

| `add_coding_attributes`(to_obj, from_obj)                                                  |                                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`frame_to_meta`](#recode.base.frame_to_meta)(frame)                      | Defines header for serialization of tabluar data                                                                                                                                                                                 |
| [`meta_to_frame`](#recode.base.meta_to_frame)(meta)                       | Deserializes header for deserialization of tabular data                                                                                                                                                                          |
| [`mk_codec`](#recode.base.mk_codec)([chk_format, n_channels, ...])   | Enable the definition of codec specs based on format characters of the python struct module ([https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters)) |
| [`mk_encoder_and_decoder`](#recode.base.mk_encoder_and_decoder)([chk_format, ...]) | Enable the definition of codec specs based on format characters of the python struct module ([https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters)) |
| [`specs_from_frames`](#recode.base.specs_from_frames)(frames)                 | Implicitly defines the codec specs based on the frames to encode/decode.                                                                                                                                                         |

### Classes

| [`ChunkedDecoder`](#recode.base.ChunkedDecoder)(chk_to_frame[, chk_format, ...])   | Deserializes numerical streams and sequences serialized by ChunkedEncoder                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ChunkedEncoder`](#recode.base.ChunkedEncoder)(frame_to_chk[, chk_format, ...])   | Serializes numerical streams and sequences                                                                                                                                                                                       |
| [`IterativeDecoder`](#recode.base.IterativeDecoder)(chk_to_frame)                    | Creates an iterator of deserialized chunks of numerical streams and sequences serialized by ChunkedEncoder                                                                                                                       |
| [`MetaDecoder`](#recode.base.MetaDecoder)(chk_to_frame, meta_to_frame)          | Deserializes tabular data serialized by MetaEncoder                                                                                                                                                                              |
| [`MetaEncoder`](#recode.base.MetaEncoder)(frame_to_chk, frame_to_meta)          | Serializes tabular data (must be formatted as list of dicts)                                                                                                                                                                     |
| [`StructCodecSpecs`](#recode.base.StructCodecSpecs)([chk_format, n_channels, ...])   | Enable the definition of codec specs based on format characters of the python struct module ([https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters)) |
| [`codec_tuple`](#recode.base.codec_tuple)(encode, decode)                       |                                                                                                                                                                                                                                  |

### *class* recode.base.ChunkedDecoder(chk_to_frame, chk_format=None, n_channels=None, chk_size_bytes=None)

Bases: [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]

Deserializes numerical streams and sequences serialized by ChunkedEncoder

### *class* recode.base.ChunkedEncoder(frame_to_chk, chk_format=None, n_channels=None, chk_size_bytes=None)

Bases: [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]], [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]

Serializes numerical streams and sequences

### *class* recode.base.IterativeDecoder(chk_to_frame)

Bases: [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]

Creates an iterator of deserialized chunks of numerical streams and sequences serialized
by ChunkedEncoder

### *class* recode.base.MetaDecoder(chk_to_frame, meta_to_frame)

Bases: [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]

Deserializes tabular data serialized by MetaEncoder

### *class* recode.base.MetaEncoder(frame_to_chk, frame_to_meta)

Bases: [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]], [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]

Serializes tabular data (must be formatted as list of dicts)

### *class* recode.base.StructCodecSpecs(chk_format='d', n_channels=None, chk_size_bytes=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Enable the definition of codec specs based on format characters of the
python struct module
([https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters))

* **Parameters:**
  * **chk_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The format of a chunk, as specified by the struct module
    See [https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters)
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Expected number of channels. If given, will assert that the
    number of channels expressed by the `chk_format` is indeed what is expected.
  * **chk_size_bytes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Expected number of bytes per chunk.
    If given, will assert that the chunk size expressed by the `chk_format` is
    indeed the one expected.

#### NOTE
All encoder/decoder (codec) specs can be expressed through the `chk_format`.
Yet, though `n_channels` and `chk_size_bytes` are both optional, it is advised to
include them in production code since they act as extra confirmation of the codec
to be used. Encoding and decoding problems can be hard to notice until much
later on downstream, and are therefore hard to debug.

To utilise recode, first define your codec specs. If your frame is only one channel,
then your format string will include two characters maximum: an optional special character to
control the byte order, size and alignment (@, =, <, >, !), and a format character to specify
the type of data being packed/unpacked. The format character should match the data type of the
samples in the frame so they are properly encoded/decoded.

This can be seen in the following example.

```pycon
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
```

The only reason (but it’s a good one) to specify `n_channels` is to assert them.

```pycon
>>> specs = StructCodecSpecs(chk_format='@hh', n_channels=2)
>>> print(specs)
StructCodecSpecs(chk_format='@hh', n_channels=2, chk_size_bytes=4)
>>> encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
>>> decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
>>> frames = [(1, 2), (3, 4), (5, 6)]
>>> b = encoder(frames)
>>> assert b == b'\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00'
>>> decoded_frames = list(decoder(b))
>>> assert decoded_frames == frames
```

On the other hand, if each channel has a different data type, say (int, float, int),
then your format string needs a format character for each of your channels.
This can be seen in the following example, which also shows the use
of a different byte character (=).

```pycon
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
```

You can also use the IterativeDecoder which will return an iterator of frames instead of the
full list of frames, similar to what struct.iter_unpack does.
IterativeDecoder can be instantiated and called in the same way as ChunkedDecoder.
An example of IterativeDecorator can be seen below.

```pycon
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
```

Along with using recode for the kinds of data we have looked at so far,
it can also be applied to DataFrames when
they have been converted to a list of dicts using MetaEncoder and MetaDecoder.
An example of this can be seen below.

```pycon
>>> data = [{'foo': 1.1, 'bar': 2.2},
...         {'foo': 513.23, 'bar': 456.1},
...         {'foo': 32.0, 'bar': 6.7}]
>>> specs = StructCodecSpecs(chk_format='dd')
>>> print(specs)
StructCodecSpecs(chk_format='dd', n_channels=2, chk_size_bytes=16)
>>> encoder = MetaEncoder(frame_to_chk = specs.frame_to_chk, frame_to_meta = frame_to_meta)
>>> decoder = MetaDecoder(chk_to_frame = specs.chk_to_frame, meta_to_frame = meta_to_frame)
>>> b = encoder(data)
>>> assert decoder(b) == data
```

### *class* recode.base.codec_tuple(encode, decode)

Bases: [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

#### decode

Alias for field number 1

#### encode

Alias for field number 0

### recode.base.frame_to_meta(frame)

Defines header for serialization of tabluar data

```pycon
>>> rows = [{'customer': 1}, {'customer': 2}, {'customer': 3}]
>>> assert frame_to_meta(rows) == b'\x08\x00customer'
```

### recode.base.meta_to_frame(meta)

Deserializes header for deserialization of tabular data

```pycon
>>> meta = b'\x1c\x00customer.apple.banana.tomato\x01\x00\x01\x00\x02\x00\x03\x00\x02\x00\
... x03\x00\x02\x00\x05\x00\x01\x00\x03\x00\x04\x00\t\x00'
>>> assert meta_to_frame(meta)[0] == ['customer', 'apple', 'banana', 'tomato']
```

### recode.base.mk_codec(chk_format='d', n_channels=None, chk_size_bytes=None)

Enable the definition of codec specs based on format characters of the
python struct module
([https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters))

* **Parameters:**
  * **chk_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The format of a chunk, as specified by the struct module
    See [https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters)
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Expected number of channels. If given, will assert that the
    number of channels expressed by the `chk_format` is indeed what is expected.
  * **chk_size_bytes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Expected number of bytes per chunk.
    If given, will assert that the chunk size expressed by the `chk_format` is
    indeed the one expected.
* **Returns:**
  A (named)tuple with encode and decode functions

```pycon
>>> from recode import mk_codec
>>> encoder, decoder = mk_codec()
>>> b = encoder([0, -3, 3.14])
>>> b
b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xc0\x1f\x85\xebQ\xb8\x1e\t@'
>>> decoder(b)
[0.0, -3.0, 3.14]
```

What about those channels?
Well, some times you need to encode/decode multi-channel streams, such as:

```pycon
>>> multi_channel_stream = [[3, -1], [4, -1], [5, -9]]
```

Say, for example, if you were dealing with stereo waveform
(with the standard PCM_16 format), you’d do it this way:

```pycon
>>> encoder, decoder = mk_codec('hh')
>>> pcm_bytes = encoder(iter(multi_channel_stream))
>>> pcm_bytes
b'\x03\x00\xff\xff\x04\x00\xff\xff\x05\x00\xf7\xff'
>>> decoder(pcm_bytes)
[(3, -1), (4, -1), (5, -9)]
```

The `n_channels` and `chk_size_bytes` arguments are there if you want to assert
that your number of channels and chunk size are what you expect.
Again, these are just for verification, because we know how easy it is to
misspecify the `chk_format`, and how hard it can be to notice that we did.

It is advised to use these in any production code, for the sanity of everyone!

```pycon
>>> mk_codec('hhh', n_channels=2)
Traceback (most recent call last):
  ...
AssertionError: You said there'd be 2 channels, but I inferred 3
>>> mk_codec('hhh', chk_size_bytes=3)
Traceback (most recent call last):
  ...
AssertionError: The given chk_size_bytes 3 did not match the inferred (from chk_format) 6
```

Finally, so far we’ve done it this way:

```pycon
>>> encoder, decoder = mk_codec('hHifd')
```

But see that what’s actually returned is a NAMED tuple, which means that you can
can also get one object that will have `.encode` and `.decode` properties:

```pycon
>>> codec = mk_codec('hHifd')
>>> to_encode = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
>>> encoded = codec.encode(to_encode)
>>> decoded = codec.decode(encoded)
>>> decoded
[(1, 2, 3, 4.0, 5.0), (6, 7, 8, 9.0, 10.0)]
```

And you can checkout the properties of your encoder and decoder (they
should be the same)

```pycon
>>> codec.encode.chk_format
'hHifd'
>>> codec.encode.n_channels
5
>>> codec.encode.chk_size_bytes
24
```

### recode.base.mk_encoder_and_decoder(chk_format='d', n_channels=None, chk_size_bytes=None)

Enable the definition of codec specs based on format characters of the
python struct module
([https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters))

* **Parameters:**
  * **chk_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The format of a chunk, as specified by the struct module
    See [https://docs.python.org/3/library/struct.html#format-characters](https://docs.python.org/3/library/struct.html#format-characters)
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Expected number of channels. If given, will assert that the
    number of channels expressed by the `chk_format` is indeed what is expected.
  * **chk_size_bytes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Expected number of bytes per chunk.
    If given, will assert that the chunk size expressed by the `chk_format` is
    indeed the one expected.
* **Returns:**
  A (named)tuple with encode and decode functions

```pycon
>>> from recode import mk_codec
>>> encoder, decoder = mk_codec()
>>> b = encoder([0, -3, 3.14])
>>> b
b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xc0\x1f\x85\xebQ\xb8\x1e\t@'
>>> decoder(b)
[0.0, -3.0, 3.14]
```

What about those channels?
Well, some times you need to encode/decode multi-channel streams, such as:

```pycon
>>> multi_channel_stream = [[3, -1], [4, -1], [5, -9]]
```

Say, for example, if you were dealing with stereo waveform
(with the standard PCM_16 format), you’d do it this way:

```pycon
>>> encoder, decoder = mk_codec('hh')
>>> pcm_bytes = encoder(iter(multi_channel_stream))
>>> pcm_bytes
b'\x03\x00\xff\xff\x04\x00\xff\xff\x05\x00\xf7\xff'
>>> decoder(pcm_bytes)
[(3, -1), (4, -1), (5, -9)]
```

The `n_channels` and `chk_size_bytes` arguments are there if you want to assert
that your number of channels and chunk size are what you expect.
Again, these are just for verification, because we know how easy it is to
misspecify the `chk_format`, and how hard it can be to notice that we did.

It is advised to use these in any production code, for the sanity of everyone!

```pycon
>>> mk_codec('hhh', n_channels=2)
Traceback (most recent call last):
  ...
AssertionError: You said there'd be 2 channels, but I inferred 3
>>> mk_codec('hhh', chk_size_bytes=3)
Traceback (most recent call last):
  ...
AssertionError: The given chk_size_bytes 3 did not match the inferred (from chk_format) 6
```

Finally, so far we’ve done it this way:

```pycon
>>> encoder, decoder = mk_codec('hHifd')
```

But see that what’s actually returned is a NAMED tuple, which means that you can
can also get one object that will have `.encode` and `.decode` properties:

```pycon
>>> codec = mk_codec('hHifd')
>>> to_encode = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
>>> encoded = codec.encode(to_encode)
>>> decoded = codec.decode(encoded)
>>> decoded
[(1, 2, 3, 4.0, 5.0), (6, 7, 8, 9.0, 10.0)]
```

And you can checkout the properties of your encoder and decoder (they
should be the same)

```pycon
>>> codec.encode.chk_format
'hHifd'
>>> codec.encode.n_channels
5
>>> codec.encode.chk_size_bytes
24
```

### recode.base.specs_from_frames(frames)

Implicitly defines the codec specs based on the frames to encode/decode.
specs_from_frames returns a tuple of an iterator of frames and the defined StructCodecSpecs. If
frames is an iterable, then the iterator can be ignored like the following example.

```pycon
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
```

If frames is an iterator, then we can still use specs_from_frames as long as we redefine frames
from the output like in the following example.

```pycon
>>> frames = iter([[1.1,2.2],[3.3,4.4]])
>>> frames, specs = specs_from_frames(frames)
>>> print(specs)
StructCodecSpecs(chk_format='dd', n_channels=2, chk_size_bytes=16)
>>> encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
>>> decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
>>> b = encoder(frames)
>>> decoded_frames = list(decoder(b))
>>> assert decoded_frames == [(1.1,2.2),(3.3,4.4)]
```
