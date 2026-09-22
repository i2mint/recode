# recode.audio

Encoding audio

This module illustrates how one can use recode to make audio codecs.

```pycon
>>> wav_bytes = encode_wav_bytes([1, 2, 3], 42)
>>> wav_bytes
b'RIFF*\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00*\x00\x00\x00T\x00\x00\x00\x02\x00\x10\x00data\x06\x00\x00\x00\x01\x00\x02\x00\x03\x00'
>>> wf, sr = decode_wav_bytes(wav_bytes)
>>> sr
42
>>> wf
[1, 2, 3]
```

The wav codecs are based on pcm codecs along with wav header codecs
(i.e. parsing and generation – using the builtin `wave` package).

Make pcm encoders and decoders:

```pycon
>>> encode, decode = mk_pcm_audio_codec('int16')
>>> encoded = encode([1, 2, 3])
>>> encoded
b'\x01\x00\x02\x00\x03\x00'
>>> decode(encoded)
[1, 2, 3]
```

Or just encode directly:

```pycon
>>> encode_pcm_bytes([1, 2, 3])
b'\x01\x00\x02\x00\x03\x00'
```

Or decode directly:

```pycon
>>> encode_pcm_bytes([1, 2, 3])
b'\x01\x00\x02\x00\x03\x00'
```

### Functions

| [`decode_pcm_bytes`](#recode.audio.decode_pcm_bytes)(pcm_bytes[, width, n_channels])   |                                                                                   |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`decode_wav_bytes`](#recode.audio.decode_wav_bytes)(wav_bytes, \*[, ...])             | Decode WAV bytes into a `(waveform, sample_rate)` pair.                           |
| [`decode_wav_header_bytes`](#recode.audio.decode_wav_header_bytes)(wav_header_bytes)          | Get a dict of params decoded from a wav header                                    |
| [`encode_pcm_bytes`](#recode.audio.encode_pcm_bytes)(wf[, width, n_channels])          | Encode waveform (e.g. list of numbers) into PCM bytes.                            |
| [`encode_wav_bytes`](#recode.audio.encode_wav_bytes)(wf, sr[, width_bytes, ...])       | Encode waveform (e.g. list of numbers) into PCM bytes with WAV header.            |
| [`encode_wav_header_bytes`](#recode.audio.encode_wav_header_bytes)(sr, width_bytes, \*)       | Make a WAV header from given parameters.                                          |
| [`extract_wav_header_from_file`](#recode.audio.extract_wav_header_from_file)(filepath, \*[, ...])  | Extract the header of a WAV file -- everything before the audio -- from its path. |
| [`header_size_of_wav_bytes`](#recode.audio.header_size_of_wav_bytes)(wav_bytes[, meta])        | Size, in bytes, of everything preceding the audio payload.                        |
| [`mk_pcm_audio_codec`](#recode.audio.mk_pcm_audio_codec)([width, n_channels])            | Make a (encoder, decoder) pair for PCM data with given width and n_channels.      |
| [`num_find_num_type_for`](#recode.audio.num_find_num_type_for)(num[, target_num_sys, ...])  | Find the target_num_sys equivalent of input num checking multiple unit options    |
| [`num_type_for`](#recode.audio.num_type_for)(num[, num_sys, target_num_sys])       | Translate from one (sample width) number type to another.                         |

### Exceptions

| [`ShortWavData`](#recode.audio.ShortWavData)   | The `data` chunk carries fewer bytes than its own header declares.   |
|-----------------------------------------------------------------|----------------------------------------------------------------------|

### *exception* recode.audio.ShortWavData

Bases: [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning)

The `data` chunk carries fewer bytes than its own header declares.

Raised as a warning rather than an error because the audio that *is* present is
still worth decoding – a partially downloaded file, or one written to a stream
whose length was never patched back into the header. What must not happen is for
the shortfall to pass unmentioned, since the caller cannot otherwise tell a
truncated file from a complete one.

### recode.audio.decode_pcm_bytes(pcm_bytes, width=2, n_channels=1)

* **Parameters:**
  * **width** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – The width of a sample (in bits, bytes, numpy dtype, pyaudio …)
    (Will try to figure it out)
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of channels
* **Returns:**
  The decoded waveform

```pycon
>>> decode_pcm_bytes(b'\x01\x00\x02\x00\x03\x00')
[1, 2, 3]
```

### recode.audio.decode_wav_bytes(wav_bytes, , eight_bit_unsigned=True)

Decode WAV bytes into a `(waveform, sample_rate)` pair.

* **Parameters:**
  * **wav_bytes** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The bytes of a RIFF/WAVE container holding uncompressed PCM
  * **eight_bit_unsigned** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Read 8-bit audio as the unsigned PCM the WAV spec
    mandates (0..255 on disk, biased to -128..127 here). Pass `False` for the
    pre-recode#12 behaviour, which read those bytes as signed – silence came back
    as -128 – and so round-tripped with `encode_wav_bytes` but with nothing else.
    Widths of 9 bits and up are signed in the spec and are unaffected either way.
* **Returns:**
  `(wf, sr)` – the decoded waveform and its sample rate
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `wav_bytes` is not a RIFF/WAVE container with a `data`
    chunk. (Before recode#4 the same inputs raised `AssertionError`, `wave.Error`
    or `EOFError` depending on how they were malformed; they are unified here.)
  * [**ShortWavData**](#recode.audio.ShortWavData) – *warning*, not an exception – see below.

```pycon
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
```

The `data` chunk is located by walking the RIFF structure, so chunks that sit
*after* the audio – `LIST`/`INFO` metadata, which ffmpeg, Audacity and iTunes
all append – do not shift the waveform:

```pycon
>>> import struct
>>> info = b'INFOISFT' + struct.pack('<I', 6) + b'Lavf58'
>>> with_trailing_metadata = wav_bytes + b'LIST' + struct.pack('<I', len(info)) + info
>>> decode_wav_bytes(with_trailing_metadata)[0]
[0, 1, -1, 2, -2]
```

A file carrying less audio than its header declares decodes to the whole frames
that are actually there, and says so:

```pycon
>>> truncated = wav_bytes[:-4]
>>> import warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     _ = warnings.simplefilter('always')
...     wf, sr = decode_wav_bytes(truncated)
>>> wf
[0, 1, -1]
>>> caught[0].category.__name__
'ShortWavData'
```

8-bit WAV audio is stored *unsigned*, so a byte of 128 is silence rather than full
negative (recode#12). Pass `eight_bit_unsigned=False` to get the old signed reading:

```pycon
>>> eight_bit = encode_wav_bytes([-128, 0, 127], sr=42, width_bytes=1)
>>> eight_bit[44:]
b'\x00\x80\xff'
>>> decode_wav_bytes(eight_bit)[0]
[-128, 0, 127]
>>> decode_wav_bytes(eight_bit, eight_bit_unsigned=False)[0]
[0, -128, -1]
```

### recode.audio.decode_wav_header_bytes(wav_header_bytes)

Get a dict of params decoded from a wav header

For examples, see the `encode_wav_header_bytes` function, it’s inverse.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
>>> from recode.audio import encode_wav_header_bytes
>>> header_bytes = encode_wav_header_bytes(44100, 2, n_channels=3)
>>> decode_wav_header_bytes(header_bytes)
{'sr': 44100,
 'width_bytes': 2,
 'n_channels': 3,
 'nframes': 0,
 'comptype': None}
```

Stdlib `wave` is the primary reader. A `WAVE_FORMAT_EXTENSIBLE` header, which it
refuses before Python 3.12, is parsed directly instead so the same file decodes on
every supported version – see `_extensible_wav_header_params()`.

### recode.audio.encode_pcm_bytes(wf, width=16, n_channels=1)

Encode waveform (e.g. list of numbers) into PCM bytes.

* **Parameters:**
  * **wf** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Number`](https://docs.python.org/3/library/numbers.html#numbers.Number)]) – Waveform to encode
  * **width** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – The width of a sample (in bits, bytes, numpy dtype, pyaudio …)
    (will try to figure it out by itself)
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of channels
* **Returns:**
  The pcm-bytes-encoded waveform

```pycon
>>> encode_pcm_bytes([1, 2, 3])
b'\x01\x00\x02\x00\x03\x00'
```

### recode.audio.encode_wav_bytes(wf, sr, width_bytes=2, n_channels=1, , eight_bit_unsigned=True)

Encode waveform (e.g. list of numbers) into PCM bytes with WAV header.

* **Parameters:**
  * **wf** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Number`](https://docs.python.org/3/library/numbers.html#numbers.Number)]) – Waveform to encode (iterable of numbers)
  * **sr** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sample rate in Hz
  * **width_bytes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The width of a sample in bytes
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of channels
  * **eight_bit_unsigned** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Write 8-bit audio as the unsigned PCM the WAV spec
    mandates (samples biased by 128 into 0..255 on disk). Pass `False` for the
    pre-recode#12 behaviour, which wrote them signed – files that recode read
    back correctly and every other tool did not. Widths of 9 bits and up are
    signed in the spec and are unaffected either way.
* **Returns:**
  The complete WAV file bytes (header + data)
* **Return type:**
  [*bytes*](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### Examples

```pycon
>>> wav_bytes = encode_wav_bytes([0, 1, -1, 2, -2], sr=42)
>>> header_bytes, data_bytes = wav_bytes[:44], wav_bytes[44:]
>>> data_bytes
b'\x00\x00\x01\x00\xff\xff\x02\x00\xfe\xff'
```

See that the header bytes can be decoded to get the right information about our waveform:

```pycon
>>> decode_wav_header_bytes(header_bytes)
{'sr': 42, 'width_bytes': 2, 'n_channels': 1, 'nframes': 5, 'comptype': None}
```

See that our wave_bytes can be decoded to get the original waveform and sample rate:

```pycon
>>> decoded_wf, decoded_sr = decode_wav_bytes(wav_bytes)
>>> decoded_wf
[0, 1, -1, 2, -2]
>>> decoded_sr
42
```

### recode.audio.encode_wav_header_bytes(sr, width_bytes, , n_channels=1, nframes=0, comptype=None)

Make a WAV header from given parameters.

* **Parameters:**
  * **sr** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The sample rate (i.e. “frame rate” i.e. “chk_rate”)
  * **width_bytes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The “sample width” in bytes
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of channels (default is 1)
  * **nframes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – 

    Optional number of frames (default is 0).

    NOTE:
    : If a wav file is to be read correctly, the num of frames (i.e.
      samples/chks) should be exactly the number you’ll actually be writing in the
      wave file.
  * **comptype** – No supported by python’s wave module (yet).
* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

```pycon
>>> header_bytes = encode_wav_header_bytes(44100, 2, n_channels=3)
>>> len(header_bytes)
44
>>> header_bytes[:31]
b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x03\x00D\xac\x00\x00\x98\t\x04'
```

You can decode those params (including those you didn’t specify, but were
defaulted) with the `decode_wav_header_bytes` inverse function.

```pycon
>>> from recode.audio import decode_wav_header_bytes
>>> params = decode_wav_header_bytes(header_bytes)
>>> params
{'sr': 44100,
 'width_bytes': 2,
 'n_channels': 3,
 'nframes': 0,
 'comptype': None}
>>> assert encode_wav_header_bytes(**params) == header_bytes
```

### recode.audio.extract_wav_header_from_file(filepath, , read_size=65536)

Extract the header of a WAV file – everything before the audio – from its path.

Useful for reading the header of a WAV file without having to read the entire file
into memory, which is what you want when WAV files are large and/or numerous: only
as much of the file as the header occupies is read.

The answer is the same one [`header_size_of_wav_bytes()`](#recode.audio.header_size_of_wav_bytes) gives for the same
bytes, because both locate the `data` chunk by walking the RIFF structure rather
than inferring where it must be. This used to compute
`chunk_size + 8 - subchunk2_size`, reading bytes 40-44 as the size of the audio –
true only of a bare 44-byte header. With a `LIST`/`INFO` chunk after the audio (what
ffmpeg, Audacity and iTunes write) it returned audio bytes as header; with one
before, a truncated prefix that parses as nothing at all (recode#4).

* **Parameters:**
  * **filepath** – The path to the WAV file
  * **read_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – How many bytes to read at a time while looking for the audio
* **Returns:**
  The bytes of the WAV file header, i.e. everything preceding the `data`
  chunk’s contents
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the file is not a RIFF/WAVE container with a reachable `data`
  chunk. (It used to answer such files with arbitrary bytes and no complaint.)

### recode.audio.header_size_of_wav_bytes(wav_bytes, meta=None)

Size, in bytes, of everything preceding the audio payload.

That is the offset of the `data` chunk’s contents, found by walking the RIFF
structure (see `_wav_data_chunk()`). For a well-formed file with nothing after
the audio this is the same number the old size-subtraction produced; unlike it, it
stays correct when the file carries trailing metadata or an over-declared `data`
size.

`meta` – an already-decoded header, once passed to save re-parsing it – is
accepted for backwards compatibility and no longer used.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

```pycon
>>> header_size_of_wav_bytes(
...     b'RIFF.\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'
...     b'*\x00\x00\x00T\x00\x00\x00\x02\x00\x10\x00data\n\x00\x00\x00'
...     b'\x00\x00\x01\x00\xff\xff\x02\x00\xfe\xff'
... )
44
```

### recode.audio.mk_pcm_audio_codec(width=16, n_channels=1)

Make a (encoder, decoder) pair for PCM data with given width and n_channels.

PCM data is what’s used in the uncompressed raw WAVE formats (such as used in CDs).
See [https://en.wikipedia.org/wiki/Pulse-code_modulation](https://en.wikipedia.org/wiki/Pulse-code_modulation).

* **Parameters:**
  * **width** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – The width of a sample (in bits, bytes, numpy dtype, pyaudio …)
    (Will try to figure it out)
  * **n_channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of channels
* **Returns:**
  A (encoder, decoder) pair of functions that are inverse of each other

```pycon
>>> encode, decode = mk_pcm_audio_codec('int16')
>>> encoded = encode([1, 2, 3])
>>> encoded
b'\x01\x00\x02\x00\x03\x00'
>>> decode(encoded)
[1, 2, 3]
```

Let’s check over more combinations of width and n_channels that we can decode
what we encode to get back the same thing:

```pycon
>>> wf = [-3, -2, -1, 0, 1, 2, 3]
>>> for width in [16, 2, 'int16', 'paInt16', 'PCM_16', 32, 4, 'int32']:
...     for channel in wf:
...         encode, decode = mk_pcm_audio_codec('int16')
...         encoded = encode(wf)
...         assert isinstance(encoded, bytes)
...         assert decode(encoded) == wf
```

### recode.audio.num_find_num_type_for(num, target_num_sys='struct', num_sys_search_order=('n_bits', 'n_bytes', 'dtype', 'pyaudio', 'soundfile'))

Find the target_num_sys equivalent of input num checking multiple unit options

### recode.audio.num_type_for(num, num_sys='n_bits', target_num_sys='struct')

Translate from one (sample width) number type to another.

* **Parameters:**
  * **num**
  * **num_sys**
  * **target_num_sys**
* **Returns:**

```pycon
>>> num_type_for(16, "n_bits", "soundfile")
'PCM_16'
>>> num_type_for(3, "n_bytes", "soundfile")
'PCM_24'
```

#### TIP
Use with `functools.partial` when you have some fix translation endpoints.

```pycon
>>> from functools import partial
>>> get_dtype_from_n_bytes = partial(
...     num_type_for, num_sys="n_bytes", target_num_sys="dtype"
... )
>>> get_dtype_from_n_bytes(8)
'float64'
```
