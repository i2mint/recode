
# recode
Make codecs for fixed size structured chunks serialization and deserialization of
sequences, tabular data, and time-series.

To install:	```pip install recode```

[Docs here](https://otosense.github.io/recode/module_docs/recode.html)

Make codecs for fixed size structured chunks serialization and deserialization of
sequences, tabular data, and time-series.

The easiest and bigest bang for your buck is ``mk_codec``

```python
>>> from recode import mk_codec
>>> encoder, decoder = mk_codec()
```

``encoder`` will encode a list (or any iterable) of numbers into bytes

```python
>>> b = encoder([0, -3, 3.14])
>>> b
b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xc0\x1f\x85\xebQ\xb8\x1e\t@'

```

``decoder`` will decode those bytes to get you back your numbers

```python
>>> decoder(b)
[0.0, -3.0, 3.14]
```

There's only really one argument you need to know about in ``mk_codec``.
The first argument, called `chk_format`, which is a string of characters from
the "Format" column of the python 
[format characters](https://docs.python.org/3/library/struct.html#format-characters)

The length of the string specifies the number of "channels",
and each individual character of the string specifies the kind of encoding you should
apply to each "channel" (hold your horses, we'll explain).

The one we've just been through is in fact

```python
>>> encoder, decoder = mk_codec('d')
```

That is, it will expect that your data is a list of numbers, and they'll be encoded
with the 'd' format character, that is 8-bytes doubles.
That default is goo because it gives you a lot of room, but if you knew that you
would only be dealing with 2-byte integers (as in most WAV audio waveforms),
you would have chosen `h`:

```python
>>> encoder, decoder = mk_codec('h')
```

What about those channels?
Well, some times you need to encode/decode multi-channel streams, such as:

```python
>>> multi_channel_stream = [[3, -1], [4, -1], [5, -9]]
```

Say, for example, if you were dealing with stereo waveform
(with the standard PCM_16 format), you'd do it this way:

```python
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

```python
>>> mk_codec('hhh', n_channels=2)
Traceback (most recent call last):
  ...
AssertionError: You said there'd be 2 channels, but I inferred 3
>>> mk_codec('hhh', chk_size_bytes=3)
Traceback (most recent call last):
  ...
AssertionError: The given chk_size_bytes 3 did not match the inferred (from chk_format) 6
```

Finally, so far we've done it this way:


```python
>>> encoder, decoder = mk_codec('hHifd')
```

But see that what's actually returned is a NAMED tuple, which means that you can
can also get one object that will have `.encode` and `.decode` properties:

```python
>>> codec = mk_codec('hHifd')
>>> to_encode = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
>>> encoded = codec.encode(to_encode)
>>> decoded = codec.decode(encoded)
>>> decoded
[(1, 2, 3, 4.0, 5.0), (6, 7, 8, 9.0, 10.0)]
```

And you can checkout the properties of your encoder and decoder (they
should be the same)

```python
>>> codec.encode.chk_format
'hHifd'
>>> codec.encode.n_channels
5
>>> codec.encode.chk_size_bytes
24
```


# Further functionality and under-the-hood peeps

This section shows various examples of recode and it how it can be used with:
- Single channel numerical streams
- Multi-channel numerical streams
- DataFrames
- Iterators


```python
from recode import (ChunkedEncoder, 
                    ChunkedDecoder, 
                    MetaEncoder, 
                    MetaDecoder, 
                    IterativeDecoder, 
                    StructCodecSpecs, 
                    specs_from_frames,
                    frame_to_meta,
                    meta_to_frame)
```

## Quick run through

First define the frame you want to encode


```python
frame = [1,2,3]
```

Next define the specifications for that encoding


```python
specs = StructCodecSpecs(chk_format='h')
specs
```




    StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)



Next define an encoder and encode your frame to bytes


```python
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
b = encoder(frame)
b
```




    b'\x01\x00\x02\x00\x03\x00'



Once you need your original frame again, define a decoder and decode your frame


```python
decoder = ChunkedDecoder(specs.chk_to_frame)
decoded_frames = list(decoder(b))
decoded_frames
```




    [1, 2, 3]



## Step-by-step explanation

### Define your StructCodecSpecs

StructCodecSpecs is used to define the specs for making codecs for fixed size structured chunks serialization and deserialization of sequences, tabular data, and time-series. This definition is based on format strings of the [python struct module](https://docs.python.org/3/library/struct.html#format-strings).

There are two ways to define StructCodecSpecs, the first being to explicitly 
define it using the StructCodecSpecs class. 
StructCodecSpecs takes three arguments: `chk_format`, `n_channels`, 
and `chk_size_bytes`. 

Only `chk_format` is required, 
as `n_channels` and `chk_size_bytes` can be determined based on `chk_format`. 
If `n_channels` or  `chk_size_bytes` are given, they will be used to 
assert that the values inferred from `chk_format` match.

If this is not the case, then do not provide an argument for `n_channels` 
and instead pass a string with the format character matching the data type for 
each channel in the frame to `chk_format`. 
For example if the first channel 
contains integers and the second contains floats, then `chk_format = 'hd'`.


```python
frame = [1,2,3]
specs = StructCodecSpecs(chk_format='h')
specs
```




    StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)



The second way to define StructCodecSpecs is to use `specs_from_frames` which will implictly define StructCodecSpecs based on the frame that is going to be encoded/decoded. This function will return a tuple containing an iterator of the passed frame and the defined StructCodecSpecs. The first element of the tuple can be ignored if the frame passed is not an iterator.


```python
_, specs = specs_from_frames(frame)
specs
```




    StructCodecSpecs(chk_format='h', n_channels=1, chk_size_bytes=2)



If frame is an iterator, then redefine frame as the first argument of the tuple so the first element of frame is not lost for encoding.


```python
frame = iter([[1,2], [3,4]])
frame, specs = specs_from_frames(frame)
specs, list(frame)
```




    (StructCodecSpecs(chk_format='hh', n_channels=2, chk_size_bytes=4),
     [[1, 2], [3, 4]])



### Define your Encoder

Your Encoder will allow you to, you guessed it, encode your frames! There are two Encoders currently defined in recode: `ChunkedEncoder` and `MetaEncoder`.

`ChunkedEncoder` should be your goto encoder for sequences, while `MetaEncoder` works best for tabular data (currently must be in the format of list of dicts).


```python
frame = [1,2,3]
_, specs = specs_from_frames(frame)
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
b = encoder(frame)
b
```

    b'\x01\x00\x02\x00\x03\x00'



When using a `MetaEncoder`, an extra argument named frame_to_meta is required, which can be easily imported from recode!


```python
frame = [{'foo': 1, 'bar': 1}, {'foo': 2, 'bar': 2}, {'foo': 3, 'bar': 4}]
_, specs = specs_from_frames(frame)
encoder = MetaEncoder(frame_to_chk=specs.frame_to_chk, frame_to_meta=frame_to_meta)
b = encoder(frame)
b
```

    b'\x07\x00foo.bar\x01\x00\x01\x00\x02\x00\x02\x00\x03\x00\x04\x00'



### Define your Decoder

Next up your Decoder will allow you to decode your encoded bytes. There are three Encoders currently defined in recode: `ChunkedDecoder`, `IterativeDecoder`, and `MetaDecoder`.

Either `ChunkedDecoder` or `IterativeDecoder` will work well for sequences, with the only difference being that `IterativeDecoder` returns an iterator of decoded chunks while `ChunkedDecoder` returns a mapping which can be converted to an iterable. `MetaDecoder` works best for tabular data encoded with `MetaEncoder`. 


```python
frame = [1,2,3]
_, specs = specs_from_frames(frame)
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
decoder = ChunkedDecoder(specs.chk_to_frame)
b = encoder(frame)
list(decoder(b))
```

    [1, 2, 3]



As is shown in the following example, an `IterativeDecoder` will return an unpack_iterator which which can easily be converted to a standard iterable like a list, or be used as an iterator.


```python
frame = [[1,1],[2,2]]
_, specs = specs_from_frames(frame)
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
decoder = IterativeDecoder(chk_to_frame=specs.chk_to_frame)
b = encoder(frame)
iter_frames = decoder(b)
print(type(iter_frames))
next(iter_frames), next(iter_frames)
```

    <class 'unpack_iterator'>
    ((1, 1), (2, 2))



When using a `MetaDecoder`, an extra argument named meta_to_frame is required, which can be easily imported from recode!


```python
frame = [{'foo': 1, 'bar': 1}, {'foo': 2, 'bar': 2}, {'foo': 3, 'bar': 4}]
_, specs = specs_from_frames(frame)
encoder = MetaEncoder(frame_to_chk=specs.frame_to_chk, frame_to_meta=frame_to_meta)
decoder = MetaDecoder(chk_to_frame=specs.chk_to_frame, meta_to_frame=meta_to_frame)
b = encoder(frame)
decoded_frames = decoder(b)
decoded_frames
```

    [{'foo': 1, 'bar': 1}, {'foo': 2, 'bar': 2}, {'foo': 3, 'bar': 4}]



## Encoding a waveform

### Create a synthetic waveform using `hum`


```python
from hum.gen.sine_mix import freq_based_stationary_wf
import matplotlib.pyplot as plt
import numpy as np

DFLT_N_SAMPLES = 21 * 2048
DFLT_SR = 44100
```


```python
wf_mix = freq_based_stationary_wf(freqs=(200, 400, 600, 800), weights=None,
                             n_samples = DFLT_N_SAMPLES, sr = DFLT_SR)
plt.plot(wf_mix[:300]);
```


    
![png](notebooks/recode_demo_files/recode_demo_35_0.png)
    



```python
wf_mix
```




    array([0.        , 0.07114157, 0.14170624, ..., 0.49480205, 0.54853009,
           0.5979443 ])



### Encode the wf


```python
specs = StructCodecSpecs('d')
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
b = encoder(wf_mix)
b[:100]
```




    b'\x00\x00\x00\x00\x00\x00\x00\x00\x1b:\x11\x8dU6\xb2?\xb8!\xa2\x15n#\xc2?\xe6I\xedz\x15\x06\xcb?L;\xe6\xfah\xd8\xd1?}\x8c\x9b\x9a\xf5\x08\xd6?D\xcc\xf1\x897\x0c\xda?\xe3\xc5\xbeb/\xda\xdd?DE\xb6K\xb6\xb5\xe0?\x82|\xfcz\x90\\\xe2?\xae3:$\x99\xde\xe3?\x1e\x95\xbe\xef$9\xe5?\x81\xd7\xdc\xec'



### Decode and compare to wf_mix


```python
decoded_wf_mix = decoder(b)
plt.plot(decoded_wf_mix[:300]);
```


    
![png](notebooks/recode_demo_files/recode_demo_40_0.png)
    



```python
np.all(decoded_wf_mix == wf_mix)
```




    True



## Encoding a pandas dataframe


```python
import pandas as pd
```

### Create/import your dataframe


```python
df = pd.DataFrame(data = [[1,2,3],[4,5,6]], columns = ['foo', 'bar', 'set'])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>bar</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### Prep the dataframe for encoding


```python
frame = df.to_dict('records')
frame
```




    [{'foo': 1, 'bar': 2, 'set': 3}, {'foo': 4, 'bar': 5, 'set': 6}]



### Encode the list of dicts


```python
_, specs = specs_from_frames(frame)
encoder = MetaEncoder(frame_to_chk=specs.frame_to_chk, frame_to_meta=frame_to_meta)
decoder = MetaDecoder(chk_to_frame=specs.chk_to_frame, meta_to_frame=meta_to_frame)
b = encoder(frame)
b
```




    b'\x0b\x00foo.bar.set\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00'




```python
decoded_frame = decoder(b)
decoded_frame
```




    [{'foo': 1, 'bar': 2, 'set': 3}, {'foo': 4, 'bar': 5, 'set': 6}]




```python
decoded_df = pd.DataFrame(decoded_frame)
print(np.all(decoded_df == df))
decoded_df
```

    True





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>bar</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## Miscellaneous information

### Byte order, Size, and Alignment

This table provides the characters associated with Byte order, Size, and Alignment. If one of these characters is not given as the first character of the format string, then `@` will be assumed. More information about Byte order, Size, and Alignment can be found [here](https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment).

|Character|Byte order|Size|Alignment|
|---------|----------|----|---------|
|@|native|native|native|
|=|native|standard|none|
|<|little-endian|standard|none|
|>|big-endian|standard|none|
|!|network (=big-endian)|standard|none|


# Full Examples

## Single channel numerical stream

```python
from recode import StructCodecSpecs, ChunkedEncoder, ChunkedDecoder
specs = StructCodecSpecs(chk_format='h')
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
frames = [1, 2, 3]
b = encoder(frames)
assert b == b'\x01\x00\x02\x00\x03\x00'
decoded_frames = list(decoder(b))
assert decoded_frames == frames
```


## Multi-channel numerical stream
```python
from recode import StructCodecSpecs, ChunkedEncoder, ChunkedDecoder
specs = StructCodecSpecs(chk_format='@hh', n_channels = 2)
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
frames = [(1, 2), (3, 4), (5, 6)]
b = encoder(frames)
assert b == b'\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00'
decoded_frames = list(decoder(b))
assert decoded_frames == frames
```


## Iterative decoder
```python
from recode import StructCodecSpecs, ChunkedEncoder, IterativeDecoder
specs = StructCodecSpecs(chk_format = 'hdhd')
encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
decoder = IterativeDecoder(chk_to_frame = specs.chk_to_frame)
frames = [(1,1.1,1,1.1),(2,2.2,2,2.2),(3,3.3,3,3.3)]
b = encoder(frames)
iter_frames = decoder(b)
assert next(iter_frames) == frames[0]
next(iter_frames)
```    

	(2, 2.2, 2, 2.2)



## DataFrame (as list of dicts) using MetaEncoder/MetaDecoder
```python
from recode import StructCodecSpecs, MetaEncoder, MetaDecoder, frame_to_meta, meta_to_frame
data = [{'foo': 1.1, 'bar': 2.2}, {'foo': 513.23, 'bar': 456.1}, {'foo': 32.0, 'bar': 6.7}]
specs = StructCodecSpecs(chk_format='dd', n_channels = 2)
encoder = MetaEncoder(frame_to_chk = specs.frame_to_chk, frame_to_meta = frame_to_meta)
decoder = MetaDecoder(chk_to_frame = specs.chk_to_frame, meta_to_frame = meta_to_frame)
b = encoder(data)
assert decoder(b) == data
```


## Implicitly define codec specs based on frames

```python
from recode import specs_from_frames, ChunkedEncoder, ChunkedDecoder
frames = [1,2,3]
_, specs = specs_from_frames(frames)
encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
b = encoder(frames)
assert b == b'\x01\x00\x02\x00\x03\x00'
decoded_frames = list(decoder(b))
decoded_frames
```

	[1, 2, 3]


## Implicit definition of iterator
```python
from recode import specs_from_frames, ChunkedEncoder, IterativeDecoder
frames = iter([[1.1,2.2],[3.3,4.4]])
frames, specs = specs_from_frames(frames)
encoder = ChunkedEncoder(frame_to_chk = specs.frame_to_chk)
decoder = ChunkedDecoder(chk_to_frame=specs.chk_to_frame)
b = encoder(frames)
decoded_frames = list(decoder(b))
assert decoded_frames == [(1.1,2.2),(3.3,4.4)]
```

# More


## Example of recode functionality to read and write audio files

In the below example we can see that the functionality of reading and writing audio to bytes is replicated in `recode`. 
An example of a waveform audio file is first created using `mk_wf`. 
Then, that example wave form is converted to bytes using `BytesIO` and `soundfile.write`. 
Then, that same example wave form is converted to bytes using `recode` functionality. 
Finally it can be seen through the assertions that `soundfile`'s read/write and `recode`'s encode/decode provide 
the same functionality for audio files (aside from the header in `soundfile` as a result of the .wav format).

```python
import soundfile as sf
from io import BytesIO
from enum import Enum
import numpy as np
from recode import ChunkedDecoder, ChunkToFrame, ChunkedEncoder, StructCodecSpecs

class Kind(Enum):
    random = 'random'
    increasing = 'increasing'

def mk_wf(n_samples=2048, kind: Kind=Kind.random, **kwargs):
    if kind == Kind.random:
        dtype_str = kwargs.get('num_type', 'int16')
        if dtype_str.startswith('int'):
            low = kwargs.get('low', -30000)
            high = kwargs.get('high', 30000)
            wf = np.random.randint(low=low, high=high, size=n_samples, dtype=dtype_str)
        else:
            raise TypeError("Don't know how to handle this case")
    else:
        raise TypeError("Don't know how to handle this case")
    return wf

wf = mk_wf(25)
sr = 10000

file = BytesIO()
sf.write(file, wf, sr, format = 'WAV')
file.seek(0)
b = file.read()

specs = StructCodecSpecs('h')
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
decoder = ChunkedDecoder(chk_size_bytes=specs.chk_size_bytes, 
                         chk_to_frame=specs.chk_to_frame, 
                         n_channels=specs.n_channels)
d = encoder(wf)

assert d == b[44:]

sf_read = sf.read(BytesIO(b), dtype='int16')
recode_read = list(decoder(d))

assert np.all(sf_read[0] == recode_read)
assert np.all(recode_read == wf)
assert np.all(sf_read[0] == wf)
```

## Compare performance of IterativeDecoder and ChunkedDecoder

Both `IterativeDecoder` and `ChunkedDecoder` work well with encoded sequences. 
Additionally, an `IterativeDecoder` can replace the functionality of a `ChunkedDecoder` as the iterator 
it returns can easily be converted to a list. 
Because of this we will compare the efficiency of `IterativeDecoder` and `ChunkedDecoder` to see if we should replace the 
functionality of `ChunkedDecoder` with `IterativeDecoder`.

```python
from hum.gen.sine_mix import freq_based_stationary_wf
from recode import (ChunkedEncoder, 
                    ChunkedDecoder, 
                    IterativeDecoder, 
                    StructCodecSpecs)
```

Create a synthetic waveform to test both decoders on:

```python
DFLT_N_SAMPLES = 21 * 2048
DFLT_SR = 44100

wf_mix = freq_based_stationary_wf(freqs=(200, 400, 600, 800), weights=None,
                             n_samples = DFLT_N_SAMPLES*300, sr = DFLT_SR)
```

Define the decoders and encode the waveform:

```python
specs = StructCodecSpecs('d')
encoder = ChunkedEncoder(frame_to_chk=specs.frame_to_chk)
decoder = ChunkedDecoder(
    chk_size_bytes=specs.chk_size_bytes,
    chk_to_frame=specs.chk_to_frame,
    n_channels=specs.n_channels
)
idecoder = IterativeDecoder(chk_to_frame = specs.chk_to_frame_iter)
b = encoder(wf_mix)
```

Helper function so the output of the decoders are in the same format:

```python
def it_to_list(it):
    frame = list(it)
    frame = [item for tup in frame for item in tup]
    return frame
```

Performance of `ChunkedDecoder`:

```python
%timeit -r 2 -n 5 list(decoder(b))
```
```
5.59 s ± 14.5 ms per loop (mean ± std. dev. of 2 runs, 5 loops each)
```

Performance of `IterativeDecoder`:

```python
%timeit -r 2 -n 5 it_to_list(idecoder(b))
```
```
2.04 s ± 13.4 ms per loop (mean ± std. dev. of 2 runs, 5 loops each)
```

From these results, we can clearly see that we should replace the functionality of `ChunkedDecoder` with `IterativeDecoder` 
as it is nearly three times more efficient.
