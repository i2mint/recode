# recode.util

Utils for use throughout the package

### Functions

| [`get_struct`](#recode.util.get_struct)(str_type)      |    |
|----------------------------------------------------------------------------|----|
| [`list_of_dicts`](#recode.util.list_of_dicts)(cols, vals) |    |
| [`spy`](#recode.util.spy)(iterable[, n])        |    |
| [`take`](#recode.util.take)(n, iterable)         |    |

### recode.util.get_struct(str_type)

```pycon
>>> assert get_struct(type(1)) == 'h'
>>> assert get_struct(type(1.001)) == 'd'
```

### recode.util.list_of_dicts(cols, vals)

```pycon
>>> cols = ['foo', 'bar']
>>> vals = [[1,2], [3,4], [5,6]]
>>> list_of_dicts(cols, vals)
[{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}, {'foo': 5, 'bar': 6}]
```

### recode.util.spy(iterable, n=1)

```pycon
>>> peek, it = spy([1,2,3], 1)
>>> assert peek == [1]
>>> assert next(it) == 1
>>> assert list(it) == [2,3]
```

### recode.util.take(n, iterable)

```pycon
>>> assert take(3, [1,2,3,4,5]) == [1,2,3]
```
