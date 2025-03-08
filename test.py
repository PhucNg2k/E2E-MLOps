from box import ConfigBox, BoxError
from ensure import ensure_annotations, EnsureError

@ensure_annotations
def add( x: int, y: int):
    return x + y



add(3, 3)

try:
    add( 3, 3.2)
except EnsureError as e:
    print(e)


d = {
    "key": 123,
    "key1": 456
}

print(d['key'])

d = ConfigBox(d)
try:    
    print(d.key)
    print(d.key2)
except BoxError as e:
    print(e)
    raise AttributeError("key is not in dict")
except AttributeError as e:
    print(e)
