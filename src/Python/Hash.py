# Hashing function for signed integers, since Processing does not have uint. This way, I can directly use the hash function in Processing.

def hash(x):
    x ^= (x >> 31)

    x = (x * 0x45d9f3b) & 0x7FFFFFFF
    x ^= (x >> 16)
    x = (x * 0x45d9f3b) & 0x7FFFFFFF
    x ^= (x >> 16)

    return x
