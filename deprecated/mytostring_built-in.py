import DataHash
import math

def mytostring(x, delimiter='-', hashlen=float('NaN')):
    """
    Built-in version
    -------------
    Relatively general function for converting an input to a string that represents its content. Used by struct2string.

    Example:
   
    print(mytostring([2.33, 3, 4, 1, 'test']))

    """
    if x is None:
        s = 'NULL'
    elif isinstance(x, str):
        s = x
    elif (isinstance(x, (int, float)) or isinstance(x, bool)) and isinstance(x, (int, float, str)):
        s = '-'.join(map(str, x))
        if not math.isnan(hashlen) and len(s) > hashlen:
            s = DataHash(x)
    elif isinstance(x, list):
        s = delimiter.join(mytostring(i) for i in x)
    else:
        raise ValueError('No valid type')

    return s


if __name__ == '__main__':
    print(mytostring([2.33, 432432, 4, 1, 'test']))