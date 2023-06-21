import numpy as np
import DataHash
def mytostring(x, delimiter='-', hashlen=np.nan):
    """
    Numpy version
    -------------
    Relatively general function for converting an input to a string that represents its content. Used by struct2string.

    Example:
   
    print(mytostring([2.33, 3, 4, 1, 'test']))

    """

    if not x:
        s = 'NULL'
    elif isinstance(x, str):
        s = x
    elif (np.isscalar(x) or isinstance(x, list)) and any(isinstance(x, t) for t in (int, float, complex, bool, np.number)):
        # Handle scalar and vector
        if isinstance(x, list):
            # Vector
            s = '-'.join(map(str, x))
        else:
            # Scalar
            s = str(x)
        if not np.isnan(hashlen) and len(s) > hashlen:
            s = DataHash(x)
    elif isinstance(x, list):
        # Handle cell array
        s = delimiter.join(mytostring(i, delimiter=delimiter, hashlen=hashlen) for i in x)
    else:
        raise ValueError('No valid type')

    return s

if __name__ == '__main__':

    print(mytostring([2.33, 332323232, 4, 1, 'test']))
