def mytostring(x, delimiter='-', hashlen=float('NaN')):
    """
    Relatively general function for converting an input to a string that
    represents its content. Used by struct2string.
    """
    if x is None:
        s = 'NULL'
    elif isinstance(x, str):
        s = x
    elif (isinstance(x, (int, float)) or isinstance(x, bool)) and isinstance(x, (int, float, str)):
        s = '-'.join(map(str, x))
        if not math.isnan(hashlen) and len(s) > hashlen:
            s = 'DataHash(x)'  # Replace this line with your DataHash function
    elif isinstance(x, list):
        s = delimiter.join(mytostring(i) for i in x)
    else:
        raise ValueError('No valid type')

    return s
