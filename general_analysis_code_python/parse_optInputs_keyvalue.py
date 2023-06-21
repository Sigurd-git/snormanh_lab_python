def parse_optInputs_keyvalue(optargs:dict, I:dict=None, *varargin ):
    '''
    I: dictionary, key-value pairs, which should be the default parameters.
    optargs: list, key-value pairs, which should be the updated parameters.
    varargin: list, optional arguments to control the behavior of this function. TODO

    return:
    I: dictionary, key-value pairs, which are the merged parameters.
    C: dictionary, key-value pairs, which are the parameters whose values indicating they are changed or not.
    C_values: list, values of the parameters.
    all_keys: list, all keys in I.      
    parameter_strings: str, string representation of the parameters.      TODO

    '''
    if I is None:
        I = optargs
        C = {}
        C_values = []

    else:
        C = {key:False for key in I.keys()}
        C_values = []
        for key, value in optargs.items():
            if key in I:
                if I[key] != value:
                    C[key] = True
                    I[key] = value
                    C_values.append(value)
                
            else:
                raise ValueError(f'key {key} does not exist in I')
    

    all_keys = list(I.keys())

    return I, C, C_values, all_keys,''


    



if __name__ == '__main__':

    # parse key-value pairs
    optargs = {'key1': [1, 2, 3], 'key2': list(range(1, 4)), 'key3': 'abc'}
    I, C,_,_,_ = parse_optInputs_keyvalue(optargs)

    # specifying default values
    I = {'key1': [1, 2, 3], 'key2': list(range(1, 4)), 'key3': 'abc'}
    I, C,_,_,_ = parse_optInputs_keyvalue({'key1': [4, 5, 6]}, I)


    # use defaults to catch a spelling mistake
    I,_,_,_,_ = parse_optInputs_keyvalue({'keys1': [4, 5, 6]}, I)
