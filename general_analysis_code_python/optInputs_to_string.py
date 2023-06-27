import parse_optInputs_keyvalue
import struct2string

def optInputs_to_string(I, C_value, always_include, always_exclude, **varargin):
    """
    Takes two of the outputs of parse_optInputs_keyvalue and turns them into a string used to save the results of an analysis.
    
    By default, only variables that are changed are included in the string,
    unless that variable is included in the list "always_include".
    
    Variables can be excluded even if they are changed by specifying them in
    "always_exclude".
    
    -- Example --
    
    I = {'a': 'TCI', 'b': [1,2,3], 'c': ['hello','world']}
    varargin = ['a', 'quilting', 'c', ['goodbye','world']]
    I, C_value = parse_optInputs_keyvalue(varargin, I)
    always_include = ['b']
    always_exclude = ['a']
    str = optInputs_to_string(I, C_value, always_include, always_exclude)
    
    """
    # print(varargin)
    # print(type(varargin))
    # exit()
    # Initialize default parameters
    P = {'maxlen': 100, 'delimiter': '/'}
    # P = parse_optInputs_keyvalue.parse_optInputs_keyvalue(varargin, P, 'noloop')

    # Check there is no overlap between fields to include and exclude
    overlapping_fields = set(always_include).intersection(always_exclude)
    if overlapping_fields:
        error_msg = '\n'.join(overlapping_fields)
        raise ValueError(f'The following fields are present in both always_include and always_exclude:\n{error_msg}')

    # Remove fields to always exclude from string
    for field in always_exclude:
        if field not in I:
            raise ValueError(f'{field} cannot be excluded because it is not a possible field')
        I.pop(field, None)
        if field in C_value:
            C_value.pop(field, None)

    # Include fields that should always be included
    Z = {}
    for field in always_include:
        if field not in I:
            raise ValueError(f'{field} cannot be always included because it is not a possible field')
        Z[field] = I[field]

    # Include additional fields that have been changed
    f = set(C_value.keys())
    for field in f:
        if field not in always_include:
            Z[field] = C_value[field]


    # Convert to string
    result = struct2string.struct2string(Z, maxlen=P['maxlen'], delimiter=P['delimiter'])

    
    return result

# Test code
if __name__ == '__main__':
    I = {'a': 'TCI', 'b': [1,2,3], 'c': ['hello','world']}
    varargin = ['a', 'quilting', 'c', ['goodbye','world']]
    # 
    I,_,C_value,_,_ = parse_optInputs_keyvalue.parse_optInputs_keyvalue(varargin, I)
    always_include = ['b']
    always_exclude = ['a']
    result = optInputs_to_string(I, C_value, always_include, always_exclude)
    print(result)