import mytostring
import parse_optInputs_keyvalue

def struct2string(S, maxlen=100, delimiter='', omit_field=[], include_fields=None, **varargin):

    '''
    Creates a string the fields / values in the input struct. Useful for
    parameter handling / saving.
    
    Example:
    --------
    S = {}
    S['a'] = 'doowicky'
    S['b'] = [3.14, 42]
    print(struct2string(S))
    print(struct2string(S, maxlen=3))
    print(struct2string(S, maxlen=3, delimiter='/'))
    '''
    
    P = dict(maxlen=maxlen, delimiter=delimiter, omit_field=omit_field, include_fields=include_fields)
    
    P, _ = parse_optInputs_keyvalue(varargin, P, noloop=True)

    idstring = ['']

    if P['include_fields']:
        f = [x for x in P['include_fields'] if x in S.keys()]
    else:
        f = list(S.keys())
    
    for i in range(len(f)):
        omit = len(P['omit_field']) > 0 and f[i] in P['omit_field']
        
        if not omit:
            str_to_add = f'{f[i]}-{mytostring(S[f[i]], hashlen=P["maxlen"]-len(f[i])-1)}'
            
            if len(idstring[-1]) == 0:
                idstring[-1] = str_to_add
            else:
                if len(idstring[-1]) + len(str_to_add) + 1 > P['maxlen']:
                    idstring.append(str_to_add)
                else:
                    idstring[-1] = f'{idstring[-1]}_{str_to_add}'
                    
    if len(P['delimiter']) > 0:
        s = idstring[0]
        for i in range(1, len(idstring)):
            s = f'{s}{P["delimiter"]}{idstring[i]}'
        idstring = s

    if isinstance(idstring, list) and len(idstring) == 1:
        idstring = idstring[0]
        
    return idstring
