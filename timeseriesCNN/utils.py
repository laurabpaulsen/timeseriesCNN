'''
utils (type checking)
'''

def _check_valid_types(inputted_types=list, valid_types=list):
    '''
    check if the content of inputted_types is valid
    '''
    for inputted_type in inputted_types:
        if inputted_type not in valid_types:
            raise ValueError(f'{inputted_type} is not valid. Valid values are {valid_types}')