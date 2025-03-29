
# These settings are used in the functions

def days_type():
    '''
    t: trading days
    c: calendar days
    '''
    type = 'c' #choose between 't' and 'c'
    return type + '_' # todo: move '_' to the functions that call this one


def clean_t_days():
    '''
    True: If days_type() = t, then clean using trading days
    False: If days_type() = t, then still clean using calendar days
    if days_type() = c, then clean using calendar days regardless of clean_t_days()
    '''
    return True

