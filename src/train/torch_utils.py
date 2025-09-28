class Parameters(dict): 
    og_getattr = dict.__getitem__
    og_setattr = dict.__setitem__

    def __getattr__(self, x):
        try:
            res = self.og_getattr(x.lower()) 
            return res
        except KeyError:
            raise AttributeError(x)

    def __setattr__(self, x, v):
        return self.og_setattr(x.lower(), v)

"""
class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params
    
    def __getattr__(self, x):
        if x == 'params': 
            return self
        try:
            res = self.params[x.lower()]
            return res
        except KeyError:
            raise AttributeError(x)
"""