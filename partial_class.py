from functools import partialmethod
__all__ = ['partial_class']

def partial_class(cls, *args, **kwargs):
    class PartialClass(cls):
        source_class = cls
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return PartialClass