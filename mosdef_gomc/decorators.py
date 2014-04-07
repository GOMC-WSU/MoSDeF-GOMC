import mbuild.unit as units

#=============================================================================
# EXCEPTIONS
#=============================================================================


class UnitsException(Exception):
    """Exception denoting that an argument has the incorrect units."""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ValueException(Exception):
    """Exception denoting that an argument has the incorrect value."""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

#=============================================================================
# DECORATOR
#=============================================================================


#TODO: Do we need to use 'from functools import wraps' to help us here?
def accepts(*types):
    """
    Decorator for class methods that should accept only specified types.

    EXAMPLE

    @accepts(float, int)
    def function(a, b):
        return b*a

    """
    def check_accepts(f):
        nargs = (f.func_code.co_argcount - 1)  # exclude self
        assert len(types) == nargs, ("Incorrect number of args supplied in "
                "@accepts decorator for class method %s" % (f.func_name))

        def new_f(*args, **kwds):
            for (a, t) in zip(args[1:], types):
                if a is not None:
                    assert isinstance(a, t), "arg %r does not match %s" % (a, t)
            return f(*args, **kwds)

        new_f.func_name = f.func_name  # copy function name
        new_f.func_doc = f.func_doc  # copy docstring
        return new_f

    return check_accepts


def accepts_compatible_units(*units):
    """
    Decorator for class methods that should accept only arguments compatible with specified units.

    Each argument of the function will be matched with an argument of @acceptunits.
    Those arguments of the function that correspond @acceptunits which are not None
    will be checked to ensure they are compatible with the specified units.

    EXAMPLE

    @acceptsunits(units.meter, None, units.kilocalories_per_mole)
    def function(a, b, c): pass
    function(1.0 * units.angstrom, 3, 1.0 * units.kilojoules_per_mole)

    """
    def check_units(f):
        nargs = (f.func_code.co_argcount - 1) # exclude self
        assert len(units) == nargs, "incorrect number of units supplied in @accepts_compatible_units decorator for class method %s" % (f.func_name)
        def new_f(*args, **kwds):
            for (a, u) in zip(args[1:], units):
                if u is not None:
                    assert (a.unit).is_compatible(u), "arg %r does not have units compatible with %s" % (a,u)
            return f(*args, **kwds)
        new_f.func_name = f.func_name # copy function name
        new_f.func_doc = f.func_doc # copy docstring
        return new_f
    return check_units

def returns(rtype):
    """
    Decorator for functions that should only return specific types.
    EXAMPLE

    @returns(int)
    def function(): return 7

    """

    def check_returns(f):
        def new_f(*args, **kwds):
            result = f(*args, **kwds)
            assert isinstance(result, rtype), "return value %r does not match %s" % (result,rtype)
            return result
        new_f.func_name = f.func_name # copy function name
        new_f.func_doc = f.func_doc # copy docstring
        return new_f
    return check_returns 
