class Symbol(str):
    """
    A class representing a human-readable identifier of a variable in a polynomial expression.

    Internally, each variable is represented by an integer to ensure efficient manipulation
    of polynomial expressions. For each defined variable, one (or multiple) integer index is introduced and
    linked with the `Symbol` using the `indices` dictionary within the state object. This
    allows for a clear mapping between human-readable variable symbols and their internal
    integer representations.
    """
