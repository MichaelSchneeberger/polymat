class UniqueNameSelector:
    """
    Ensures unique variable names by tracking used names and raising exceptions on duplicates.
    Usage:
        select_unique = UniqueNameSelector()
        polymat.define_variable(select_unique("x"))  # Allows "x"
        polymat.define_variable(select_unique("x"))  # Raises ValueError
    """
    
    def __init__(self):
        self._used_names = set()

    def __call__(self, name):
        name = name.split('_', 1)[0]

        """Check for uniqueness and return the name if unused."""
        if name in self._used_names:
            raise ValueError(f"Variable name '{name}' is already defined. Use a unique name.")
        
        self._used_names.add(name)
        return name