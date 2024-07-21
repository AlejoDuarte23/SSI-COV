
import functools
import inspect
import numpy as np

def print_input_sizes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Function to get size or shape of the argument
        def get_size(arg):
            if isinstance(arg, np.ndarray):
                return f"shape: {arg.shape}"
            try:
                return f"length: {len(arg)}"
            except TypeError:
                return f"not applicable (type: {type(arg).__name__})"
        
        # Get the function's argument names
        arg_names = inspect.signature(func).parameters
        
        # Print the function name
        print(f"Function '{func.__name__}' called with:")
        
        # Print positional arguments with their names
        for i, (arg_name, arg) in enumerate(zip(arg_names, args)):
            print(f"Argument '{arg_name}' (positional): {get_size(arg)}")
        
        # Print keyword arguments
        for key, value in kwargs.items():
            print(f"Argument '{key}' (keyword): {get_size(value)}")
        
        return func(*args, **kwargs)
    return wrapper