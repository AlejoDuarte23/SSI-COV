import functools
import inspect
import time
from collections.abc import Callable, Sized
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

P = ParamSpec("P")
R = TypeVar("R")


def print_input_sizes(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[misc,unused-ignore]
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> Any:
        # Function to get size or shape of the argument
        def get_size(arg: object) -> str:
            if isinstance(arg, np.ndarray):
                return f"shape: {arg.shape}"
            if isinstance(arg, Sized):
                return f"length: {len(arg)}"
            return f"not applicable (type: {type(arg).__name__})"

        # Get the function's argument names
        arg_names = inspect.signature(func).parameters

        # Print the function name
        print(f"Function '{func.__name__}' called with:")

        # Print positional arguments with their names
        for _i, (arg_name, arg) in enumerate(zip(arg_names, args, strict=False)):
            print(f"Argument '{arg_name}' (positional): {get_size(arg)}")

        # Print keyword arguments
        for key, value in kwargs.items():
            print(f"Argument '{key}' (keyword): {get_size(value)}")

        return func(*args, **kwargs)

    return cast(Callable[..., Any], wrapper)


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[misc,unused-ignore]
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__}, elapsed time: {elapsed_time:.6f} seconds")
        return result

    return cast(Callable[..., Any], wrapper)
