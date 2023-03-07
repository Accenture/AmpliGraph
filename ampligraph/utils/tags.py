# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import warnings


class experimentalWarning(Warning):
    """Warning that is triggered when the
    experimental function is run.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def experimental(func):
    """
    Decorator - a function that accepts another function
    and marks it as experimental, meaning it may change in
    future releases, or its execution is not guaranteed.

    Example:

    >>>@experimental
    >>>def a_function():
    >>>    "Demonstration function"
    >>>    return "demonstration"

    >>>a_function()
    experimentalWarning: 'Experimental! Function: a_function is experimental.
    Use at your own risk.'
    warnings.warn(experimentalWarning(msg))
    demonstration

    To disable experimentalWarning set this in the module:
    >>>warnings.filterwarnings("ignore", category=experimentalWarning)

    """

    def mark_experimental():
        msg = f"Experimental! Function: {func.__name__} is experimental. Use \
                at your own risk."

        warnings.warn(experimentalWarning(msg))

        return func()

    return mark_experimental


def deprecated(*args, **kwargs):
    """
    Decorator - a function that accepts another function
    and marks it as deprecated, meaning it may be discontinued in
    future releases, and is provided only for backward compatibility purposes.

    ---------------
    Example:

    >>>@deprecated(instead="module2.another_function")
    >>>def a_function():
    >>>    "Demonstration function"
    >>>    return "demonstration"

    >>>a_function()
    DeprecationWarning: Deprecated! Function: a_function is deprecated.
    Instead use module2.another_function.
    warnings.warn(DeprecationWarning(msg))
    demonstration
    """

    def mark_deprecated(func):
        msg = f"Deprecated! Function: {func.__name__} is deprecated. \
                Instead use {kwargs['instead']}."

        warnings.warn(DeprecationWarning(msg))

        return func

    return mark_deprecated
