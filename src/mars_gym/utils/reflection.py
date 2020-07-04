import importlib
import inspect
from typing import Type, TypeVar, Set

try:
    from typing import GenericMeta  # python 3.6
except ImportError:
    # in 3.7, GenericMeta doesn't exist but we don't need it
    class GenericMeta(type):
        pass

T = TypeVar("T")


def load_attr(attr_path: str, expected_type: Type[T]) -> T:
    splitted_path = attr_path.split(".")
    module_path = ".".join(splitted_path[:-1])
    attr_name = splitted_path[-1]

    module = importlib.import_module(module_path)
    attr = getattr(module, attr_name)

    if isinstance(expected_type, GenericMeta):  # the expected_type is a type itself
        if not issubclass(attr, expected_type.__args__[0]):
            raise ValueError(f"{attr_path} should be a sub class of {expected_type}")
    else:
        if not isinstance(attr, expected_type):
            raise ValueError(f"{attr_path} should be of type {expected_type}")

    return attr


def get_attribute_names(class_: Type) -> Set[str]:
    return set(
        list(zip(*(inspect.getmembers(class_, lambda a: not (inspect.isroutine(a))))))[
            0
        ]
    )
