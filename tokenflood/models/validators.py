from typing import TypeVar, Sequence, Union

T = TypeVar("T")
Number = Union[int, float]

def non_empty_list(l: Sequence[T]) -> Sequence[T]:
    if len(l) == 0:
        raise ValueError("list must not be empty.")
    return l

def at_least_size(n: int):
    def size_check(l: Sequence[T]) -> Sequence[T]:
        if len(l) < n:
            raise ValueError(f"list must have at least {n} elements.")
        return l
    return size_check

def all_non_empty_strings(l: Sequence[str]) -> Sequence[str]:
    if not all(l):
        raise ValueError("all elements must be non-empty.")
    return l

def unique_elements(l: Sequence[T]) -> Sequence[T]:
    if not len(list(set(l))) == len(l):
        raise ValueError("elements must be unique.")
    return l

def all_strictly_positive(l: Sequence[Number]) -> Sequence[Number]:
    if not all([x > 0 for x in l]):
        raise ValueError("all elements must be larger than 0.")
    return l

def all_positive(l: Sequence[Number]) -> Sequence[Number]:
    if not all([x >= 0 for x in l]):
        raise ValueError("all elements must be larger or equal 0.")
    return l
