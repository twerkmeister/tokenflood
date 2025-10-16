from typing import TypeVar, Sequence, Union

T = TypeVar("T")
Number = Union[int, float]


def non_empty_list(seq: Sequence[T]) -> Sequence[T]:
    if len(seq) == 0:
        raise ValueError("list must not be empty.")
    return seq


def at_least_size(n: int):
    def size_check(seq: Sequence[T]) -> Sequence[T]:
        if len(seq) < n:
            raise ValueError(f"list must have at least {n} elements.")
        return seq

    return size_check


def all_non_empty_strings(seq: Sequence[str]) -> Sequence[str]:
    if not all(seq):
        raise ValueError("all elements must be non-empty.")
    return seq


def unique_elements(seq: Sequence[T]) -> Sequence[T]:
    if not len(list(set(seq))) == len(seq):
        raise ValueError("elements must be unique.")
    return seq


def all_strictly_positive(seq: Sequence[Number]) -> Sequence[Number]:
    if not all([x > 0 for x in seq]):
        raise ValueError("all elements must be larger than 0.")
    return seq


def all_positive(seq: Sequence[Number]) -> Sequence[Number]:
    if not all([x >= 0 for x in seq]):
        raise ValueError("all elements must be larger or equal 0.")
    return seq
