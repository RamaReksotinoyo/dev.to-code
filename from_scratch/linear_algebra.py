from typing import List, Dict, Callable


Vector = List[float] #Built-in variables

def add(a: Vector, b: Vector) -> Vector:
    """
    Adds two vectors
    params:
        v, w: Vector that we want to add
    return:
        Added vector
    """
    assert len(a) == len(b), "vector size must be the same"
    return [a_i + b_i for a_i, b_i in zip(a, b)]  


def subtract(a: Vector, b: Vector) -> Vector:
    """
    Subtracts two vectors
    params:
        v, w: Vector that we want to substract
    return:
        Subtracted vector
    """
    assert len(a) == len(b), "vector size must be the same"
    return [a_i - b_i for a_i, b_i in zip(a, b)]  


def vector_sum(vectors: List[Vector]) -> Vector:
    """
    Sums all corresponding elements
    params:
        vectors that we want to sum 
    return:
        summed up vector
    """

    num_elements = len(vectors[0])

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]



#TODO






