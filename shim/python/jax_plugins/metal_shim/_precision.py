"""Compensated float32 arithmetic for cancellation-sensitive factorizations."""

import numpy as np
from jax import lax


def two_sum(a, b):
    value = a + b
    part = value - a
    return value, (a - (value - part)) + (b - part)


def two_product(a, b):
    # Bitwise splitting avoids the overflow of Dekker's 4097*x splitter.
    def split(x):
        bits = lax.bitcast_convert_type(x, np.int32)
        high = lax.bitcast_convert_type(lax.bitwise_and(bits, lax.full_like(bits, -4096)), np.float32)
        return high, x - high

    ah, al = split(a)
    bh, bl = split(b)
    value = a * b
    return value, ((ah * bh - value) + ah * bl + al * bh) + al * bl


def sum_pairs(high, low, axis):
    count = high.shape[axis]
    if count == 0:
        shape = high.shape[:axis] + high.shape[axis + 1:]
        return lax.full(shape, 0, high.dtype), lax.full(shape, 0, high.dtype)
    while count > 1:
        half = count // 2
        left = lax.slice_in_dim(high, 0, half, axis=axis)
        right = lax.slice_in_dim(high, half, 2 * half, axis=axis)
        value, error = two_sum(left, right)
        error = error + (lax.slice_in_dim(low, 0, half, axis=axis)
                         + lax.slice_in_dim(low, half, 2 * half, axis=axis))
        value, error = two_sum(value, error)
        if count % 2:
            value = lax.concatenate([value, lax.slice_in_dim(high, count - 1, count, axis=axis)], axis)
            error = lax.concatenate([error, lax.slice_in_dim(low, count - 1, count, axis=axis)], axis)
        high, low = value, error
        count = (count + 1) // 2
    return lax.squeeze(high, (axis,)), lax.squeeze(low, (axis,))


def subtract_product(a, b, c):
    high, low = two_product(b, c)
    value, error = two_sum(a, -high)
    return value + (error - low)


def add_pairs(x, y):
    value, error = two_sum(x[0], y[0])
    return two_sum(value, error + x[1] + y[1])


def multiply_pairs(x, y):
    value, error = two_product(x[0], y[0])
    return two_sum(value, error + x[0] * y[1] + x[1] * y[0])


def divide_pairs(x, y):
    quotient = x[0] / y[0]
    product = multiply_pairs((quotient, lax.full_like(quotient, 0)), y)
    residual = add_pairs(x, (-product[0], -product[1]))
    return two_sum(quotient, (residual[0] + residual[1]) / y[0])


def sqrt_pair(x):
    root = lax.sqrt(x[0])
    square = two_product(root, root)
    residual = add_pairs(x, (-square[0], -square[1]))
    return two_sum(root, (residual[0] + residual[1]) / (2 * root))
