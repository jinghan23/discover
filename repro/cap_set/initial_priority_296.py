"""Seed priority for F_3^8 that reproducibly builds a 296-element cap set."""


def priority(el, n):
    nonzero = sum(x != 0 for x in el)
    zeros = n - nonzero
    twos = sum(x == 2 for x in el)
    mirror = sum(1 for k in range(1, n // 2) if el[k] == el[-k])
    endpoints = int(el[0] == el[-1]) + int(el[1] == el[-2])

    zero_runs = 0
    in_zero_run = False
    max_zero_run = 0
    current_zero_run = 0
    transitions = 0
    previous = el[0]
    for x in el:
        if x == 0:
            if not in_zero_run:
                zero_runs += 1
                current_zero_run = 0
            current_zero_run += 1
            max_zero_run = max(max_zero_run, current_zero_run)
            in_zero_run = True
        else:
            current_zero_run = 0
            in_zero_run = False
        if x != previous:
            transitions += 1
            previous = x

    quadratic_residue = sum((k + 1) * x * x for k, x in enumerate(el)) % 3
    linear_residue = sum((k + 1) * x for k, x in enumerate(el)) % 3

    return (
        3 * nonzero
        + zeros
        + 6 * twos
        + 6 * endpoints
        + 6 * zero_runs
        - 4 * max_zero_run
        + 4 * transitions
        + 2 * quadratic_residue
        + 5 * linear_residue
    )

