"""
Inference by Enumeration — Burglary Alarm Network
===========================================================
A classic Bayesian network example for demonstrating exact
inference when some variables are hidden (not observed).

Network structure:

    Burglary   Earthquake
         \\        /
          \\      /
           Alarm
          /      \\
    JohnCalls   MaryCalls

Burglary and Earthquake are independent root causes.
Alarm depends on both of them.
JohnCalls and MaryCalls each depend only on Alarm.

Goal: compute P(Burglary | JohnCalls=True, MaryCalls=True)
"Given that both neighbors called, how likely is it that
there's actually a burglary?"

Earthquake and Alarm are never observed directly — to answer the
query we have to sum (marginalize) over every possible value of
those hidden variables. That summing-out step is what "inference
by enumeration" actually refers to.
"""

# ---------------------------------------------------------------
# 1. Network definition
# ---------------------------------------------------------------
# Each node stores its parent list and a CPT. The CPT maps a tuple
# of parent values (in the same order as `parents`) to
# P(node = True | those parent values). Root nodes use an empty
# tuple () as their only key.

NETWORK = {
    "Burglary":   {"parents": [], "cpt": {(): 0.001}},
    "Earthquake": {"parents": [], "cpt": {(): 0.002}},
    "Alarm": {
        "parents": ["Burglary", "Earthquake"],
        "cpt": {
            (True,  True):  0.95,
            (True,  False): 0.94,
            (False, True):  0.29,
            (False, False): 0.001,
        },
    },
    "JohnCalls": {"parents": ["Alarm"], "cpt": {(True,): 0.90, (False,): 0.05}},
    "MaryCalls": {"parents": ["Alarm"], "cpt": {(True,): 0.70, (False,): 0.01}},
}

# Topological order — every parent must appear before its children.
VARIABLES = ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]


def probability(var: str, value: bool, assignment: dict) -> float:
    """P(var = value | parents(var) set according to `assignment`)."""
    node = NETWORK[var]
    parent_values = tuple(assignment[p] for p in node["parents"])
    p_true = node["cpt"][parent_values]
    return p_true if value else 1 - p_true


def enumerate_all(variables: list, evidence: dict) -> float:
    """
    The core of the enumeration algorithm.

    Walks through `variables` in topological order:
      - if a variable's value is already fixed (it's evidence, or
        it was set by the outer query loop), multiply in its
        probability and recurse on the rest
      - otherwise it's hidden, so sum over both of its possible
        values, recursing into each branch

    This is exactly a depth-first traversal of every row of the
    joint distribution that's consistent with the evidence —
    "enumeration" in the literal sense.
    """
    if not variables:
        return 1.0

    Y, rest = variables[0], variables[1:]

    if Y in evidence:
        p_y = probability(Y, evidence[Y], evidence)
        return p_y * enumerate_all(rest, evidence)

    # Y is hidden — sum out both possible values.
    total = 0.0
    for y in (True, False):
        extended_evidence = {**evidence, Y: y}
        p_y = probability(Y, y, extended_evidence)
        total += p_y * enumerate_all(rest, extended_evidence)
    return total


def enumeration_ask(query_var: str, evidence: dict) -> dict:
    """
    Computes the full posterior distribution P(query_var | evidence)
    by running enumerate_all once per possible value of the query
    variable, then normalizing the results so they sum to 1.
    """
    distribution = {}
    for value in (True, False):
        extended_evidence = {**evidence, query_var: value}
        distribution[value] = enumerate_all(VARIABLES, extended_evidence)

    total = sum(distribution.values())
    return {value: prob / total for value, prob in distribution.items()}


if __name__ == "__main__":
    evidence = {"JohnCalls": True, "MaryCalls": True}

    result = enumeration_ask("Burglary", evidence)

    print("Query: P(Burglary | JohnCalls=True, MaryCalls=True)")
    print("-" * 58)
    print(f"P(Burglary=True  | evidence) = {result[True]:.4f}  ({result[True]:.2%})")
    print(f"P(Burglary=False | evidence) = {result[False]:.4f}  ({result[False]:.2%})")

    prior = NETWORK["Burglary"]["cpt"][()]
    print("\nFor comparison, the prior belief before any evidence:")
    print(f"P(Burglary=True) = {prior:.4f}  ({prior:.2%})")

    print("\nNote: Earthquake and Alarm were never observed — both were")
    print("summed out inside enumerate_all(). That's the 'enumeration' part.")