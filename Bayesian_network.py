"""
Bayesian Network - Inference by Enumeration
--------------------------------------------------------------------
Learning Notes
--------------------------------------------------------------------
This implementation is part of my journey from PeopleSoft Developer to AI Engineer.
While studying Harvard's CS50 AI course, I wanted to understand how
Bayesian Networks actually perform inference instead of simply calling an existing library.
Rather than focusing only on the final probability, I wanted to answer questions like:

• Why does the algorithm explore every possible hidden variable?
• Where does the computation become expensive?
• Why is Variable Elimination considered a major improvement?

The objective of this implementation is to compute:

    P(Burglary | JohnCalls=True, MaryCalls=True)

using the Inference by Enumeration algorithm.

Everything here is implemented from scratch so I can understand the algorithm before moving on to more optimized approaches.

Related Articles

Medium:
How Does a Bayesian Network Actually Reach a Decision?

Next:
Variable Elimination

"""

from itertools import product

from matplotlib.pylab import rint

# --------------------------------------------------------------------
# Bayesian Network Definition
# --------------------------------------------------------------------
#
# The classic Burglary Alarm example.
#
# Burglary      Earthquake
#      \          /
#       \        /
#          Alarm
#         /     \
#   JohnCalls  MaryCalls
#
# Burglary and Earthquake are independent causes.
# Alarm depends on both.
# John and Mary only observe the alarm.

#
# I intentionally kept the classic burglary network.
# It's simple enough to understand manually and matches
# the example discussed in my Medium article.
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

# Parents must appear before their children.
# The enumeration algorithm assumes this ordering.
VARIABLES = ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]


def probability(var: str, value: bool, assignment: dict) -> float:
    """
    Return the probability of a node taking a given value.

    At this point the values of all parent nodes are already known,
    so we can look up the correct value from the CPT.
    """
    node = NETWORK[var]
    parent_values = tuple(assignment[p] for p in node["parents"])
    p_true = node["cpt"][parent_values]
    return p_true if value else 1 - p_true


def enumerate_all(variables: list, evidence: dict) -> float:
    """
    Recursive implementation of Inference by Enumeration.
    Known variables contribute their probability directly.

    Hidden variables are explored twice
    (True and False) and their contributions are added together.

    Thinking of each recursive call as one possible
    world made the algorithm much easier for me to understand.
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


def enumeration_ask(query_vars, evidence: dict) -> dict:
    """
    Run inference for one or more query variables.

    I started with a single query variable, then extended the
    implementation to support joint queries so I could experiment
    with different scenarios while learning.
    """
    if isinstance(query_vars, str):
        query_vars = [query_vars]

    distribution = {}
    for values in product((True, False), repeat=len(query_vars)):
        extended_evidence = {**evidence, **dict(zip(query_vars, values))}
        distribution[values if len(values) > 1 else values[0]] = enumerate_all(VARIABLES, extended_evidence)

    return normalize(distribution)


def normalize(distribution: dict) -> dict:
    """Normalize a probability distribution so values sum to 1."""
    total = sum(distribution.values())
    if total == 0:
        raise ValueError("Distribution probabilities sum to zero, cannot normalize.")
    return {value: prob / total for value, prob in distribution.items()}


def format_result(query_vars, key):
    if isinstance(query_vars, str):
        return f"{query_vars}={key}"
    return ", ".join(f"{var}={value}" for var, value in zip(query_vars, key))


def run_test_case(title: str, evidence: dict, query_vars="Burglary", note: str = None) -> None:
    """Run a single inference scenario and print the results."""
    if isinstance(query_vars, str):
        query_vars_label = query_vars
    else:
        query_vars_label = ", ".join(query_vars)
    print("-" * 100)
    print()
    print(f"{title} ({query_vars_label}):")
    print(f"Evidence: {evidence}")
    result = enumeration_ask(query_vars, evidence)

    for key, prob in sorted(result.items(), key=lambda item: item[0], reverse=True):
        label = format_result(query_vars, key)
        print(f"P({label} | evidence) = {prob:.4f}  ({prob:.2%})")

    if note:
        print("Observation:")
        for line in note.splitlines():
            print(line)
    print()


if __name__ == "__main__":
    print("=" * 100)
    print("Bayesian Network - Inference by Enumeration")
    print("=" * 100)

    run_test_case("Prior belief", {})
    run_test_case(
        "John calls",
        {"JohnCalls": True},
        note=(
            "John calling alone provides weaker evidence because false "
            "positives are possible."
        ),
    )
    run_test_case(
        "Mary calls",
        {"MaryCalls": True},
        note=(
            "Mary calling alone provides stronger evidence than John, "
            "but it still does not guarantee a burglary."
        ),
    )
    run_test_case(
        "Both neighbours call",
        {
            "JohnCalls": True,
            "MaryCalls": True,
        },
        note=(
            "Both neighbours calling increases the probability dramatically,"
            "\nbut it still doesn't guarantee there was a burglary because an "
            "earthquake could also explain the alarm."
        ),
    )

    run_test_case(
        "Alarm observed",
        {"Alarm": True},
        note=(
            "The alarm being observed makes burglary more likely, but "
            "earthquake remains a plausible explanation."
        ),
    )
    run_test_case(
        "Joint query for Burglary and Alarm",
        {"JohnCalls": True, "MaryCalls": True},
        query_vars=["Burglary", "Alarm"],
        note=(
            "Jointly querying Burglary and Alarm shows how evidence "
            "updates both the hidden cause and the alarm state."
        ),
    )
    run_test_case(
        "Earthquake observed",
        {"Earthquake": True},
        note=(
            "Knowing an earthquake occurred provides an alternative explanation "
            "for the alarm, so the probability of burglary changes differently "
            "than when only alarm-related evidence is observed."
        ),
    )
    run_test_case(
        "False alarm",
        {"Alarm": False},
        note=(
            "When the alarm is known not to have gone off, "
            "the probability of burglary becomes extremely small."
        ),
    )
    print("=" * 100 + "\n"
        "Summary")
    print("=" * 100 + "\n"
        "• New evidence changes our belief about hidden events.\n"
        "• Stronger evidence usually has a larger impact on the posterior.\n"
        "• Enumeration evaluates every possible hidden state.\n"
        "• The same calculations are repeated many times.\n"
        "• TThat repeated work motivates Variable Elimination.\n\n"
        "Next repository:\n"
        "Variable Elimination from Scratch\n")
    print("Personal Reflection")
    print("-"* 100 + "\n")
    print(
        "Implementing this algorithm changed the way I think about Bayesian Networks.\n"
        "The mathematics became much easier to understand once I could see\n"
        "how every hidden variable contributes to the final probability.\n"
    )   
