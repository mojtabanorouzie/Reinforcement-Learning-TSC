"""State encoding: rank the four approach queues into one of 24 states."""


def getState(longQueueInSection):
    """Map four approach queue lengths to a state index in 0..23.

    The state is the *ordering* of the four queues, longest first, not their
    magnitudes -- there are 4! = 24 orderings, hence 24 states. Two junctions
    with very different traffic volumes but the same relative pressure across
    approaches therefore share a state, which is what lets one agent's
    experience mean anything to another.

    `longQueueInSection` is the maximum queue length on each of the junction's
    four incoming approaches, in the order Main.py resolved them.

    Comparisons are `>=`, so ties are resolved by falling through to the first
    branch that accepts them; equal queues everywhere yields state 0.
    """
    state = 0
    if (longQueueInSection[0] >= longQueueInSection[1]) and \
            (longQueueInSection[1] >= longQueueInSection[2]) and (longQueueInSection[2] >= longQueueInSection[3]):
        state = 0
    elif (longQueueInSection[0] >= longQueueInSection[1]) and \
            (longQueueInSection[1] >= longQueueInSection[3]) and (longQueueInSection[3] >= longQueueInSection[2]):
        state = 1
    elif (longQueueInSection[0] >= longQueueInSection[2]) and \
            (longQueueInSection[2] >= longQueueInSection[1]) and (longQueueInSection[1] >= longQueueInSection[3]):
        state = 2
    elif (longQueueInSection[0] >= longQueueInSection[3]) and \
            (longQueueInSection[3] >= longQueueInSection[1]) and (longQueueInSection[1] >= longQueueInSection[2]):
        state = 3
    elif (longQueueInSection[0] >= longQueueInSection[2]) and \
            (longQueueInSection[2] >= longQueueInSection[3]) and (longQueueInSection[3] >= longQueueInSection[1]):
        state = 4
    elif (longQueueInSection[0] >= longQueueInSection[3]) and \
            (longQueueInSection[3] >= longQueueInSection[2]) and (longQueueInSection[2] >= longQueueInSection[1]):
        state = 5
    elif (longQueueInSection[1] >= longQueueInSection[0]) and \
            (longQueueInSection[0] >= longQueueInSection[2]) and (longQueueInSection[2] >= longQueueInSection[3]):
        state = 6
    elif (longQueueInSection[1] >= longQueueInSection[0]) and \
            (longQueueInSection[0] >= longQueueInSection[3]) and (longQueueInSection[3] >= longQueueInSection[2]):
        state = 7
    elif (longQueueInSection[2] >= longQueueInSection[0]) and \
            (longQueueInSection[0] >= longQueueInSection[1]) and (longQueueInSection[1] >= longQueueInSection[3]):
        state = 8
    elif (longQueueInSection[3] >= longQueueInSection[0]) and \
            (longQueueInSection[0] >= longQueueInSection[1]) and (longQueueInSection[1] >= longQueueInSection[2]):
        state = 9
    elif (longQueueInSection[2] >= longQueueInSection[0]) and \
            (longQueueInSection[0] >= longQueueInSection[3]) and (longQueueInSection[3] >= longQueueInSection[1]):
        state = 10
    elif (longQueueInSection[3] >= longQueueInSection[0]) and \
            (longQueueInSection[0] >= longQueueInSection[2]) and (longQueueInSection[2] >= longQueueInSection[1]):
        state = 11
    elif (longQueueInSection[1] >= longQueueInSection[2]) and \
            (longQueueInSection[2] >= longQueueInSection[0]) and (longQueueInSection[0] >= longQueueInSection[3]):
        state = 12
    elif (longQueueInSection[1] >= longQueueInSection[3]) and \
            (longQueueInSection[3] >= longQueueInSection[0]) and (longQueueInSection[0] >= longQueueInSection[2]):
        state = 13
    elif (longQueueInSection[2] >= longQueueInSection[1]) and \
            (longQueueInSection[1] >= longQueueInSection[0]) and (longQueueInSection[0] >= longQueueInSection[3]):
        state = 14
    elif (longQueueInSection[3] >= longQueueInSection[1]) and \
            (longQueueInSection[1] >= longQueueInSection[0]) and (longQueueInSection[0] >= longQueueInSection[2]):
        state = 15
    elif (longQueueInSection[2] >= longQueueInSection[3]) and \
            (longQueueInSection[3] >= longQueueInSection[0]) and (longQueueInSection[0] >= longQueueInSection[1]):
        state = 16
    elif (longQueueInSection[3] >= longQueueInSection[2]) and \
            (longQueueInSection[2] >= longQueueInSection[0]) and (longQueueInSection[0] >= longQueueInSection[1]):
        state = 17
    elif (longQueueInSection[1] >= longQueueInSection[2]) and \
            (longQueueInSection[2] >= longQueueInSection[3]) and (longQueueInSection[3] >= longQueueInSection[0]):
        state = 18
    elif (longQueueInSection[1] >= longQueueInSection[3]) and \
            (longQueueInSection[3] >= longQueueInSection[2]) and (longQueueInSection[2] >= longQueueInSection[0]):
        state = 19
    elif (longQueueInSection[2] >= longQueueInSection[1]) and \
            (longQueueInSection[1] >= longQueueInSection[3]) and (longQueueInSection[3] >= longQueueInSection[0]):
        state = 20
    elif (longQueueInSection[3] >= longQueueInSection[1]) and \
            (longQueueInSection[1] >= longQueueInSection[2]) and (longQueueInSection[2] >= longQueueInSection[0]):
        state = 21
    elif (longQueueInSection[2] >= longQueueInSection[3]) and \
            (longQueueInSection[3] >= longQueueInSection[1]) and (longQueueInSection[1] >= longQueueInSection[0]):
        state = 22
    elif (longQueueInSection[3] >= longQueueInSection[2]) and \
            (longQueueInSection[2] >= longQueueInSection[1]) and (longQueueInSection[1] >= longQueueInSection[0]):
        state = 23
    return state
