"""Action selection and the green-time splits the actions stand for."""

import random


def actionSelection(randomProbability, qTable, numberOfAction):
    """Pick an action epsilon-greedily and return it with its green times.

    `qTable` is the single Q-table row for the current state and
    `randomProbability` that state's own exploration rate, which the caller
    decays independently per state.

    Returns [action, phaseDuration, actionType]. `actionType` is "best" or
    "random"; the caller uses it both to decide whether to decay exploration
    and to keep exploratory decisions out of the exported dataset.
    """
    if random.random() > randomProbability:
        action = qTable.index(max(qTable))
        actionType = "best"
    else:
        action = random.randint(0, numberOfAction - 1)
        actionType = "random"
    phaseDuration = getPhaseDuration(action)
    return [action, phaseDuration, actionType]


def getPhaseDuration(action):
    """Map an action index to green times, in seconds, for the four phases.

    All 19 splits total 92 s. The agent can only redistribute green time
    between approaches, never lengthen or shorten the cycle: the remaining 8 s
    of Main.py's 100 s cycle belong to the inter-green phases, which the
    controller does not touch.

    The splits are the rearrangements of three duration multisets:

        actions 0-5    {33, 33, 13, 13}   two approaches favoured
        actions 6-17   {33, 23, 23, 13}   one favoured, one starved
        action  18     {23, 23, 23, 23}   uniform, the default branch below

    Anything outside 0..18 also falls through to the uniform split.
    """
    phaseDuration = [23] * 4  # Action 18
    if action == 0:
        phaseDuration = [33, 33, 13, 13]
    elif action == 1:
        phaseDuration = [33, 13, 33, 13]
    elif action == 2:
        phaseDuration = [33, 13, 13, 33]
    elif action == 3:
        phaseDuration = [13, 33, 33, 13]
    elif action == 4:
        phaseDuration = [13, 33, 13, 33]
    elif action == 5:
        phaseDuration = [13, 13, 33, 33]
    elif action == 6:
        phaseDuration = [33, 23, 23, 13]
    elif action == 7:
        phaseDuration = [33, 23, 13, 23]
    elif action == 8:
        phaseDuration = [33, 13, 23, 23]
    elif action == 9:
        phaseDuration = [23, 33, 23, 13]
    elif action == 10:
        phaseDuration = [23, 33, 13, 23]
    elif action == 11:
        phaseDuration = [13, 33, 23, 23]
    elif action == 12:
        phaseDuration = [23, 23, 33, 13]
    elif action == 13:
        phaseDuration = [23, 13, 33, 23]
    elif action == 14:
        phaseDuration = [13, 23, 33, 23]
    elif action == 15:
        phaseDuration = [23, 23, 13, 33]
    elif action == 16:
        phaseDuration = [23, 13, 23, 33]
    elif action == 17:
        phaseDuration = [13, 23, 23, 33]
    return phaseDuration
