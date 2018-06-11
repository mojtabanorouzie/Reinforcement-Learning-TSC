import random


def actionSelection(randomProbability, qTable, numberOfAction):
    if random.random() > randomProbability:
        action = qTable.index(max(qTable))
        actionType = "best"
    else:
        action = random.randint(0, numberOfAction - 1)
        actionType = "random"
    phaseDuration = getPhaseDuration(action)
    return [action, phaseDuration, actionType]


def getPhaseDuration(action):
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
