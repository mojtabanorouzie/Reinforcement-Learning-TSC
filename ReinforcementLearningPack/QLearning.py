"""The agent and its temporal-difference update."""

import random


class ReinforcementLearningAgent:
    """One learning traffic signal controller, bound to one junction.

    Holds a `numberOfState` x `numberOfAction` Q-table and everything needed
    to carry a decision across control cycles: `state` and `action` are the
    pair chosen last cycle, still awaiting their reward.

    Exploration is per-state rather than global. `probabilityOfRandomAction`
    starts at 1.0 for every state and is decayed only when that state is
    actually visited and explored, so rarely seen traffic patterns keep
    exploring long after common ones have settled. `counter` records how often
    each state has been seen and gates the dataset export in CreateDataSet.

    `oldDta` is a five-slot sliding window of recent delay times; GetReward
    scores each new observation against it.
    """
    def __init__(self, agentId, junctionIdSectionIn, junctionIdSectionOut, controlType, numberOfPhases,
                 numberOfAction, numberOfState, initLearningRate, initDiscountFactor):
        self.id = agentId
        self.idSectionIn = junctionIdSectionIn
        self.idSectionOut = junctionIdSectionOut
        self.control_type = controlType
        self.numberOfPhases = numberOfPhases
        self.state = random.randint(0, numberOfState - 1)
        self.action = random.randint(0, numberOfAction - 1)
        self.probabilityOfRandomAction = [1.0 for i in range(numberOfState)]
        self.qTable = [[0 for i in range(numberOfAction)] for j in range(numberOfState)]
        self.learningRate = initLearningRate
        self.discountFactor = initDiscountFactor
        self.oldDta = [0 for i in range(5)]
        self.counter = [0 for i in range(numberOfState)]


def updateQTable(qValue, qValueNew, state, action, newState, newAction, reward, learningRate, discountFactor):
    """Apply one temporal-difference update and return the new Q-value.

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

    This is the on-policy (SARSA) form. The bootstrap term `qValueNew` is the
    Q-value of the action the caller actually selected for the next state, not
    the maximum over that state's actions -- so exploratory moves are folded
    into the estimate rather than discarded. Main.py passes
    qTable[currentState][currentAction], where currentAction came from the
    epsilon-greedy selector.

    `state`, `action`, `newState` and `newAction` are accepted but not used:
    the caller has already resolved both Q-values from them.
    """
    qValue += learningRate * (reward + (discountFactor * qValueNew) - qValue)
    return qValue
