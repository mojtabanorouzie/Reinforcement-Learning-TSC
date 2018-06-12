import random


class ReinforcementLearningAgent:
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
    qValue += learningRate * (reward + (discountFactor * qValueNew) - qValue)
    return qValue
