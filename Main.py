from AAPI import *
from ReinforcementLearningPack import QLearning, GetState, ActionSelection, GetReward, CreateDataSet
import threading

# Global Variables
warmup = 1800
cycle = 100
eGreedy = 0.01
initLearningRate = 0.5
initDiscountFactor = 0.5
decayProbability = 0.02
decayLearningRate = 0.005
incrementDiscountFactor = 0.005
numberOfState = 24
numberOfAction = 19
tempTime = -1
agents = []
createDataSet = True


def mainProcess(index, timeSta):
    # 1. Get feature from network (Long Queue, Delay Time and Density)
    longQueueInSection = [0] * 4
    delayTime = [0] * 4
    density = [0] * 4
    if AKIIsGatheringStatistics() >= 0:
        for i in range(4):
            statisticalInfo = AKIEstGetParcialStatisticsSection(agents[index].idSectionIn[i], 100, 0)
            if statisticalInfo.report == 0:
                longQueueInSection[i] = statisticalInfo.LongQueueMax
                delayTime[i] = statisticalInfo.DTa
                density[i] = statisticalInfo.Density
            else:
                longQueueInSection[i] = 0
                delayTime[i] = 0
                density[i] = 0
    else:
        AKIPrintString("Warning AKIIsGatheringStatistics")
    # 2. Get State
    currentState = GetState.getState(longQueueInSection)
    # 3.1 Action Selection
    [currentAction, phaseDuration, actionType] = ActionSelection.actionSelection(
        agents[index].probabilityOfRandomAction[currentState], agents[index].qTable[currentState],
        numberOfAction)
    if agents[index].probabilityOfRandomAction[currentState] >= eGreedy and actionType == "random":
        agents[index].probabilityOfRandomAction[currentState] -= decayProbability
    # 3.2 Set green time for each phase
    ECIChangeTimingPhase(agents[index].id, 1, phaseDuration[0], timeSta)
    ECIChangeTimingPhase(agents[index].id, 3, phaseDuration[1], timeSta)
    ECIChangeTimingPhase(agents[index].id, 5, phaseDuration[2], timeSta)
    ECIChangeTimingPhase(agents[index].id, 7, phaseDuration[3], timeSta)
    # 4. Get Reward
    [reward, agents[index].oldDta] = GetReward.getReward(agents[index].oldDta, delayTime)
    # 5 .Create a dataset of agent experience
    if createDataSet and actionType == "best" and CreateDataSet.check_convergence(
            agents[index].counter[agents[index].state], reward):
        CreateDataSet.create_dataset(agents[index].state, agents[index].action, delayTime, density,
                                     longQueueInSection)
    # 6. Update Q-Table
    agents[index].qTable[agents[index].state][agents[index].action] = QLearning.updateQTable(
        agents[index].qTable[agents[index].state][agents[index].action],
        agents[index].qTable[currentState][currentAction], agents[index].state, agents[index].action,
        currentState, currentAction, reward, agents[index].learningRate, agents[index].discountFactor)
    # 7. Update learning rate and discount factor
    if agents[index].learningRate >= 0.01:
        agents[index].learningRate -= decayLearningRate
    if agents[index].discountFactor <= 0.9:
        agents[index].discountFactor += incrementDiscountFactor
    if agents[index].id == 549:
        AKIPrintString(
            "from " + str(agents[index].state) + " to " + str(currentState) + " | with action " + str(
                agents[index].action) + " | reward : " + str(reward) + " | action type : " + str(actionType))
    # 8. Set new state and action
    agents[index].counter[agents[index].state] += 1
    agents[index].state = currentState
    agents[index].action = currentAction


def AAPILoad():
    AKIPrintString("Load")
    return 0


def AAPIInit():
    AKIPrintString("Init")
    numberOfJunctions = AKIInfNetNbJunctions()
    global agents
    for index in range(numberOfJunctions):
        # Get attribute of network
        junctionId = AKIInfNetGetJunctionId(index)
        junctionIdSectionIn = []
        junctionIdSectionOut = []
        for j in range(1, ECIGetNumberSignalGroups(junctionId) + 1, 1):
            num_of_turning = ECIGetNumberTurningsofSignalGroup(junctionId, j)
            for k in range(num_of_turning):
                inputSectionId = intp()
                outputSectionId = intp()
                ECIGetFromToofTurningofSignalGroup(junctionId, j, k, inputSectionId, outputSectionId)
                junctionIdSectionIn.append(int(inputSectionId.value()))
                junctionIdSectionOut.append(int(outputSectionId.value()))
        junctionIdSectionIn = list(set(junctionIdSectionIn))
        junctionIdSectionOut = list(set(junctionIdSectionOut))
        controlType = ECIGetControlType(junctionId)
        numOfPhases = ECIGetNumberPhases(junctionId)
        # Initial Agent
        agents.append(QLearning.ReinforcementLearningAgent(junctionId, junctionIdSectionIn, junctionIdSectionOut,
                                                           controlType, numOfPhases, numberOfAction, numberOfState,
                                                           initLearningRate, initDiscountFactor))
    return 0


def AAPIManage(time, timeSta, timTrans, SimStep):
    return 0


def AAPIPostManage(time, timeSta, timTrans, SimStep):
    global tempTime
    global agents
    if (int(time) % cycle == 0) and (int(time) != tempTime) and (int(time) > warmup):
        tempTime = int(time)
        numberOfJunctions = AKIInfNetNbJunctions()
        threads = []
        for index in xrange(numberOfJunctions):
            t = threading.Thread(name='agent' + str(index), target=mainProcess, args=(index, timeSta,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    return 0


def AAPIFinish():
    AKIPrintString("Finish")
    return 0


def AAPIUnLoad():
    AKIPrintString("UnLoad")
    return 0


def AAPIEnterVehicle(idveh, idsection):
    return 0


def AAPIExitVehicle(idveh, idsection):
    return 0
