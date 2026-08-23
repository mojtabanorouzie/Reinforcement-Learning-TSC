"""Aimsun API script: one tabular RL agent per signalised junction.

Aimsun loads this file as an AAPI ("Aimsun API") extension and calls the
AAPI* hooks below at fixed points in the simulation. The module is not a
program -- it has no __main__ and cannot run standalone. It requires a
licensed Aimsun installation, its compiled `_AAPI` extension module, and
Python 2.7.

Lifecycle, in call order:

    AAPILoad        once, when the script is loaded
    AAPIInit        once, before the simulation starts. Walks the network,
                    resolves each junction's incoming and outgoing sections
                    from its signal groups, and creates one
                    ReinforcementLearningAgent per junction.
    AAPIPostManage  every simulation step. Every `cycle` seconds after the
                    warm-up, runs one control decision per junction, each on
                    its own thread.
    AAPIFinish      once, when the simulation ends
    AAPIUnLoad      once, when the script is unloaded

Each agent observes the queue lengths on its four incoming approaches,
picks one of `numberOfAction` green-time splits, applies it to the
junction's four green phases, and updates its Q-table from the change in
delay time. See mainProcess for the per-decision sequence.
"""

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

# Junction whose decisions are traced to the Aimsun log. Tracing every
# junction on every cycle floods the log, so one is singled out. The value is
# an Aimsun junction id and is specific to the network this was run against.
debugJunctionId = 549

# Indices of the four green phases in the Aimsun signal plan. The even-numbered
# phases in between are inter-green and are deliberately left untouched, which
# is what keeps the cycle length fixed while the splits vary.
greenPhases = [1, 3, 5, 7]


def mainProcess(index, timeSta):
    """Run one control decision for agent `index` and update its Q-table.

    Called on its own thread once per control cycle. `timeSta` is Aimsun's
    simulation time, forwarded to ECIChangeTimingPhase.

    The numbered steps below match the algorithm: observe, encode a state,
    choose and apply an action, score the previous action, optionally export
    the experience, then apply the temporal-difference update. Note the
    off-by-one that this ordering implies and that is intended: the reward and
    the update are attributed to the state/action pair from the *previous*
    cycle (agents[index].state / .action), because only now is its effect on
    delay time observable.

    Each thread reads and writes only agents[index], so agents do not race
    against each other -- but see CreateDataSet.create_dataset, which appends
    to one shared file from all of them.
    """
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
    for i in range(len(greenPhases)):
        ECIChangeTimingPhase(agents[index].id, greenPhases[i], phaseDuration[i], timeSta)
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
    if agents[index].id == debugJunctionId:
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
    """Create one agent per junction, wired to that junction's approaches.

    For each junction, every turning of every signal group is queried for its
    (from, to) sections; the de-duplicated "from" set becomes the agent's
    incoming approaches. mainProcess then reads statistics from the first four
    of them, so this assumes four-approach junctions.
    """
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
    """Fire one control decision per junction, once per `cycle` seconds.

    Aimsun calls this every simulation step, so the body is guarded three
    ways: the step must land on a cycle boundary, must not repeat a cycle
    already handled (Aimsun can call back more than once for the same second),
    and must be past the `warmup` period during which the network fills up.

    Junctions are then processed concurrently, one thread each, and joined
    before returning so no decision spills into the next simulation step.
    """
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
