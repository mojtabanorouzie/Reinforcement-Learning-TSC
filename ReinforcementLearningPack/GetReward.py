def getReward(oldDta, delayTime):
    # Harmonic Mean
    dta = max(delayTime)
    hm = harmonicMean(oldDta)
    # Shift Right
    oldDta = shiftRight(oldDta)
    oldDta[4] = dta
    reward = dta - hm
    if reward == 0:
        return [0, oldDta]
    else:
        return [-reward, oldDta]


def harmonicMean(array):
    result = 0
    for i in range(len(array)):
        if array[i] != 0:
            result += 1.0 / array[i]
    if result != 0:
        result = len(array) / result
    return result


def shiftRight(array):
    temp = array[0]
    for i in range(len(array) - 1):
        array[i] = array[i + 1]
    array[len(array) - 1] = temp
    return array
