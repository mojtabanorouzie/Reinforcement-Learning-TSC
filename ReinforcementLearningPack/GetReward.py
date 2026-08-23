"""Reward: score the current delay against a sliding window of recent delay."""


def getReward(oldDta, delayTime):
    """Return [reward, updatedWindow] for the delay times just observed.

    `delayTime` is the average delay on each of the four incoming approaches;
    `oldDta` is the five-slot window of the previous cycles' scores.

    The junction is scored by its *worst* approach, max(delayTime), so an
    action that clears three approaches by starving the fourth earns nothing.
    The baseline is the harmonic mean of the window, which is pulled towards
    the window's smallest entries -- a harder target than the arithmetic mean.

    Reward is positive when the current worst delay is below that baseline,
    i.e. when things improved. The window is advanced with the new score before
    returning, so the caller must store what comes back.
    """
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
    """Harmonic-mean baseline over `array`, or 0 when every entry is zero.

    Zero entries are skipped in the sum of reciprocals rather than treated as
    an error, because the window starts out all-zero and fills in over the
    first five cycles. The divisor stays len(array) throughout, so while the
    window is still filling the result sits *above* the harmonic mean of the
    entries actually present -- an inflated baseline, and so an easier one to
    beat, for the first few cycles after warm-up.
    """
    result = 0
    for i in range(len(array)):
        if array[i] != 0:
            result += 1.0 / array[i]
    if result != 0:
        result = len(array) / result
    return result


def shiftRight(array):
    """Rotate `array` one place towards index 0, in place, and return it.

    Despite the name this rotates left: the oldest entry lands in the last
    slot rather than being dropped. That is deliberate and only correct
    because getReward immediately overwrites the last slot with the new score,
    which together make a five-slot FIFO window.
    """
    temp = array[0]
    for i in range(len(array) - 1):
        array[i] = array[i + 1]
    array[len(array) - 1] = temp
    return array
