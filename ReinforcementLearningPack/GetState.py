def getState(longQueueInSection):
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
            (longQueueInSection[3] >= longQueueInSection[0]) and (longQueueInSection[0] >= longQueueInSection[2]):
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
