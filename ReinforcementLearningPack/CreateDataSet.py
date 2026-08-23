"""Export settled agent experience as labelled rows, for later use off-line.

Once an agent has stopped exploring a given state, the action it greedily
picks there is its learned answer for that traffic situation. This module
writes those answers out alongside the raw measurements that produced them,
one CSV row per decision, so the mapping can be learned by something other
than a Q-table.

Nothing downstream of that lives in this repository: it produces the data and
stops. See the README for what was built on top of it.
"""

import csv

# A state's Q-values are treated as settled once it has been visited this many
# times; below that, a greedy action is still mostly an artefact of
# initialisation and is not worth exporting.
MIN_VISITS_BEFORE_EXPORT = 30

# 1 state + 4 delay times + 4 densities + 4 queue lengths + 1 action label.
DATASET_COLUMNS = 14

# Written relative to the process working directory, which for an AAPI script
# is Aimsun's, not this package's.
DATASET_PATH = "../dataset.csv"


def check_convergence(counter, reward):
    """Is this decision worth exporting as a training example?

    `counter` is how many times the agent has been in this state and `reward`
    the reward the decision earned. Both gates matter: the state must have
    been seen enough times for its Q-values to mean something, and the
    decision must have actually improved delay. Main.py adds a third gate at
    the call site -- only greedy actions, never exploratory ones.
    """
    flag = False
    if (counter > MIN_VISITS_BEFORE_EXPORT) and (reward > 0):
        flag = True
    return flag


def create_dataset(state, action, dta, density, longQueue):
    """Append one experience row to the dataset file.

    Columns, in order (no header row is written):

        0       state index, 0..23
        1-4     delay time per approach
        5-8     density per approach
        9-12    maximum queue length per approach
        13      action index, the label

    Keeping both the raw measurements and the state index is the point: a
    model trained on columns 1-12 is not confined to the 24-state abstraction
    the agent had to use.

    Note that the row straddles two control cycles. Main.py passes the
    previous cycle's state and action for columns 0 and 13, but the
    measurements read on the current cycle for columns 1-12 -- so column 0 is
    not the ranking of the values beside it. Left as it is because the code
    gives no indication of which alignment was meant; see the README.
    """
    dataset = [[0 for i in range(DATASET_COLUMNS)]]
    dataset[0][0] = state
    dataset[0][13] = action
    j = 5
    k = 9
    for i in range(4):
        dataset[0][i + 1] = dta[i]
        dataset[0][j] = density[i]
        dataset[0][k] = longQueue[i]
        j += 1
        k += 1
    with open(DATASET_PATH, "ab") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for value in dataset:
            writer.writerow(value)
