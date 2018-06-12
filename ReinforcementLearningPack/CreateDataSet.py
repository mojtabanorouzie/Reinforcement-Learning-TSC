import csv


def check_convergence(counter, reward):
    flag = False
    if (counter > 30) and (reward > 0):
        flag = True
    return flag


def create_dataset(state, action, dta, density, longQueue):
    dataset = [[0 for i in range(14)]]
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
    with open("../dataset.csv", "ab") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for value in dataset:
            writer.writerow(value)
