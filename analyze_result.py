#! /bin/python3

import json
import os

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

import numpy as np

if __name__ == "__main__":
    try:
        os.mkdir((f'_pics/total/'))
    except Exception as e:
        print(e)

    with open('result_2_64.json', 'r') as file:
        data = json.load(file)

    x_array = []
    y_array = []
    z1_array = []
    z2_array = []

    for x, yz in data.items():
        for y, array in yz.items():
            x_array.append(int(x))
            y_array.append(int(y))
            z1_array.append(array[0][2])
            z2_array.append(array[1][2])

    n = len(set(x_array))
    m = len(set(y_array))
    z1 = np.zeros(shape=(n, m), dtype=np.uint32)
    z2 = np.zeros(shape=(n, m), dtype=np.uint32)
    for i in range(len(x_array)):
        z1[x_array[i]  * n // 8][y_array[i] * m // 1024] = z1_array[i]
        z2[x_array[i]  * n // 8][y_array[i] * m // 1024] = z2_array[i]

    fig, ax = plt.subplots()
    ax.hist(z1_array, bins=10)
    ax.set_title("Количество изменённых строк в первом методе")
    plt.savefig(f'_pics/total/m1_chanded_hist.png')

    fig, ax = plt.subplots()
    ax.hist(z2_array, bins=10)
    ax.set_title("Количество удалённых строк во втором методе")
    plt.savefig(f'_pics/total/m2_deleted_hist.png')

    fig, ax = plt.subplots()
    im = ax.imshow(z1)

    ax.set_xticks(range(m), labels=set(y_array))
    ax.set_yticks(range(n), labels=set(x_array))

    ax.set_title("Количество изменённых строк в первом методе")
    plt.xlabel("Ячейки")
    plt.ylabel("Каналы")
    fig.tight_layout()
    ax=plt.gca()
    for PCM in ax.get_children():
        if isinstance(PCM, mpl.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax) 
    plt.savefig(f'_pics/total/m1_chanded_hm.png')

    fig, ax = plt.subplots()
    im = ax.imshow(z2)

    ax.set_xticks(range(m), labels=set(y_array))
    ax.set_yticks(range(n), labels=set(x_array))

    ax.set_title("Количество удалённых строк во втором методе")
    plt.xlabel("Ячейки")
    plt.ylabel("Каналы")
    fig.tight_layout()
    ax=plt.gca()
    for PCM in ax.get_children():
        if isinstance(PCM, mpl.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax) 
    plt.savefig(f'_pics/total/m2_chanded_hm.png')

    print("First:")
    print("\tMean - ", z1.mean())
    print("\tMedian - ", np.median(z1))
    print("\tMax - ", z1.max())
    print("\tMin - ", z1.min())

    print("Second:")
    print("\tMean - ", z2.mean())
    print("\tMedian - ", np.median(z2))
    print("\tMax - ", z2.max())
    print("\tMin - ", z2.min())
