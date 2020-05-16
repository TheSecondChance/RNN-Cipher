import matplotlib.pyplot as plt
import numpy as np

def plot_ciphertext(ciphertext):
    V = []
    E = []

    for block in ciphertext:
        V.append(block[0][0][0])
        E.append([block[1][0][0], block[1][0][1]])
    
    E = np.array(E)

    X = np.arange(len(ciphertext))

    plt.plot(X, V, '-o', markersize=3)
    plt.title('Первая часть шифротекста')
    plt.xlabel('Номер блока шифротекста')
    plt.ylabel('Часть шифротекста V')
    plt.show()

    fig, ax = plt.subplots(2,1)
    fig.suptitle('Вторая часть шифротекста')
    ax[0].plot(X, E[:, 0], '-o', markersize=3)
    ax[0].set_ylabel('Часть шифротекста E0')

    ax[1].plot(X, E[:, 1], '-o', markersize=3)
    ax[1].set_xlabel('Номер блока шифротекста')
    ax[1].set_ylabel('Часть шифротекста E1')
    plt.show()