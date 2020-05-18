import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_ciphertext(ciphertext):
    V = []
    E = []

    for block in ciphertext:
        V.append(block[0][0][0])
        E_block = []
        for e in block[1][0]:
            E_block.append(e)
        E.append(E_block)

    E_block_len = len(block[1][0])
    if E_block_len == 2:
        shape = (2,1)
    elif E_block_len <= 4:
        shape = (2,2)
    elif E_block_len <= 8:
        shape = (3,3)
    else:
        shape = (4,4)

    E = np.array(E)
    X = np.arange(len(ciphertext))

    plt.plot(X, V, '-o', markersize=3)
    plt.xlabel('N')
    plt.ylabel('V')
    plt.show()

    
    fig, axes = plt.subplots(shape[0], shape[1])
    if shape[1] == 1:
        for i in range(E_block_len):
            axes[i].plot(X, E[:,i], '-o', markersize=3)
            axes[i].set_ylabel(f'E{i}')
            axes[i].set_xlabel('N')
    else:
        for l in range(E_block_len):
            i = l // shape[1]
            j = l % shape[1]
            axes[i][j].plot(X, E[:,l], '-o', markersize=3)
            axes[i][j].set_ylabel(f'E{l}')
            axes[i][j].set_xlabel('N')
    plt.tight_layout()
    plt.show()