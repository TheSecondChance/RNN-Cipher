import os
import time
import dataset
import string
import random
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES

from RNN_Cipher import RNN_Cipher

def rnn_test(plaintext, block_size):
    return

if __name__ == '__main__':
    BLOCK_SIZE = 16

    # Get training dataset for key expansion
    if not os.path.exists('dataset.csv'):
        x_train, y_train = dataset.create(BLOCK_SIZE, 10000)
        dataset.save((x_train, y_train), 'dataset.csv')
    else:
        x_train, y_train = dataset.load('dataset.csv')
    
    # Generate plaintext
    text_length = 10000
    symbols = string.printable
    plaintext = ''.join(random.choice(symbols) for i in range(text_length))
    

    key = b'Sixteen byte key'
    

    rnn = RNN_Cipher(False)
    rnn.KeyExpansion(x_train, y_train)
    
    start_time = time.time()

    padded_plaintext = pad(plaintext.encode(), BLOCK_SIZE)
    ciphertext_blocks = rnn.Encrypt(padded_plaintext)
    decrypted_text = rnn.Decrypt(ciphertext_blocks)
    decrypted_text = unpad(decrypted_text, BLOCK_SIZE)
    
    elapsed_time = time.time() - start_time

    # e_cipher = AES.new(key, AES.MODE_CBC)
    # plaintext = pad(plaintext.encode(), AES.block_size)
    # ciphertext = e_cipher.encrypt(plaintext)
    
    # d_cipher = AES.new(key, AES.MODE_CBC, e_cipher.IV)
    # decrypted_text = d_cipher.decrypt(ciphertext)
    # decrypted_text = unpad(decrypted_text, AES.block_size)

    print(f'\nВремя выполнения шифрования/расшифровки: {round(elapsed_time * 1000, 4)} мс.')