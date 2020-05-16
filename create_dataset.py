import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

def convert_to_bitarray(hexstring):
    bits_number = int(len(hexstring) / 2 * 8)
    binstring = bin(int(hexstring, base=16))[2:].zfill(bits_number)
    bitarray = [int(i) for i in list(binstring)] 
    return bitarray

def convert_to_hexstring(bitarray):
    hexchars_number = int(len(bitarray) / 8) * 2
    binstring = ''.join([str(b) for b in bitarray])
    hexstring = hex(int(binstring, base=2))[2:].zfill(hexchars_number)
    return hexstring

def get_data():
    KEY = bytes.fromhex('3541c8a113caf6e13f6c8fb694fb94ac')

    with open('training_data/plaintext.txt', 'rb') as f:
        plaintext = f.read()

    plaintext = pad(plaintext, 16)

    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = cipher.encrypt(plaintext)

    plaintext_bytearray = np.array(list(plaintext))
    ciphertext_bytearray = np.array(list(ciphertext))

    BLOCK_SIZE = 2

    x_train = np.reshape(plaintext_bytearray, [int(len(plaintext_bytearray)/BLOCK_SIZE), BLOCK_SIZE])
    y_train = np.reshape(ciphertext_bytearray, [int(len(ciphertext_bytearray)/BLOCK_SIZE), BLOCK_SIZE])

    x_train = x_train / 256
    y_train = y_train / 256

    return (x_train, y_train)