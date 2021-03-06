import os
import numpy as np
import tensorflow as tf
from Crypto.Util.Padding import pad, unpad

import dataset
from graphs import *

class RNN_Cipher:
    def __init__(self, use_saved_weights=True):    
        # Hyperparameters
        self.epochs = 4
        self.learning_rate = tf.Variable(1.0, shape=[])
        self.initial_weights = 0.5
        self.initial_biases = 0.0
        self.critical_rate = 0.04

        # Layers configuration
        self.layers_config = [16, 16, 1, 8]
        self.V_index = self.layers_config.index(1)
        
        # Block size
        self.block_size = self.layers_config[-1]

        # Placeholders for computational graphs
        self.X = tf.placeholder(tf.float32, [None, self.layers_config[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.layers_config[-1]])
        self.V = tf.placeholder(tf.float32, [None, 1])

        # Neural network weights
        self.weights = {}
        for i in range(1, len(self.layers_config)):
            weights = tf.Variable(tf.constant(self.initial_weights, shape=(self.layers_config[i-1], self.layers_config[i])))
            if i == self.V_index:
                self.weights['v'] = weights
            elif i == len(self.layers_config) - 1:
                self.weights['out'] = weights
            else:
                self.weights[f'h{i}'] = weights

        # Neural network biases
        self.biases = {}
        for i in range(1, len(self.layers_config)):
            biases = tf.Variable(tf.constant(self.initial_biases, shape=(self.layers_config[i],)))
            if i == self.V_index:
                self.biases['v'] = biases
            elif i == len(self.layers_config) - 1:
                self.biases['out'] = biases
            else:
                self.biases[f'b{i}'] = biases

        self.network = self.Network(self.X)
        self.loss = tf.losses.mean_squared_error(self.Y, self.Network(self.X))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # Final output from key expansion process
        self.M0 = tf.Variable([[0.0] * self.block_size], shape=(1, self.block_size), dtype=tf.float32)
        
        # Initial value for learning adaptation process
        self.T_prev = 0

        # Initialize global session
        self.isWeightsLoaded = False     
        self.session = tf.Session()
        if not os.path.exists('saved_sessions/') or use_saved_weights == False:
            init = tf.global_variables_initializer()
            self.session.run(init)
        else:
            print('\n[!] Restoring weights from file...')
            tf.train.Saver().restore(self.session, 'saved_sessions/rnn-cipher')
            self.isWeightsLoaded = True
            print('[*] Pretrained session initialized!')
    
    def Summary(self):
        row_format = '{:<20}' * 3
        header = row_format.format('Слой', 'Размер', 'Число параметров')
        print('_'*60)
        print(header)
        print('='*60)
        total_params = 0
        hidden_index = 1
        for i in range(len(self.layers_config)):
            if i == 0:
                print(row_format.format('Входной', self.layers_config[i], 0))
                print('_'*60)
            else:
                params = self.layers_config[i-1] * self.layers_config[i] + self.layers_config[i]
                total_params += params
                if i == len(self.layers_config) - 1:              
                    print(row_format.format('Выходной', self.layers_config[i], params))
                    print('='*60)
                else:             
                    print(row_format.format(f'Скрытый {hidden_index}', self.layers_config[i], params))
                    print('_'*60)
                    hidden_index += 1
        print(f'Всего параметров: {total_params}')
        print('_'*60)

    # Neural network output computation graph
    def Network(self, X):
        H = X
        for i in range(1, len(self.layers_config)):
            if i == self.V_index:
                V = tf.sigmoid(tf.matmul(H, self.weights['v']) + self.biases['v'])
                H = V
            elif i == len(self.layers_config) - 1:
                OUT = tf.sigmoid(tf.matmul(H, self.weights['out']) + self.biases['out'])
            else:
                H = tf.sigmoid(tf.matmul(H, self.weights[f'h{i}']) + self.biases[f'b{i}'])   
        return OUT    
    
    def KeyExpansion(self, x_train, y_train):       
        if self.isWeightsLoaded == True:
            print('\n[*] Symmeric encryption key was already set!')
        else:
            print('\n[!] Starting key expansion process...')
            for epoch in range(self.epochs):
                print(f'\nEpoch #{epoch + 1}')
                
                Y = np.array([0.0] * self.block_size, dtype=np.float32)
                iteration = 0

                for m, y in zip(x_train, y_train):
                    # Perform concatenation of output from previous iteration and current input
                    x = np.array(np.concatenate((Y, m), axis=None), dtype=np.float32)
                    
                    x_batch = np.array([x], dtype=np.float32)
                    y_batch = np.array([y], dtype=np.float32)
                    
                    # Calculate feed-forward output from network
                    Y = self.session.run(self.network, feed_dict={self.X: x_batch})
                    # Adjust weights and biases
                    self.session.run(self.optimizer, feed_dict={self.X: x_batch, self.Y: y_batch})

                    iteration += 1
                    if iteration % 1000 == 0:
                        print(f'Iteration #{iteration}/{len(x_train)}')
            
            print(f'\n[*] Key expansion finished!')
            
            # Saving IV for encryption process and all variables
            self.session.run(tf.assign(self.M0, Y))
            tf.train.Saver().save(self.session, 'saved_sessions/rnn-cipher')
            print('[*] Weights were saved to a file!')      
        
        return
    
    # Encryption computation graph
    def EncryptBlock(self, X):
        H = X
        for i in range(1, len(self.layers_config)):
            if i == self.V_index:
                V = tf.sigmoid(tf.matmul(H, self.weights['v']) + self.biases['v'])
                H = V
            elif i == len(self.layers_config) - 1:
                OUT = tf.sigmoid(tf.matmul(H, self.weights['out']) + self.biases['out'])
            else:
                H = tf.sigmoid(tf.matmul(H, self.weights[f'h{i}']) + self.biases[f'b{i}'])   
        return (V, OUT)
    
    def Encrypt(self, plaintext):
        # Prepare plaintext for encryption
        plaintext_bytes = np.array(list(plaintext))
        scaled_plaintext = plaintext_bytes / 255 # Scale bytes
        plaintext_blocks = np.reshape(scaled_plaintext, [int(len(scaled_plaintext) / self.block_size), self.block_size])
             
        ciphertext_blocks = []
        encrypt_block = self.EncryptBlock(self.X)
        Y = self.session.run(self.M0)[0]

        for block in plaintext_blocks:                            
            x = np.array(np.concatenate((Y, block), axis=None), dtype=np.float32)
            x_batch = np.array([x], dtype=np.float32)
            
            # Encrypt one plaintext block
            V, Y = self.session.run(encrypt_block, feed_dict={self.X: x_batch})
            E = block - Y        
            ciphertext_blocks.append((V, E))

            # One iteration of learning process
            y_batch = np.array([block], dtype=np.float32)
            MSE = self.session.run(self.loss, feed_dict={self.X: x_batch, self.Y: y_batch})
            self.session.run(self.optimizer, feed_dict={self.X: x_batch, self.Y: y_batch})

            # Adaptation of learning process
            self.LearningAdaptation(MSE)
        
        # Restore KeyExpansion weights
        tf.train.Saver().restore(self.session, 'saved_sessions/rnn-cipher')
        
        return ciphertext_blocks
    
    # Decryption computation graph
    def DecryptBlock(self, V):
        H = V
        for i in range(self.V_index + 1, len(self.layers_config)):
            if i == len(self.layers_config) - 1:
                OUT = tf.sigmoid(tf.matmul(H, self.weights['out']) + self.biases['out'])
            else:
                H = tf.sigmoid(tf.matmul(H, self.weights[f'h{i}']) + self.biases[f'b{i}'])
        return OUT
    
    def Decrypt(self, ciphertext_blocks):
        # Restore KeyExpansion weights
        tf.train.Saver().restore(self.session, 'saved_sessions/rnn-cipher')
        
        # Restore original learning rate
        self.DropAdaptation()

        plaintext_blocks = []
        decrypt_block = self.DecryptBlock(self.V)
        
        Y_prev = self.session.run(self.M0)

        for block in ciphertext_blocks:
            Y = self.session.run(decrypt_block, feed_dict={self.V: block[0]})
            M = Y + block[1]
            plaintext_blocks.append(M[0])

            # One iteration of learning process
            X = np.array([np.concatenate((Y_prev[0], M[0]), axis=None)])
            MSE = self.session.run(self.loss, feed_dict={self.X: X, self.Y: M})
            self.session.run(self.optimizer, feed_dict={self.X: X, self.Y: M})
            
            Y_prev = Y

            # Adaptation of learning process
            self.LearningAdaptation(MSE)

        # Convert plaintext blocks to string of bytes
        plaintext_blocks = np.array(np.array(plaintext_blocks) * 255) # Unscale bytes
        plaintext_blocks = np.rint(plaintext_blocks).astype(int) # Convert bytes to int
        plaintext_bytearray = plaintext_blocks.flatten()
        plaintext = bytes(plaintext_bytearray.tolist())
        
        return plaintext

    def LearningAdaptation(self, MSE):
        delta = 0.2
        current_LR = self.session.run(self.learning_rate)

        T = delta * self.T_prev + (1 - delta) * MSE
        if T <= self.critical_rate:
            new_LR = current_LR * 2.0
        elif T > self.critical_rate and T > self.T_prev:
            new_LR = current_LR * 0.9
        else:
            new_LR = current_LR
        
        # Update learning rate for optimizer
        self.session.run(tf.assign(self.learning_rate, new_LR))
        self.T_prev = T
    
    def DropAdaptation(self):
        self.T_prev = 0
        self.session.run(tf.assign(self.learning_rate, 1.0))

if __name__ == '__main__':
    BLOCK_SIZE = 8
    
    # Get training dataset for key expansion
    if not os.path.exists('dataset.csv'):
        x_train, y_train = dataset.create(BLOCK_SIZE, 1000)
        dataset.save((x_train, y_train), 'dataset.csv')
    else:
        x_train, y_train = dataset.load('dataset.csv')

    cipher = RNN_Cipher(False)
    cipher.Summary()
    cipher.KeyExpansion(x_train, y_train)
    
    # plaintext = b'Artificial neural networks (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains.'
    plaintext = b'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAzzzzzzzzzzzzzzzzzzzz' * 10
    plaintext = pad(plaintext, BLOCK_SIZE)
    
    ciphertext_blocks = cipher.Encrypt(plaintext)

    plot_ciphertext(ciphertext_blocks)

    decrypted_text = cipher.Decrypt(ciphertext_blocks)
    decrypted_text = unpad(decrypted_text, BLOCK_SIZE)
    print(decrypted_text.decode())