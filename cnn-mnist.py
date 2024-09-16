#!/usr/bin/env python
# coding: utf-8

# # Classificação do MNIST com Rede Neural Convolucional (CNN)

# ## Descrição:
# Este projeto busca implementar os conceitos básicos de CNNs aplicados à classificação de imagens do banco de dados MNIST e utilizando a biblioteca TensorFlow.

# ### Importação das bibliotecas

# In[11]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ### Obtenção dos dados

# In[2]:


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)


# ### Construção do modelo
# O modelo segue a estrutura da CNN do diagrama abaixo:
# ![](https://camo.githubusercontent.com/6b0272073260037a2ce54ec90e6640ea0192fb7b3bfbbd5518eb340adcea9262/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f61616d696e692f696e74726f746f646565706c6561726e696e672f6d61737465722f6c6162322f696d672f636f6e766e65745f6669672e706e67)

# In[26]:


def build_cnn_model():
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(24, 3, activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(36, 3, activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    return cnn_model


# ### Treinamento do modelo

# In[29]:


cnn_model = build_cnn_model()

# Configura o modelo
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'],
    jit_compile=True
)

BATCH_SIZE = 64
EPOCHS = 5

# Treina o modelo com os dados de treino
cnn_model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Avalia a precisão com o os dados de teste
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)

print('\nTest accuracy:', test_acc)


# In[5]:


predictions = cnn_model.predict(test_images)


# ### Verificação das previsões
# Escolha um índice arbitrário de imagem e verifique a previsão do modelo!

# In[36]:


index = 95 # Insira o índice aqui
print("Predicted label:", predictions[index].argmax())
print("True label:", test_labels[index])
plt.imshow(test_images[index,:,:,0], cmap=plt.cm.binary)


# # Referências
# 
# Baseado no segundo Lab do curso do MIT "Introduction to Deep Learning"
# 
# 
# [Repositório do exercício](https://github.com/aamini/introtodeeplearning/tree/master/lab2)
# 
# [Documentação do TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras)
# 
# © MIT Introduction to Deep Learning
# http://introtodeeplearning.com
# 

# In[ ]:




