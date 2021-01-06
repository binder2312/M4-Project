{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FLnw42vmOzTV"
   },
   "source": [
    "## SDS 2020 - Module 3: Individual Assignment "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Assignment\n",
    "toc:true\n",
    "branch: master\n",
    "badges: true\n",
    "comments: true\n",
    "categories: [fastpages, jupyter]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "5VrnWb-uafdp"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras.models import Sequential\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PqQ4cf8lVZsv"
   },
   "source": [
    "## Data and preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "xtEFIUYWMbyc"
   },
   "outputs": [],
   "source": [
    "# Loading up data\n",
    "tweets = pd.read_json(\"https://github.com/SDS-AAU/SDS-master/raw/e2c959494d53859c1844604bed09a28a21566d0f/M3/assignments/trump_vs_GPT2.gz\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "c3CUpW7va5_8"
   },
   "outputs": [],
   "source": [
    "# Adding columnnames to the dataframe\n",
    "tweets.columns = [\"text\", \"status\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "BeVFAUL977-6",
    "outputId": "33070bc3-340c-40c3-e4fe-5040d150420d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        1\n",
       "1        1\n",
       "2        1\n",
       "3        1\n",
       "4        1\n",
       "        ..\n",
       "14731    0\n",
       "14732    0\n",
       "14733    0\n",
       "14734    0\n",
       "14735    0\n",
       "Name: status, Length: 14736, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Tranforming from boolean to integer - for the neural net\n",
    "tweets['status'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 203
    },
    "id": "BaGVvHfo2Bie",
    "outputId": "c7994636-6819-4fab-e3ea-c8e199be2f8a"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>status</th>\n",
       "      <th>length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>I was thrilled to be back in the Great city of...</td>\n",
       "      <td>True</td>\n",
       "      <td>210</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>The Unsolicited Mail In Ballot Scam is a major...</td>\n",
       "      <td>True</td>\n",
       "      <td>100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>As long as I am President, I will always stand...</td>\n",
       "      <td>True</td>\n",
       "      <td>82</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Our Economy is doing great, and is ready to se...</td>\n",
       "      <td>True</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>If I do not sound like a typical Washington po...</td>\n",
       "      <td>True</td>\n",
       "      <td>90</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                text  status  length\n",
       "0  I was thrilled to be back in the Great city of...    True     210\n",
       "1  The Unsolicited Mail In Ballot Scam is a major...    True     100\n",
       "2  As long as I am President, I will always stand...    True      82\n",
       "3  Our Economy is doing great, and is ready to se...    True      81\n",
       "4  If I do not sound like a typical Washington po...    True      90"
      ]
     },
     "execution_count": 5,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# I want to see the length of the tweets to check for outliers\n",
    "length = []\n",
    "[length.append(len(str(text))) for text in tweets['text']]\n",
    "tweets['length'] = length\n",
    "tweets.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "x9y4QERJ2hMn",
    "outputId": "8526c489-d91b-4252-f22a-1f8acee0c039"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "296\n",
      "163\n"
     ]
    }
   ],
   "source": [
    "# Seems like i have to remove some since the min value is 0\n",
    "print(min(tweets['length']))\n",
    "print(max(tweets['length']))\n",
    "print(round(sum(tweets['length'])/len(tweets['length'])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "QiH1uVbC3X9T",
    "outputId": "3dd9c09a-127f-4114-d6f3-a51baf13b938"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "408"
      ]
     },
     "execution_count": 7,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 408 tweets with a length under 10 words\n",
    "# I'm choosing this since i want a full sentence and not just single words\n",
    "len(tweets[tweets['length'] < 10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "ZiW0rLGD6bja"
   },
   "outputs": [],
   "source": [
    "# Keeping all rows with a length over or equal to 10 words\n",
    "tweets = tweets[tweets.length >= 10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "7XzxOhfueKPo"
   },
   "outputs": [],
   "source": [
    "# Instantiating of the tokenizer setting oov_token to True \n",
    "# any unknown words will be replaced\n",
    "# Also removing punctuation by default\n",
    "tokenizer = Tokenizer(lower=True, oov_token=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "RZ4gyMHY7kY-"
   },
   "outputs": [],
   "source": [
    "# Defining my dependent and independent variable\n",
    "# I want to predict the status 0 or 1, so that will be the dependent variable\n",
    "X = tweets['text']\n",
    "y = tweets['status']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "dsaV6ce48H41"
   },
   "outputs": [],
   "source": [
    "# Making the train test split for the model\n",
    "# Picking a test size of 0.3 - tried to adjust it, but result didn't get better\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "_2IjK-hEay7Y"
   },
   "outputs": [],
   "source": [
    "# Fitting the tokenizer on the training data \n",
    "tokenizer.fit_on_texts(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "H4xR0nbw9iKg"
   },
   "outputs": [],
   "source": [
    "# I have to fit the tokenizer on the data before i can get the vocab size\n",
    "# Here i have the size for the input_dim in the Embedding layer\n",
    "max_vocab = len(tokenizer.word_index)+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "babdjMxNcFOo"
   },
   "outputs": [],
   "source": [
    "# And then turning words into numbers/sequences for the neural net\n",
    "sequences_train = tokenizer.texts_to_sequences(X_train)\n",
    "sequences_test = tokenizer.texts_to_sequences(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "OhFFEHomiFxC"
   },
   "outputs": [],
   "source": [
    "# Furthermore padding the sequences to give them the same length \n",
    "# This is required when giving input to the neural net\n",
    "data_train = pad_sequences(sequences_train)\n",
    "data_test = pad_sequences(sequences_test, maxlen=data_train.shape[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fG0PKwNZT56r"
   },
   "source": [
    "## Building the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "4JbxeW54r78O"
   },
   "outputs": [],
   "source": [
    "# Using a Sequential model which can process sequences of integers which is what i got\n",
    "model = keras.Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "jjRp8gBG9zK1"
   },
   "outputs": [],
   "source": [
    "# Adding an Embedding layer to compress the input\n",
    "# Where the input size is equal to the vocabulary size + 1\n",
    "# Keeping a small number for output since it gave the best results\n",
    "model.add(layers.Embedding(max_vocab, 16))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "CV4Bgvu3-iXl"
   },
   "outputs": [],
   "source": [
    "# Adding BidirectionalLSTM layer because i want the model to read both back and forth\n",
    "# Where the normal LSTM model only reads from left to right\n",
    "# So this one should give a better result than the normal one\n",
    "# Also adding Dropout of 0.9 since i found that the model performed better with this\n",
    "model.add(layers.Bidirectional(layers.LSTM(8, return_sequences=True, dropout=0.9)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "id": "kmCHEdQY_3Ra"
   },
   "outputs": [],
   "source": [
    "# Output layer with sigmoid activation function\n",
    "model.add(layers.Dense(1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "kVGgOnOzZ3ay",
    "outputId": "bf13d348-854a-4319-b94d-5ac8143d7afd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "embedding (Embedding)        (None, None, 16)          209920    \n",
      "_________________________________________________________________\n",
      "bidirectional (Bidirectional (None, None, 16)          1600      \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, None, 1)           17        \n",
      "=================================================================\n",
      "Total params: 211,537\n",
      "Trainable params: 211,537\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# We can see there is 211.537 trainable parameters \n",
    "# I found that when trying with over 1 million parameters the model gets overfitted real quick\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b5xTNro_Uwpd"
   },
   "source": [
    "## Training and evaluation of the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "G8Bg9Xu___K2"
   },
   "outputs": [],
   "source": [
    "# Using binary_crossentropy since there are only two label classes 0 and 1\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "IHAAyT2DAMIV",
    "outputId": "11bf32db-fc36-4a22-8687-803673196421"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/6\n",
      "157/157 [==============================] - 3s 19ms/step - loss: 0.6148 - accuracy: 0.6818 - val_loss: 0.4837 - val_accuracy: 0.8163\n",
      "Epoch 2/6\n",
      "157/157 [==============================] - 2s 15ms/step - loss: 0.4795 - accuracy: 0.8043 - val_loss: 0.4379 - val_accuracy: 0.8270\n",
      "Epoch 3/6\n",
      "157/157 [==============================] - 2s 15ms/step - loss: 0.4384 - accuracy: 0.8196 - val_loss: 0.3953 - val_accuracy: 0.8462\n",
      "Epoch 4/6\n",
      "157/157 [==============================] - 2s 15ms/step - loss: 0.4003 - accuracy: 0.8390 - val_loss: 0.3772 - val_accuracy: 0.8491\n",
      "Epoch 5/6\n",
      "157/157 [==============================] - 2s 15ms/step - loss: 0.3732 - accuracy: 0.8518 - val_loss: 0.3602 - val_accuracy: 0.8547\n",
      "Epoch 6/6\n",
      "157/157 [==============================] - 2s 15ms/step - loss: 0.3514 - accuracy: 0.8588 - val_loss: 0.3466 - val_accuracy: 0.8591\n"
     ]
    }
   ],
   "source": [
    "# I found that the number of epochs fits on 6 and also batch size of 64\n",
    "# I have tried many different setups, and in many cases i ended up overfitting the training set\n",
    "r = model.fit(data_train, y_train, epochs=6, validation_data=(data_test, y_test), batch_size=64)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "aa_7vXlwdp7P"
   },
   "source": [
    "I can see here that the loss score is decreasing for both the training and test set in each epoch as well as the accuracy is increasing which is the goal for this step here where i want to achieve a low loss and a high accuracy meaning that the model is as precise as it can be. \n",
    "The reason why i didn't choose more epochs is because the training set starts to get overfitted from here. PLEASE BE AWARE THAT THE val_accuracy IS FOUND UNDER THE 'Epoch 1/6 157/157'."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 293
    },
    "id": "FA6otSvdK690",
    "outputId": "74122ff6-11f9-4e79-bc8b-a2817a08c077"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light",
      "tags": []
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Now i want to compare the train and test results in the following two graphs\n",
    "plt.plot(r.history['accuracy'])\n",
    "plt.plot(r.history['val_accuracy'])\n",
    "plt.title('Model Accuracy')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xlabel('Epoch', labelpad=2)\n",
    "plt.legend(['train', 'test'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "iY2RHfQfRSd3"
   },
   "source": [
    "From this i can see that both the train and test set is improving over the 6 epochs and then ending up being approximately the same in the last epoch. (0, 1, 2, 3, 4, 5 - so six)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 295
    },
    "id": "IAXEUdn_f5zV",
    "outputId": "d6d0c695-8bcd-4526-ab73-b97c93f04d0a"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light",
      "tags": []
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Taking a look at the loss score here\n",
    "plt.plot(r.history['loss'])\n",
    "plt.plot(r.history['val_loss'])\n",
    "plt.title('Loss')\n",
    "plt.ylabel('Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.legend(['train', 'test'], loc='upper right')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YIOf3P_CRg5s"
   },
   "source": [
    "The same is the case for the loss score, where the graphs are almost meeting each other in the 6th epoch which means that it is the near perfect fit. \n",
    "\n",
    "I had a lot of trouble finding a good match, where in many cases the model ended up being overfit - being that the training set was improving really quick ending on 0.98 on the accuracy score but with high loss score on the validation set - which should be as low as possible. \n",
    "I have both tried adding more layers to the model and also adjusting the number of neurons in the layers, but i found that keeping the model on few layers with few neurons and a high dropout rate gave me the best performing neural net. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "L1g3W9svSPwx"
   },
   "source": [
    "So this means that the model will be able to detect a fake tweet with an accuracy of 86%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "D8HMAN0ma6QE"
   },
   "outputs": [],
   "source": [
    "!jupyter nbconvert --to html \"/content/M3A1.ipynb\""
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "name": "M3A1",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
