# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:26:52 2023

@author: Pranav
"""

import pandas as pd

import torch

import numpy as np

import torch.nn as nn

import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# Reading English-Hindi data pairs from aksharantar dataset
df = pd.read_csv('./hin/hin_train.csv')
df.columns = ['English', 'Hindi']

# val_data = pd.read_csv('H:/My Drive/Courses/Jan-May 2023/Deep_Learning/Assignment 3/aksharantar_sampled/aksharantar_sampled/hin/hin_val.csv')
# val_data.columns = ['English', 'Hindi']

PAD_token = 0
SOS_token = 1
EOS_token = 2

df['length_Hindi_words'] = df['Hindi'].apply(lambda x: len([*x]))
df['length_English_words'] = df['English'].apply(lambda x: len([*x]))


df = df[(df['length_English_words'] < 15) & (df['length_Hindi_words'] < 14)]
class Lang:
  '''
  This class add letters from the language to make a dictionary of 
  letters in the language. Corresponding to each letter there is a associated 
  index. This index is unique for each letter in the dictionary. 
  '''
  def __init__(self, name):
      self.name = name
      self.letter2index = {}
      self.letter2count = {}
      self.index2letter = {}
      self.index2letter = { 0: "PAD", 1: "SOS", 2: "EOS"}
      self.n_letters = 3
      self.max_size = 0


  def addWord(self, word):
      if self.max_size < len(word):
          self.max_size = len(word)
      for letter in [*word]:
          self.addLetter(letter)
          
          

  def addLetter(self, letter):
      if letter not in self.letter2index:
          self.letter2index[letter] = self.n_letters
          self.letter2count[letter] = 1
          self.index2letter[self.n_letters] = letter
          self.n_letters += 1
      else:
          self.letter2count[letter] += 1


def lettertonumber(df, lang):
    language = Lang(lang)

    for letter in df[lang]:
        language.addWord(letter)
  
    return language

# Make dictionary for english words
english = lettertonumber(df, 'English')
# Make dictionary for Hindi words
hindi = lettertonumber(df, 'Hindi')

def word2number(word, language_dict):
    
    '''
    Args:
        word: The word in the language
        language_dict: The object of Lang class.
    Returns:
        tensor of size equal to the number of the letter in the word with
        elements being the index corresponding to letters in the dictionary.
    '''
    
    rep = torch.zeros(language_dict.max_size + 1, dtype = torch.long)
    for index, letter in enumerate(word):
        rep[index] = language_dict.letter2index[letter]
        
    rep[index + 1] = EOS_token
    return rep


def dataset(df, lang1 = english, lang2 = hindi):

    df["English Representation"] = df["English"].apply(lambda x: word2number(x, lang1))
    df["Hindi Representation"] = df["Hindi"].apply(lambda x: word2number(x, lang2))
    
    return df
    

df1 = dataset(df)

df1.reset_index(inplace = True)

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        input_tensor = self.df.loc[idx, 'English Representation']
        target_tensor = self.df.loc[idx, 'Hindi Representation']
        
        
        return input_tensor, target_tensor
    
    
from torch.utils.data import DataLoader

dataset = CustomDataset(df1)
data_loader = DataLoader(dataset, batch_size = 20, shuffle = True, drop_last = True)

i, t = next(iter(data_loader))

import torch.nn as nn

# Write embedding and check the output dimension of embedding and see the result.

#embedding = nn.Embedding(english.n_letters, 120, padding_idx = 0)
#ihat = embedding(i) # output shape == batch x length_sequence x n_embeddings
#ihat = ihat.view(-1, 20, 120)

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size,
                 num_layers,
                 batch_size,
                 verbose = False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        #input_size = Vocabulary size of the language
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.num_layers = num_layers
        self.verbose = verbose
    
    def forward(self, input, hidden):
        if self.verbose == True:
            print(f'Input Shape: {input.shape}')
        # Size of Input = (L, )
        # The view(-1, 1) reshapes the input to size (n, 1)
        
        embedded = self.embedding(input)
        # Size of embedded = (L x 1 x hidden_size)
        output = embedded.permute(1, 0, 2)
        # Size of embded is now change to (L x batch_size x hidden_size)
        
        if self.verbose == True:
            print(f'Embedding Shape: {embedded.shape}')
        output, hidden = self.gru(output, hidden)
        # Size of output = (L x batch_size x hidden_size)
        # Size of hidden = (1 x batch_size x hidden_size)
        
        if self.verbose == True:
            print('Shape of Input:', input.shape)
            print('Shape of embedded:', embedded.shape)
            print('Shape of output:', output.shape)
            print('Shape of hidden:', hidden.shape)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device = device)
    
class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size,
                 num_layers,
                 batch_size, verbose = True):
        
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.num_layers = num_layers
        self.verbose = verbose
        self.batch_size = batch_size
        
    def forward(self, input, hidden):
        # Size of input = (1, 1) # One word at a time 
        #print('Size of input:', input.shape)
        embed = self.embedding(input)
        # Size of embed = (1 x 1 x hidden_size)
        #print('Size of embed: ', embed.shape)
        output = self.relu(embed).view(1, decoder.batch_size, -1)
        
        # size of hidden = (1 x 1 x 120)
        output, hidden = self.gru(output, hidden)
        # size of output = (1 x 1 x hidden_size)
        output = self.softmax(self.out(output[0]))
        # size of output = (1 x V)
        if self.verbose == True:
            print('Shape of Input:', input.shape)
            print('Shape of embed:', embed.shape)
            print('Shape of output:', output.shape)
            print('Shape of hidden:', hidden.shape)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device = device)
    
import torch.optim as optim

def train(input_tensor, target_tensor, encoder, decoder,
              encoder_optimizer, decoder_optimizer, criterion,
              teacher_forcing = True, lang = hindi, verbose = False):
    
    encoder_hidden = encoder.initHidden()
    
    loss = 0
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #input_length = input_tensor.size(0) # L x 1 x features
    target_length = target_tensor.size(1) # L x 1 x features
    
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    decoder_input = torch.tensor([SOS_token] * decoder.batch_size, device = device)
    
    decoder_hidden = encoder_hidden #Size = (1, 1)
    
    predicted_output = torch.zeros(target_tensor.shape[0],
                                   target_tensor.shape[1], dtype = torch.long, 
                                   device = device)
    

        
    for i in range(target_length):
        
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        predicted_output[:, i: i+1] = topi
        #pdb.set_trace()
        loss += criterion(decoder_output, target_tensor[:, i])
        if teacher_forcing:
          decoder_input = target_tensor[:, i]
        else:
          decoder_input = topi
            
    
    #print(target_tensor)
    
    if verbose:
        #pdb.set_trace()
        true_hindi_word = [lang.index2letter[elements.item()] for elements in target_tensor[0, :]]
        predicted_hindi_word = [lang.index2letter[elements.item()] for elements in predicted_output[0, :]]
        print(len(true_hindi_word), len(predicted_hindi_word))
        true_hindi_word = ''.join(true_hindi_word)
        predicted_hindi_word = ''.join(predicted_hindi_word)
        print(true_hindi_word, predicted_hindi_word)
    
            
    try:
      loss.backward()
    except:
      print(loss)  
    
    word_accuracy = predicted_output == target_tensor
    word_accuracy = sum(torch.all(word_accuracy, dim = 1)).item()
    letter_accuracy = torch.sum(predicted_output == target_tensor)
    
    #print(predicted_output)
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item(), word_accuracy, letter_accuracy

import matplotlib.pyplot as plt

from tqdm import tqdm
def trainiter(encoder, decoder, dataloader, n,
              learning_rate = 0.001, epochs = 30):
    
    
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    
    accuracy = []
    for j in tqdm(range(epochs)):
        
        teacher_forcing = True if j/epochs < 0.5 else False
        epoch_loss = 0
        letter_accuracy = 0
        word_accuracy = 0
        for input_tensor, target_tensor in dataloader:
            
            
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            
            # Calculate loss per words and update parameters per word
          
            loss, predict_indicator, letter_indicator = train(input_tensor,
                                                              target_tensor, encoder,
                                                              decoder,
                                                              encoder_optimizer,
                                                              decoder_optimizer, criterion,
                                                              teacher_forcing)
           

            word_accuracy += predict_indicator
            epoch_loss += loss
            letter_accuracy += letter_indicator
            
            
            # if i % print_every == 0:
            #     print_loss_avg = print_loss_total/ print_every
            #     print_loss_total = 0
            #     print(f'Loss at {i}: {print_loss_avg}')
        
      
        
        epoch_loss /= 52000
        print(f'Epoch {j}====Word Accuracy: {word_accuracy}, Loss:  {epoch_loss} Letter Accuracy: {letter_accuracy}')
        
        accuracy.append(word_accuracy)
        
    
    print('Accuracy:', accuracy)
                

    
hidden_layer_size = 128
batch_size = 20
num_layers = 4
    
encoder = EncoderRNN(english.n_letters, hidden_layer_size,num_layers,
                     batch_size, verbose = False).to(device)
decoder = DecoderRNN(hidden_layer_size, hindi.n_letters,num_layers, 
                     batch_size, verbose = False).to(device)


n = df1.shape[0]

trainiter(encoder, decoder, data_loader, n)
