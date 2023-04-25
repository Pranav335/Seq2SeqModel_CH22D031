# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:25:40 2023

@author: Pranav Agrawal

Seq2seq model 
"""

import pandas as pd

import torch

import numpy as np

import torch.nn as nn


df = pd.read_csv('H:/My Drive/Courses/Jan-May 2023/Deep_Learning/Assignment 3/aksharantar_sampled/aksharantar_sampled/hin/hin_train.csv')
df.columns = ['English', 'Hindi']


SOS_token = 0
EOS_token = 1

class Lang:

  def __init__(self, name):
    self.name = name
    self.letter2index = {}
    self.letter2count = {}
    self.index2letter = {}
    self.index2letter = { 0: "SOS", 1: "EOS"}
    self.n_letters = 2


  def addWord(self, word):
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

english = lettertonumber(df, 'English')
hindi = lettertonumber(df, 'Hindi')

def word2number(word, language_dict):
  word = [*word]
  rep = [language_dict.letter2index[letter] for letter in word]
  rep.append(EOS_token)
  rep = torch.tensor(rep, dtype = torch.long)
  return rep


def dataset(df, lang1 = english, lang2 = hindi):

  df["English Representation"] = df["English"].apply(lambda x: word2number(x, lang1))
  df["Hindi Representation"] = df["Hindi"].apply(lambda x: word2number(x, lang2))

  return df


df1 = dataset(df)


class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size, verbose = False):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)
    self.verbose = verbose

  def forward(self, input, hidden):
    if self.verbose == True:
        print(f'Input Shape: {input.shape}')
    embedded = self.embedding(input.view(-1, 1))
    # The view(-1, 1) reshapes the input to size (n, 1)
    output = embedded
    if self.verbose == True:
        print(f'Embedding Shape: {embedded.shape}')
    output, hidden = self.gru(output, hidden)
    if self.verbose == True:
        print('Shape of Input:', input.shape)
        print('Shape of embedded:', embedded.shape)
        print('Shape of output:', output.shape)
        print('Shape of hidden:', hidden.shape)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size, verbose = False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.verbose = verbose
        
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        if self.verbose == True:
            print('Shape of Input:', input.shape)
            print('Shape of output:', output.shape)
            print('Shape of hidden:', hidden.shape)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
    
# encoder = EncoderRNN(english.n_letters, 120, verbose = True)
# hidden = encoder.initHidden()
# #encoder(df1["English Representation"][1], hidden)
# embedding = nn.Embedding(hindi.n_letters, 120)
# decoder_input = torch.tensor([[SOS_token]])
# output = embedding(decoder_input)
# enc_output, hidden = encoder(df1["English Representation"][1], hidden)
# gru = nn.GRU(120, 120)
# dec_out, hidden = gru(output, hidden)
# linear = nn.Linear(120, hindi.n_letters)
# dec_out = linear(dec_out)
# softmax = nn.LogSoftmax(dim = 1)
# softmax(dec_out)
# topv, topi = dec_out.topk(1)
# decoder_input = topi.squeeze().detach()

import torch.optim as optim

def train(input_tensor, target_tensor, encoder, decoder,
             encoder_optimizer, decoder_optimizer, criterion):
    
    encoder_hidden = encoder.initHidden()
    
    loss = 0
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #input_length = input_tensor.size(0) # L x 1 x features
    target_length = target_tensor.size(0) # L x 1 x features
    
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    decoder_input = torch.tensor([SOS_token])
    
    decoder_hidden = encoder_hidden
    
    for i in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        # print('Decoder output shape',decoder_output.shape)
        # print('target_tensor shape', target_tensor[i].view(1).shape)
        # print('target_tensor', target_tensor[i])
        loss += criterion(decoder_output, target_tensor[i].view(1))
        if decoder_input.item() == EOS_token:
            break
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()/target_length

import matplotlib.pyplot as plt

def trainiter(encoder, decoder, n_iters, print_every = 1000, plot_every = 100,
              learning_rate = 0.01):
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    for i in range(n_iters):
        input_tensor, target_tensor = df1.iloc[i, 2:].values
        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss
        
        if i % print_every == 0:
            print_loss_avg = print_loss_total/ print_every
            print_loss_total = 0
            print(f'Loss at {i}: {print_loss_avg}')
            
        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    plt.plot(plot_losses)
    
    
    
encoder = EncoderRNN(english.n_letters, 16)
decoder = DecoderRNN(16, hindi.n_letters)

n = df1.shape[0]

trainiter(encoder, decoder, n)