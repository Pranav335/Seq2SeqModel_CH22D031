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

import argparse



# %%--------------------- Argparse -----------------------------------------

parser = argparse.ArgumentParser()

## Specify the hidden size 
parser.add_argument('-hs', '--hidden_size', default = 512, type = int,
                    help = 'Hidden size')

## Specify the batch size
parser.add_argument('-bs', '--batch_size', default = 20,
                    type = int, help = 'batch size')

## Specify the epochs
parser.add_argument('-e', '--epochs', default = 30, type = int, help = 'epoch')

## Specify the learning rate for the adam
parser.add_argument('-lr', '--learning_rate', default = 0.001, type = float,
                    help = "Specify the learning rate")

## Specify the embedding_size 
parser.add_argument('-es', '--embedding_size', default = 300, type = int,
                    help = 'Embedding size')

## Specify the number of encoder layers
parser.add_argument('-e_numl', '--encoder_layer', default = 3, type = int,
                    help = 'Number of encoder layer')

## Specify the number of decoder layers
parser.add_argument('-d_numl', '--decoder_layer', default = 3, type = int,
                    help = 'Number of decoder layer')

## Specify the cell type
parser.add_argument('-ct', '--cell_type', default = 'LSTM', type = str,
                    choices = ['RNN', 'LSTM', 'GRU'],
                    help = 'cell type')

## Specify the bidirectional
parser.add_argument('-b', '--bid', default = 0, type = int, 
                    help = 'Specify 1 for bidirectional')

## Specify the dropout
parser.add_argument('-d', '--dropout', default = 0, type = float,
                    help = 'Specify the dropout rate between 0 and 1')

## Specify whether to sweep or not
parser.add_argument('-s', '--sweep', default = 1,  type = int,
                    help = 'Specify 0 for running sweeps and 1 for running the model')

args = parser.parse_args()

# %%---------------------------- Device ---------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# %%---------------------------------------------------------------------------


# Reading English-Hindi data pairs from aksharantar dataset
df = pd.read_csv('./hin/hin_train.csv')
df.columns = ['English', 'Hindi']

df['length_Hindi_words'] = df['Hindi'].apply(lambda x: len([*x]))
df['length_English_words'] = df['English'].apply(lambda x: len([*x]))


df = df[(df['length_English_words'] < 15) & (df['length_Hindi_words'] < 14)]


# %%-------------------------Lang class----------------------------------------


PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class Lang:
  '''
  This class add letters from the language to make a dictionary of 
  letters in the language. Corresponding to each letter there is a associated 
  index. This index is unique for each letter in the dictionary. 
  '''
  def __init__(self, name):
      self.name = name
      self.letter2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
      self.letter2count = {}
      self.index2letter = {}
      self.index2letter = { 0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
      self.n_letters = 4
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

# %%--------------------- English and Hindi class -----------------------------

# English class 

english = lettertonumber(df, 'English')

# Hindi Class

hindi = lettertonumber(df, 'Hindi')

# %%---------------------- Validation dataset processing ----------------------


# Validation dataset processing

valdf = pd.read_csv('./hin/hin_valid.csv')
valdf.columns = ['English', 'Hindi']

valdf['length_Hindi_words'] = valdf['Hindi'].apply(lambda x: len([*x]))
valdf['length_English_words'] = valdf['English'].apply(lambda x: len([*x]))

valdf = valdf[(valdf['length_English_words'] < 15) & (valdf['length_Hindi_words'] < 14)]

testdf = pd.read_csv('./hin/hin_test.csv')
testdf.columns = ['English', 'Hindi']


testdf['length_Hindi_words'] = testdf['Hindi'].apply(lambda x: len([*x]))
testdf['length_English_words'] = testdf['English'].apply(lambda x: len([*x]))

testdf = testdf[(testdf['length_English_words'] < 15) & (testdf['length_Hindi_words'] < 14)]

# %%------------------------------ word2number --------------------------------


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
        try:
            rep[index] = language_dict.letter2index[letter]
        except:
            rep[index] = language_dict.letter2index["UNK"]
        
    rep[index + 1] = EOS_token
    return rep

# %%--------------------------dataset function---------------------------------



def dataset(df, lang1 = english, lang2 = hindi):

    df["English Representation"] = df["English"].apply(lambda x: word2number(x, lang1))
    df["Hindi Representation"] = df["Hindi"].apply(lambda x: word2number(x, lang2))
    
    return df



# %%-----------------------------Train and Validation Dataset------------------

# Train dataset 
df1 = dataset(df)

df1.reset_index(inplace = True)

# validation dataset

valdf = dataset(valdf)

valdf.reset_index(inplace = True)

# Test dataset


testdf = dataset(testdf)

testdf.reset_index(inplace = True)

# %%------------------ Custom Dataset -----------------------------------------

# Implementating custom dataset

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
    
# %%--------------------------- DataLoaders -----------------------------------


from torch.utils.data import DataLoader

# Training dataloader

batch_size = args.batch_size
dataset = CustomDataset(df1)
data_loader = DataLoader(dataset, batch_size = batch_size,
                         shuffle = True, drop_last = True)

# Validation dataloader

val_dataset = CustomDataset(valdf)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size,
                            shuffle = True, drop_last = True)
# Test dataloader

test_dataset = CustomDataset(testdf)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True,drop_last = True)
# %%------------------------------ Encoder Class ------------------------------


import torch.nn as nn

# Encoder Class

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 batch_size,
                 cell_type = 'GRU',
                 bidirectional = True,
                 dropout = 0.8,
                 verbose = False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
       
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.dropout = 0.8
        
        #input_size = Vocabulary size of the language
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        if num_layers == 1:
            dropout = 0
        
        self.cell_type = cell_type
        if cell_type == 'GRU':
            self.gru = nn.GRU(embedding_size, hidden_size, num_layers,
                              dropout = dropout, bidirectional = bool(bidirectional))
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers,
                              dropout = dropout, bidirectional =  bool(bidirectional))
        elif cell_type == 'LSTM':
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                                dropout = dropout, bidirectional =  bool(bidirectional))
        else:
            raise ValueError('Invalid cell type argument')           
            
        
        self.num_layers = num_layers
        self.verbose = verbose
    
    def forward(self, input_data, hidden):
        if self.verbose == True:
            print(f'Input Shape: {input_data.shape}')
        # Size of Input = (L, )
        # The view(-1, 1) reshapes the input to size (n, 1)
        
        embedded = self.dropout(self.embedding(input_data))
        # Size of embedded = (L x 1 x hidden_size)
        output = embedded.permute(1, 0, 2)
        # Size of embded is now change to (L x batch_size x hidden_size)
        
        
        if self.verbose == True:
            print(f'Embedding Shape: {embedded.shape}')
        
            
        if self.cell_type =='GRU':
            output, hidden = self.gru(output, hidden)
            # Size of output = (L x batch_size x hidden_size)
            # Size of hidden = (1 x batch_size x hidden_size)
            return output, hidden
        elif self.cell_type == 'RNN':
            output, hidden = self.rnn(output, hidden)
            return output, hidden
            
        elif self.cell_type == 'LSTM':

            ho, co = hidden
            output, hidden = self.lstm(output, (ho, co))
            if self.verbose == True:
                print('Shape of Input:', input_data.shape)
                print('Shape of embedded:', embedded.shape)
                print('Shape of output:', output.shape)
                print('Shape of hidden:', hidden.shape)
            return output, hidden
        
    
    def initHidden(self):
        return torch.zeros(self.D * self.num_layers, self.batch_size,
                           self.hidden_size, device = device)
    
# %%----------------------------- Decoder Class -------------------------------

# Decoder Class
    
class DecoderRNN(nn.Module):
    
    def __init__(self, embedding_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 batch_size,
                 cell_type = 'GRU',
                 bidirectional = True,
                 dropout = 0.8,
                 verbose = False):
        
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        self.relu = nn.ReLU()
        self.cell_type = cell_type
        
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
            
        if num_layers == 1:
            dropout = 0
        
        if cell_type == 'GRU':
            self.gru = nn.GRU(embedding_size, hidden_size, num_layers,
                              dropout = dropout, bidirectional = bool(bidirectional))
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers,
                              dropout = dropout, bidirectional = bool(bidirectional))
        elif cell_type == 'LSTM':
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                                dropout = dropout, bidirectional = bool(bidirectional))
        else:
            raise ValueError('Invalid cell type argument')
            
        self.out = nn.Linear(self.D * hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.num_layers = num_layers
        self.verbose = verbose
        self.batch_size = batch_size
        
    def forward(self, input_data, hidden):
        # Size of input = (1, 1) # One word at a time 
        #print('Size of input:', input.shape)
        
        embed = self.dropout(self.embedding(input_data))
        # Size of embed = (1 x 1 x hidden_size)
        #print('Size of embed: ', embed.shape)
        output = self.relu(embed).view(1, self.batch_size, -1)
        
            
        if self.cell_type =='GRU':
            output, hidden = self.gru(output, hidden)
            # Size of output = (L x batch_size x hidden_size)
            # Size of hidden = (1 x batch_size x hidden_size)
            output = self.softmax(self.out(output[0]))
            return output, hidden
        
        elif self.cell_type == 'RNN':
            output, hidden = self.rnn(output, hidden)
            output = self.softmax(self.out(output[0]))
            return output, hidden
            
        elif self.cell_type == 'LSTM':
            ho, co = hidden
            output, hidden = self.lstm(output, (ho, co))
            # size of output = (1 x 1 x hidden_size)
            output = self.softmax(self.out(output[0]))
            return output, hidden

    def initHidden(self):
        return torch.zeros(self.D * self.num_layers, self.batch_size,
                           self.hidden_size, device = device)
    
# %%------------------------------ number2word --------------------------------



import torch.optim as optim

def number2word(predicted_output, target_tensor, lang = hindi):
    
    mask = torch.all(predicted_output == target_tensor, dim = 1)
    correct_output = predicted_output[mask, :]
    true_output = target_tensor[mask, :]
    predicted = []
    for row, row1 in zip(correct_output, true_output):
        predicted_hindi_word = [lang.index2letter[elements.item()]
                                for elements in row if elements not in [0,1,2]]
        true_word = [lang.index2letter[elements.item()]
                                for elements in row1 if elements not in [0,1,2]]
        
        true_word = ''.join(true_word)
        predicted_hindi_word = ''.join(predicted_hindi_word)
        predicted.append((predicted_hindi_word, true_word))
        
    return predicted

# %%----------------------------- Train ---------------------------------------

def train(input_tensor, target_tensor, encoder, decoder,
              encoder_optimizer, decoder_optimizer, criterion,
              teacher_forcing = True, lang = hindi, verbose = False):
    
    encoder_hidden = encoder.initHidden()
    
    loss = 0
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #input_length = input_tensor.size(0) # L x 1 x features
    target_length = target_tensor.size(1) # L x 1 x features
    
    if encoder.cell_type == 'LSTM':
        co = encoder_hidden
        encoder_hidden = (encoder_hidden, co)
        
        
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    if decoder.cell_type == 'LSTM':
        hn,cn = encoder_hidden
        hn = hn[-1].repeat(decoder.num_layers,1, 1)
        cn = cn[-1].repeat(decoder.num_layers, 1, 1)
        decoder_hidden = (hn, cn)

    else:
        decoder_hidden = encoder_hidden[-1].repeat(decoder.num_layers, 1, 1)
    
    
    
    decoder_input = torch.tensor([SOS_token] * decoder.batch_size, device = device)
    
    
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
    
            
    loss.backward()
    
    word_accuracy = predicted_output == target_tensor
    word_accuracy = sum(torch.all(word_accuracy, dim = 1)).item()
    
    if verbose:
        prediction = number2word(predicted_output, target_tensor)
        if len(prediction) > 0:
            print(prediction)
    letter_accuracy = torch.sum(predicted_output == target_tensor)
    
    #print(predicted_output)
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item(), word_accuracy, letter_accuracy

# Write the eval function

# %%------------------------- batch_word_accuracy -----------------------------
    

def batch_word_accuracy(predicted_output, target_tensor):
    word_accuracy = predicted_output == target_tensor
    word_accuracy = sum(torch.all(word_accuracy, dim = 1)).item()
    letter_accuracy = torch.sum(predicted_output == target_tensor)
    return word_accuracy, letter_accuracy


import matplotlib.pyplot as plt

from tqdm import tqdm

# %%--------------------- trainiter -------------------------------------------


def trainiter(encoder, decoder, dataloader,
              learning_rate = 0.01, epochs = 10):
    
    
    
    encoder_optimizer = optim.NAdam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.NAdam(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    
    accuracy = []
    
    for j in range(epochs):
        
        teacher_forcing = True if j/epochs < 0.5 else False
        #teacher_forcing = False
        epoch_loss = 0
        letter_accuracy = 0
        word_accuracy = 0
        word_count = 0
        for input_tensor, target_tensor in dataloader:
            
            
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            
            # Calculate loss per words and update parameters per word
          
            loss, predict_indicator, letter_indicator = train(input_tensor,
                                                              target_tensor, encoder,
                                                              decoder,
                                                              encoder_optimizer,
                                                              decoder_optimizer, criterion,
                                                              teacher_forcing)
           
            word_count += decoder.batch_size
            word_accuracy += predict_indicator
            epoch_loss += loss
            letter_accuracy += letter_indicator
            
            
            # if i % print_every == 0:
            #     print_loss_avg = print_loss_total/ print_every
            #     print_loss_total = 0
            #     print(f'Loss at {i}: {print_loss_avg}')
        
      
        word_accuracy /= word_count
        epoch_loss /= word_count
        print(f'Epoch {j}====Word Accuracy: {word_accuracy}, Loss:  {epoch_loss} Letter Accuracy: {letter_accuracy}')
        
        accuracy.append(word_accuracy)

    return accuracy[-1]

# %%---------------------------------- Accuracy -------------------------------

def accuracy(dataloader, encoder, decoder):
    
    word_accuracy = 0
    
    word_count = 0 # Word Count
    
    for input_tensor, target_tensor in dataloader:
        
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
        
        word_count += decoder.batch_size
        
        encoder_hidden = encoder.initHidden()

        if encoder.cell_type == 'LSTM':
            co = encoder_hidden
            encoder_hidden = (encoder_hidden, co)
    
    

        with torch.no_grad():
            
            encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
            
            decoder_input = torch.tensor([SOS_token]* decoder.batch_size, device = device)
            
            if decoder.cell_type == 'LSTM':
                hn,cn, = encoder_hidden
                hn = hn[-1].repeat(decoder.num_layers, 1, 1)
                cn = cn[-1].repeat(decoder.num_layers, 1, 1)
                decoder_hidden = (hn, cn)
            else:
                decoder_hidden = encoder_hidden[-1].repeat(decoder.num_layers, 1, 1)
            
            predicted_output = torch.zeros(target_tensor.shape[0],
                                           target_tensor.shape[1], dtype = torch.long,
                                           device = device)
            
            target_length = target_tensor.size(1)
            
            for i in range(target_length):
                
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                predicted_output[:, i: i+1] = topi
                decoder_input = topi
                
                
            word_accuracy_batch = predicted_output == target_tensor
            word_accuracy_batch = sum(torch.all(word_accuracy_batch, dim = 1)).item()
            
            
        word_accuracy += word_accuracy_batch
        
        
    return word_accuracy/ word_count
                
        
        
if __name__ == "__main__":        
                                                
                    
    # %%-------------------------- Function Call ----------------------------------
        
    # hidden_layer_size = 128
    # batch_size = 20
    # num_layers = 4
        
    # encoder = EncoderRNN(english.n_letters, hidden_layer_size,num_layers,
    #                      batch_size, verbose = False).to(device)
    # decoder = DecoderRNN(hidden_layer_size, hindi.n_letters,num_layers, 
    #                      batch_size, verbose = False).to(device)


    # n = df1.shape[0]

    # trainiter(encoder, decoder, data_loader, n)



    input_size = english.n_letters
    hidden_size = args.hidden_size
    num_encoder_layer = args.encoder_layer
    num_decoder_layer = args.decoder_layer
    embedding_size = args.embedding_size
    output_size = hindi.n_letters
    cell_type = args.cell_type
    bid = args.bid
    dropout = args.dropout


    if args.sweep:
        
        encoder = EncoderRNN(input_size = input_size,
                              embedding_size = embedding_size,
                              hidden_size = hidden_size,
                              num_layers = num_encoder_layer,
                              batch_size = batch_size, 
                              cell_type = cell_type,
                              bidirectional = bid,
                              dropout = dropout).to(device)
        
        decoder = DecoderRNN(embedding_size = embedding_size,
                              hidden_size = hidden_size, 
                              output_size = output_size,
                              num_layers = num_decoder_layer,
                              batch_size = batch_size,
                              cell_type = cell_type,
                              bidirectional =False,
                              dropout = dropout).to(device)
        
        epoch = args.epochs
        
        trainiter(encoder = encoder, decoder = decoder, dataloader = data_loader,
                  learning_rate = args.learning_rate, epochs = epoch)
        
        print('Validation accuracy', accuracy(val_dataloader, encoder, decoder))
        print('test accuracy', accuracy(test_dataloader, encoder, decoder))

        torch.save(encoder.state_dict(), './encoder.pt')
        torch.save(decoder.state_dict(), './decoder.pt')
    else:

        import wandb
        
        sweep_configuration = {'method':'random',
                                'name':'Loss',
                                'metric': {'goal': 'maximize', 'name':'validation_accuracy'},
                                'parameters':
                                    {
                                      'hidden_size': {'values':[512]},
                                      'num_layer': {'values': [3]},
                                      'embedding_size':{'values': [300]},
                                      'cell_type': {'values': ['LSTM']},
                                      'bid': {'values': [False]},
                                      'dropout': {'values': [0]},
                                      'epoch': {'values': [20]},
                                      'lr': {'values': [0.001]}
                                      }}
            
            
        
        def wandbsweep():
            
            run = wandb.init(project = "Assignment-3",settings = wandb.Settings(console='off'), config = sweep_configuration )
            
            wandb.run.name = 'hidden_size_' + 'optmizer_nadam' + str(wandb.config.hidden_size) + 'num_layer_' + str(wandb.config.num_layer) + 'embedding_size_' + str(wandb.config.embedding_size) + 'cell_type_' + str(wandb.config.cell_type) + 'bid_' + str(wandb.config.bid) + 'lr_' + str(wandb.config.lr)  + 'epoch_' + str(wandb.config.epoch) + 'dropout_' + str(wandb.config.dropout)

            encoder = EncoderRNN(input_size = input_size,
                                  embedding_size = wandb.config.embedding_size,
                                  hidden_size = wandb.config.hidden_size,
                                  num_layers = wandb.config.num_layer,
                                  batch_size = batch_size, 
                                  cell_type = wandb.config.cell_type,
                                  bidirectional = wandb.config.bid,
                                  dropout = wandb.config.dropout).to(device)
        
            decoder = DecoderRNN(embedding_size = wandb.config.embedding_size,
                                  hidden_size = wandb.config.hidden_size, 
                                  output_size = output_size,
                                  num_layers = wandb.config.num_layer,
                                  batch_size = batch_size,
                                  cell_type = wandb.config.cell_type,
                                  bidirectional = False,
                                  dropout = wandb.config.dropout).to(device)
            
            training_loss = trainiter(encoder = encoder, decoder = decoder, dataloader = data_loader,
                      learning_rate = wandb.config.lr, epochs = wandb.config.epoch)
            
            
            validation_accuracy = accuracy(val_dataloader, encoder, decoder)
            
            print('test accuracy:', accuracy(test_dataloader, encoder, decoder))

            wandb.log({'validation_accuracy': validation_accuracy, 'training_loss': training_loss})
            wandb.finish()
            
            
        sweep_id = wandb.sweep(sweep = sweep_configuration, project = "Assignment-3")
        wandb.agent(sweep_id, function = wandbsweep, count = 30)

        


