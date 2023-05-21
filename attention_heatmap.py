# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:56:05 2023

@author: Pranav Agrawal
"""

''' This code import the trained encoder and attention model and plots the 
attention heatmap for a sample input of size 20'''

from attention_with_batching_vers2 import  EncoderRNN, Attention

from attention_with_batching_vers2 import english,hindi, device

import pdb
import torch

print('Main script')
encoder = EncoderRNN(input_size = english.n_letters,
                     embedding_size = 300,
                     hidden_size = 512,
                     num_layers = 3,
                     batch_size = 20,
                     cell_type = 'LSTM',
                     bidirectional = False,
                     dropout = 0)

decoder = Attention(embedding_size = 300,
                    hidden_size = 512,
                    output_size = hindi.n_letters,
                    num_layers = 3,
                    batch_size = 20,
                    encoder = encoder,
                    cell_type = 'LSTM',
                    bidirectional = False,
                    dropout = 0).to(device)

encoder.load_state_dict(torch.load('encoder_attention.pt', map_location = torch.device('cpu')))
decoder.load_state_dict(torch.load('decoder_attention.pt', map_location = torch.device('cpu')))

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

def number2word(predicted_output, target_tensor, lang1 = hindi, lang2 = english, remove_pad_token = False):
    
    ''' This function takes the input tensor and target tensor and convert them 
    into word representations.
    
    Output: List of tuple of list 
    
    '''
    predicted = []
    for row, row1 in zip(predicted_output, target_tensor):
        if remove_pad_token:
            predicted_hindi_word = [lang2.index2letter[elements.item()] for elements in row if elements not in [0,1,2,3] ]
            true_word = [lang1.index2letter[elements.item()] for elements in row1 if elements not in [0,1,2,3]]
            predicted.append((predicted_hindi_word, true_word))
        else:
            predicted_hindi_word = [lang2.index2letter[elements.item()] for elements in row ]
            true_word = [lang1.index2letter[elements.item()] for elements in row1]
            predicted.append((predicted_hindi_word, true_word))


    return predicted


from batching_with_accuracy import  test_dataloader

# Output the attention weights

def prediction(input_tensor, target_tensor, encoder, decoder):
    
    input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

    encoder_hidden = encoder.initHidden()

    if encoder.cell_type == 'LSTM':
        co = encoder_hidden
        encoder_hidden = (encoder_hidden, co)

    with torch.no_grad():
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([SOS_token] * decoder.batch_size, device = device)

        if decoder.cell_type == 'LSTM':
            hn, cn = encoder_hidden
            hn = hn[-1].repeat(decoder.num_layers,1,1)
            cn = cn[-1].repeat(decoder.num_layers,1,1)
            decoder_hidden = (hn, cn)
        else:
            decoder_hidden = encoder_hidden[-1].repeat(decoder.num_layers,1,1)

        predicted_output = torch.zeros(target_tensor.shape[0],
                target_tensor.shape[1], dtype = torch.long,
                device = device)

        target_length = target_tensor.size(1)
        att_wts = {}
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input,encoder_output, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            predicted_output[:, i: i+1] = topi
            decoder_input = topi
            att_wts[f'{i}'] = decoder.att_wts
            
            
        return att_wts
    

def inference(encoder, decoder, dataloader):
    
    with open('prediction_attention.txt', 'w', encoding = 'utf-8') as f:
        f.write('Predicted word, True Word')
        f.write('\n')

        for input_tensor, target_tensor in dataloader:
    
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
    
            encoder_hidden = encoder.initHidden()
    
            if encoder.cell_type == 'LSTM':
                co = encoder_hidden
                encoder_hidden = (encoder_hidden, co)
    
            with torch.no_grad():
                encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
                decoder_input = torch.tensor([SOS_token] * decoder.batch_size, device = device)
    
                if decoder.cell_type == 'LSTM':
                    hn, cn = encoder_hidden
                    hn = hn[-1].repeat(decoder.num_layers,1,1)
                    cn = cn[-1].repeat(decoder.num_layers,1,1)
                    decoder_hidden = (hn, cn)
                else:
                    decoder_hidden = encoder_hidden[-1].repeat(decoder.num_layers,1,1)
    
                predicted_output = torch.zeros(target_tensor.shape[0],
                        target_tensor.shape[1], dtype = torch.long,
                        device = device)
    
                target_length = target_tensor.size(1)
    
                for i in range(target_length):
                    decoder_output, decoder_hidden = decoder(decoder_input,encoder_output, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    predicted_output[:, i: i+1] = topi
                    decoder_input = topi
    
    
                lines = number2word(predicted_output, target_tensor, lang2 = hindi, remove_pad_token = True)
                
                for line in lines:
                    
                    word_1 = ''.join(line[0])
                    word_2 = ''.join(line[1])
                    word = word_1  + ',' + word_2
                    f.write(word)
                    f.write('\n')
                    
inference(encoder, decoder, test_dataloader)


## Generate attention heatmap on one batch of test data.
     
i, t = next(iter(test_dataloader))

import matplotlib.pyplot as plt




import numpy as np

word_representation = number2word(i,t)

att_wts = prediction(i, t, encoder, decoder)

fig = plt.figure(figsize = (40, 40))

# Plotting attention heatmap

from matplotlib.font_manager import FontProperties
font_prop = FontProperties(fname='arial.ttf', size=30)
for i in range(9):
    # i is batch and j sequence length
    english_word = word_representation[i][0]
    hindi_word = word_representation[i][1]
    att_mat = np.zeros((hindi.max_size+ 1, english.max_size + 1))
    
    for j in range(hindi.max_size + 1):
        att_mat[j, :] = att_wts[f'{j}'][i]
        
    ax = fig.add_subplot(3,3, i+1)
    im = ax.imshow(att_mat)
    
    ax.set_xticks(np.arange(len(english_word)))
    ax.set_xticklabels(english_word, fontsize = 25)

    ax.set_yticks(np.arange(len(hindi_word)))
    ax.set_yticklabels(hindi_word, fontsize = 25, fontproperties = font_prop)
    ax.set_title(''.join(english_word), fontdict = {'fontsize': 30, 'fontweight':'medium'})
    
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    # fig.tight_layout()
    # plt.show()     
fig.subplots_adjust(wspace=0.2, hspace=0.2)
fig.suptitle('Attention Heatmap',fontsize = 50)
        
    
        
        
    

            
            