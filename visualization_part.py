from batching_with_accuracy import EncoderRNN, DecoderRNN
from batching_with_accuracy import english, hindi
import torch
import torch.nn

import pdb

print(english.n_letters)

encoder = EncoderRNN(input_size =english.n_letters,
        embedding_size =300,
        hidden_size =512,
        num_layers =3 ,
        batch_size = 20,
        cell_type ='LSTM' ,
        bidirectional =False ,
        dropout =0 )

decoder = DecoderRNN(embedding_size = 300,
        hidden_size =512, 
        output_size =hindi.n_letters,
        num_layers =3 ,
        batch_size =20 , 
        cell_type ='LSTM' ,
        bidirectional =False ,
        dropout = 0)

# IMport trained weights

encoder.load_state_dict(torch.load('encoder.pt', map_location = torch.device('cpu')))
decoder.load_state_dict(torch.load('decoder.pt', map_location = torch.device('cpu')))

# Model loaded

from batching_with_accuracy import  test_dataloader, device
from batching_with_accuracy import accuracy


encoder = encoder.to(device)
decoder = decoder.to(device)

print(accuracy(test_dataloader, encoder, decoder))



def number2word(predicted_output, target_tensor, lang = hindi):
    predicted = []
    for row, row1 in zip(predicted_output, target_tensor):
        predicted_hindi_word = [lang.index2letter[elements.item()] for elements in row if elements not in [0,1,2,3] ]
        true_word = [lang.index2letter[elements.item()] for elements in row1 if elements not in [0,1,2,3]]
        true_word = ''.join(true_word)
        predicted_hindi_word = ''.join(predicted_hindi_word)
        predicted.append((predicted_hindi_word, true_word))

    return predicted

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Write a funcntion to get the prediction from the decoder.
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

        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            predicted_output[:, i: i+1] = topi
            decoder_input = topi


        lines = number2word(predicted_output, target_tensor)

        return lines


# Writing the prediction to the files

 

def inference(encoder, decoder, dataloader):
    
    with open('prediction_vanilla.txt', 'w', encoding = 'utf-8') as f:
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
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    predicted_output[:, i: i+1] = topi
                    decoder_input = topi
    
    
                lines = number2word(predicted_output, target_tensor)
                
                for line in lines:
                    word_1 = line[0]
                    word_2 = line[1]
                    word = word_1  + ',' + word_2
                    f.write(word)
                    f.write('\n')
                            
                    
    
#inference(encoder, decoder, test_dataloader)

i,t = next(iter(test_dataloader))
input_english_word = []
for row in i:
    true_english_word = [english.index2letter[elements.item()] for elements in row if elements not in [0,1,2,3] ]
    input_english_word.append(''.join(true_english_word))
    
hindi_words = prediction(i,t, encoder, decoder)

from matplotlib.font_manager import FontProperties
font_prop = FontProperties(fname='arial.ttf', size = 50)

import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.text(0.3, 0.8, 'English Word: ' + input_english_word[0])
# ax.text(0.3, 0.5, 'True Hindi Word: ' + hindi_words[0][1], fontproperties = font_prop)
# ax.text(0.3, 0.2, 'Predicted Hindi Word: ' + hindi_words[0][0],fontproperties = font_prop)

fig = plt.figure(figsize = (40, 40))

for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    ax.text(0, 0.8, 'English Word: ' + input_english_word[i], fontproperties = font_prop)
    ax.text(0, 0.5, 'True Hindi Word: ' + hindi_words[i][1], fontproperties = font_prop)
    ax.text(0, 0.2, 'Predicted Hindi Word: ' + hindi_words[i][0],fontproperties = font_prop)
fig.subplots_adjust(wspace=0.2, hspace=0.2)
                        
                        
                        
                        
    
    
    
#i, t = next(iter(test_dataloader))
#print(prediction(i,t, encoder, decoder))
