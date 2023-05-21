# Seq2SeqModel_CH22D031

The English-Hindi data is used for all purposes.

# Vanilla Encoder - Decoder Model:


==============================================================

## batching_with_accuracy.py

This script implements the vanilla encoder-decoder model. The script can be run on the command line with users choice of hidden_size, embedding_size, encoder layer, decoder layer,batch size and many more. Kindly look into the help on the argparse of the scipt.

The script can also be used to run sweeps from commmand line.

Script contains following functionality:

1. Lang class - This class stores all the unique character in the language and contains dictionary for mapping letter to number and vice_versa.
self.index2letter - maps number to letter and self.letter2index - maps letter to number.
It also has attribute called max_size which stores the maximum length of the word for the dataset.

2. word2number - This function takes the words and convert into the tensor.

3. CustomDataset - This class gives the tensor for the given index. Will be used in generating dataloaders.

4. EncoderRNN - 

This class implements RNN, GRU, LSTM cells for encoding the input words. The function takes following arguments - 
input_size, embedding_size, hidden_size, num_layers, batch_size, cell_type, bidirectional, dropout.

5. DecoderRNN - 

This class decodes the encoded sequence. It takes following arguments - embedding_size, hidden_size, output_size, num_layers,
batch_size, cell_type, bidirectional, dropout.

6.number2word - This function takes the numeric output from the decoder and convert that output into word representation.
It takes Lang object as one of its input to convert the tensors into word. This function returns list of tuple of predicted 
word and true word.

7. train - Train the model for the given batch return the loss, total number of correct words and total number of correct letters

8. trainiter - This function calls the train function for each batch, update the word accuracy and loss.

9. accuracy - Model giives the acccuracy on the dataloader for the trained encoder and decoder.

# How to run the file:

batching_with_accuracy.py -hs 512 -e 20 -es 300 -ct 'LSTM' 

For sweeping:

batching_with_accuracy.py -s 0

This script will output the training accuracy, training loss, validation accuracy and test accuracy

===============================================================

# Attention Mechanism

## attention_with_batching_vers2.py

This script has the same functionality as the batching_with_accuracy.py except for the Attention class.

1. Attention - This class implements the RNN, GRU and LSTM with attention mechanism. The object.att_wts contains the attention scores.

The scipt can be run like the batching_with_accuracy.py with users choice of hidden layer, embeddding size and many more. 


==================================================================

### decoder.pt and encoder.pt are trained model from vanilla encoder-decoder models.
### encoder_attention.pt and decoder.attention.pt are trained model from attention driven encoder-decoder models.


## visualization_part.py

This  script loads the trained vanilla encoder-decoder model. This script writes the prediction on the test data set to the file 'prediction_vanilla.txt'
and also plot the grid for visualizing the output words from the decoder model.


## attention_heatmap.py

This code import the trained encoder and attention model and plots the attention heatmap for a sample input of size 20. The function prediction output the attention weights for the batch of input.
The function inference writes the prediction on the test data to the file 'prediction_attention.txt'. It also plots the attention heatmap on one batch of test data set.




