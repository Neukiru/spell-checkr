import os
import numpy as np
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint


np.random.seed(1234)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import preprocess_in_chuncks,transform_in_chunks
from utils import CharacterTable
from utils import batch_from_file, datagen, decode_sequences
from model import seq2seq


hidden_size = 512
nb_epochs = 100
train_batch_size = 512
val_batch_size = 128
sample_mode = 'argmax'
# Input sequences may optionally be reversed,
# shown to increase performance by introducing
# shorter term dependencies between source and target:
# "Learning to Execute"
# http://arxiv.org/abs/1410.4615
# "Sequence to Sequence Learning with Neural Networks"
# https://arxiv.org/abs/1409.3215
reverse = True

data_path = './drive/My Drive/Data/'
train_books = ['train.txt']
val_books = ['val.txt']


if __name__ == '__main__':
    # Prepare training data.
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    tokenized_file,corr_tokenized_file,maxlen,train_count = preprocess_in_chuncks(data_path,train_books,500,0,0.15,n_grammes = 2)
    
    
    
    
    train_encoder, train_decoder, train_target = transform_in_chunks(tokenized_file,corr_tokenized_file,500,maxlen)
        
    

    
    
    
    print('Size of training vocabulary =', train_count)
    print('Max sequence length in the training set:', maxlen)

    # Prepare validation data.
    tokenized_file,corr_tokenized_file,maxlen,val_count = preprocess_in_chuncks(data_path,val_books,500,1,0.15,n_grammes = 2)
    
    
    print('val count {}'.format(val_count))
    
    val_encoder, val_decoder, val_target = transform_in_chunks(tokenized_file,corr_tokenized_file,500,maxlen,1)

    val_encoder_stream =  open(val_encoder,'r')
    val_decoder_stream =  open(val_decoder,'r')
    val_target_stream =  open(val_target,'r')
    # Define training and evaluation configuration.
    input_chars = set('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ-* ')
    target_chars = set('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ-* \t')
    nb_input_chars = len(input_chars)
    nb_target_chars = len(target_chars)
    input_ctable  = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)

    train_steps = train_count // train_batch_size
    val_steps = val_count // val_batch_size

    # Compile the model.
    model, encoder_model, decoder_model = seq2seq(
        hidden_size, nb_input_chars, nb_target_chars)
    print(model.summary())
    


    # Train and evaluate.
    for epoch in range(nb_epochs):
        print('Main Epoch {:d}/{:d}'.format(epoch + 1, nb_epochs))
    
        train_encoder, train_decoder, train_target = transform_in_chunks(
            tokenized_file,corr_tokenized_file,500, maxlen, shuffle=True)
        

        train_encoder_stream =  open(train_encoder,'r')
        train_decoder_stream =  open(train_decoder,'r')
        train_target_stream =  open(train_target,'r')        
        train_encoder_batch = batch_from_file(train_encoder_stream, maxlen, input_ctable,
                                    train_batch_size, reverse)
        train_decoder_batch = batch_from_file(train_decoder_stream, maxlen, target_ctable,
                                    train_batch_size)
        train_target_batch  = batch_from_file(train_target_stream, maxlen, target_ctable,
                                    train_batch_size)    

        val_encoder_batch = batch_from_file(val_encoder_stream, maxlen, input_ctable,
                                  val_batch_size, reverse)
        val_decoder_batch = batch_from_file(val_decoder_stream, maxlen, target_ctable,
                                  val_batch_size)
        val_target_batch  = batch_from_file(val_target_stream, maxlen, target_ctable,
                                  val_batch_size)
    
        train_loader = datagen(train_encoder_batch,
                               train_decoder_batch, train_target_batch)
        val_loader = datagen(val_encoder_batch,
                             val_decoder_batch, val_target_batch)
    
        
        my_callbacks = [
        EarlyStopping(patience=4, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(filepath = 'drive/My Drive/' + 'my_model.h5', 
        verbose=1, save_best_only=True, save_weights_only=False) 
        ]
    
        model.fit_generator(train_loader,
                            steps_per_epoch=train_steps,
                            epochs=1, verbose=1,
                            validation_data=val_loader,
                            validation_steps=val_steps,
                            callbacks = my_callbacks)
        
        # On epoch end - decode a batch of misspelled tokens from the
        # validation set to visualize speller performance.
        nb_tokens = 2
        
        input_tokens, target_tokens, decoded_tokens = decode_sequences(
             val_count,val_encoder_stream, val_target_stream, input_ctable, target_ctable,
             maxlen, reverse, encoder_model, decoder_model, nb_tokens,
             sample_mode=sample_mode, random=True)
        
        
        print('-')
        print('Input tokens:  ', input_tokens)
        print('Decoded tokens:', decoded_tokens)
        print('Target tokens: ', target_tokens)
        print('-')
        train_encoder_stream.close()
        train_decoder_stream.close()
        train_target_stream.close()