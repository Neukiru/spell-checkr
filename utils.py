import re
import io
import os
import regex as re
from itertools import islice
import numpy as np

from keras.models import Model, load_model
from keras.layers import Input

from model import truncated_acc, truncated_loss

np.random.seed(1234)

SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.

CHARS = list('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწხჰ- ')
whitelist = set('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწხჰ- ')


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
          chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char2index = dict((c, i) for i, c in enumerate(self.chars))
        self.index2char = dict((i, c) for i, c in enumerate(self.chars))
        self.size = len(self.chars)
    
    def encode(self, C, nb_rows):
        """One-hot encode given string C.
        # Arguments
          C: string, to be encoded.
          nb_rows: Number of rows in the returned one-hot encoding. This is
          used to keep the # of rows for each data the same via padding.
        """
        x = np.zeros((nb_rows, len(self.chars)), dtype=np.float32)
        for i, c in enumerate(C):
            x[i, self.char2index[c]] = 1.0
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
          x: A vector or 2D array of probabilities or one-hot encodings,
          or a vector of character indices (used with `calc_argmax=False`).
          calc_argmax: Whether to find the character index with maximum
          probability, defaults to `True`.
        """
        if calc_argmax:
            indices = x.argmax(axis=-1)
        else:
            indices = x
        chars = ''.join(self.index2char[ind] for ind in indices)
        return indices, chars

    def sample_multinomial(self, preds, temperature=1.0):
        """Sample index and character output from `preds`,
        an array of softmax probabilities with shape (1, 1, nb_chars).
        """
        # Reshaped to 1D array of shape (nb_chars,).
        preds = np.reshape(preds, len(self.chars)).astype(np.float64)
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        index = np.argmax(probs)
        char  = self.index2char[index]
        return index, char


def read_text(data_path, list_of_books):
    text = ''
    for book in list_of_books:
        file_path = os.path.join(data_path, book)
        strings = io.open(file_path,mode="r", encoding="utf-8").read()
        text += strings + ' '
    return text


def tokenize(text):
    tokens = re.sub('[^\P{P}-]+', ' ', text)
    tokens = re.sub('\s+',' ', tokens)
    tokens = [''.join(filter(whitelist.__contains__, token))
              for token in re.split("[-\n ]", tokens)]      
    
    return tokens

def add_segmentation(tokens,error_rate,n_gramms = 2):
    np.random.seed(1234)
    corrected_tokens = []
    corrupted_tokens = []
    for index, obj in enumerate(tokens):
        rand = np.random.rand()
        if rand <= error_rate:
            if  index < len(tokens) - (n_gramms - 1) :
                corrected_tokens.append(" ".join(tokens[index:index + n_gramms]))
                corrupted_tokens.append("".join(tokens[index:index + n_gramms]))                        
            else:
                pass       
        else:
            if index < len(tokens) - (n_gramms - 1) :
              corrected_tokens.append(" ".join(tokens[index:index + n_gramms]))
              corrupted_tokens.append(" ".join(tokens[index:index + n_gramms]))
            else:
              pass        
                   
            
    return corrected_tokens, corrupted_tokens

def add_segmentation_errors(tokens,error_rate, n_gramms = 2):
    prohibited = False
    np.random.seed(1234)
    corrected_tokens = []
    corrupted_tokens = []
    for index, obj in enumerate(tokens):
        rand = np.random.rand()
        if rand <= error_rate and not prohibited:
            if  index < len(tokens) - (n_gramms - 1) :
                corrected_tokens.append(" ".join(tokens[index:index + n_gramms]))
                corrupted_tokens.append("".join(tokens[index:index + n_gramms]))
                prohibited = not prohibited                        
            else:
                corrected_tokens.append(tokens[index])
                corrupted_tokens.append(tokens[index])       
        else:
            if not prohibited :
               if index < len(tokens) - (n_gramms - 1) :
                   corrected_tokens.append(" ".join(tokens[index:index + n_gramms]))
                   corrupted_tokens.append(" ".join(tokens[index:index + n_gramms]))
               else:
                   corrected_tokens.append(tokens[index])
                   corrupted_tokens.append(tokens[index]) 
            prohibited = not prohibited                            
    return corrected_tokens, corrupted_tokens
    
def add_speling_erors(token, error_rate):
    """Simulate some artificial spelling mistakes."""
    assert(0.0 <= error_rate < 1.0)
    if len(token) < 3:
        return token
    rand = np.random.rand()
    # Here are 4 different ways spelling mistakes can occur,
    # each of which has equal chance.
    prob = error_rate / 4.0
    if rand < prob:
        # Replace a character with a random character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + np.random.choice(CHARS) \
                + token[random_char_index + 1:]
    elif prob < rand < prob * 2:
        # Delete a character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + token[random_char_index + 1:]
    elif prob * 2 < rand < prob * 3:
        # Add a random character.
        random_char_index = np.random.randint(len(token))
        token = token[:random_char_index] + np.random.choice(CHARS) \
                + token[random_char_index:]
    elif prob * 3 < rand < prob * 4:
        # Transpose 2 characters.
        random_char_index = np.random.randint(len(token) - 1)
        token = token[:random_char_index]  + token[random_char_index + 1] \
                + token[random_char_index] + token[random_char_index + 2:]
    else:
        # No spelling errors.
        pass
    return token



def transform(tokens,corrupted_tokens, maxlen, error_rate=0.3, shuffle=True):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    if shuffle:
        print('Shuffling data.')
        shuffle_tokens = list(zip(tokens, corrupted_tokens))
        np.random.shuffle(shuffle_tokens)
        tokens,corrupted_tokens = [list(pack) for pack in zip(*shuffle_tokens)]
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    for corrupted_token,token in zip(corrupted_tokens,tokens):
        encoder = add_speling_erors(corrupted_token, error_rate=error_rate)
        encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
        encoder_tokens.append(encoder)
        decoder = SOS + token
        decoder += EOS * (maxlen - len(decoder))
        decoder_tokens.append(decoder)
    
        target = decoder[1:]
        target += EOS * (maxlen - len(target))
        target_tokens.append(target)  
        assert(len(encoder) == len(decoder) == len(target))
    return encoder_tokens, decoder_tokens, target_tokens

def batch(tokens, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(tokens, reverse):
        while(True): # This flag yields an infinite generator.
            for token in tokens:
                if reverse:
                    token = token[:-1][::-1]
                yield token[:-1]
    
    token_iterator = generate(tokens, reverse)
    data_batch = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = ctable.encode(token, maxlen)
        yield data_batch

def get_nth_line(file,n):
  file.seek(0,0)
  line_offset = []
  offset = 0
  for line in file:
      line_offset.append(offset)
      offset += len(line)
  file.seek(0,0)

  
  file.read(line_offset[n])
  string = file.readline()
  return string[:-1]


def batch_from_file(token_stream, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    
    def generate(token_stream, reverse):
        while(True):
            token_stream.seek(0,0) # This flag yields an infinite generator.
            for token in token_stream:
                if reverse:
                    token = token[:-1][::-1]
                else:
                    token = token[:-1]
                yield token
    
    token_iterator = generate(token_stream, reverse)
    data_batch = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = ctable.encode(token, maxlen)
        yield data_batch

def preprocess_in_chuncks(data_path,list_of_books,num_lines,train_val_flag = 0,segmentation_rate = 0.5, n_grammes = 3):
  if train_val_flag == 0:
        train_val_flag = 'train'
  else:
        train_val_flag = 'val'
  data_count = 0
  maxlen = 0
  for book in list_of_books:
        file_path = os.path.join(data_path, book)    
        with open(file_path, mode = 'r', encoding= 'utf-8') as f:
            
            tokenized_file = open('{}_tokenized_file.txt'.format(train_val_flag),'a')
            corr_tokenized_file = open('{}_corr_tokenized_file.txt'.format(train_val_flag),'a')
            while True:
              next_n_lines = "".join(line for line in list(islice(f, num_lines)))
              if not next_n_lines:
                  break
              tokenized = tokenize(next_n_lines)
              tokenized = list(filter(None, tokenized))
              tokenized,corr_tokenized = add_segmentation(tokenized,segmentation_rate,n_grammes)
              tmp_len = max([len(token) for token in tokenized]) + 2
              if maxlen < tmp_len:
                maxlen = tmp_len
              tokenized_map = map(lambda x:x+'\n',tokenized)
              corr_tokenized_map = map(lambda y:y+'\n',corr_tokenized)
              data_count += len(set(tokenized))
              corr_tokenized_file.writelines(list(corr_tokenized_map))
              tokenized_file.writelines(list(tokenized_map))
                    
              
                        
  print('preprocessing complete',)        
  return '{}_tokenized_file.txt'.format(train_val_flag),'{}_corr_tokenized_file.txt'.format(train_val_flag), maxlen, data_count        

def transform_in_chunks(tokenized_file,corr_tokenized_file,chunk_size,maxlen,train_val_flag = 0,error_rate = 0.5,shuffle = False):
    if train_val_flag == 0:
        train_val_flag = 'train'
    else:
        train_val_flag = 'val'
    tokenized_stream = open(tokenized_file,'r')
    corr_tokenized_stream = open(corr_tokenized_file,'r')
    encoder_file = os.path.exists(os.path.join(os.getcwd(), '{}_encoder.txt'.format(train_val_flag)))
    if encoder_file:
        open('{}_encoder.txt'.format(train_val_flag), 'w').close()
        open('{}_decoder.txt'.format(train_val_flag), 'w').close()
        open('{}_target.txt'.format(train_val_flag), 'w').close()
        encoder_tokens = open('{}_encoder.txt'.format(train_val_flag),'a')
        decoder_tokens = open('{}_decoder.txt'.format(train_val_flag),'a')
        target_tokens = open('{}_target.txt'.format(train_val_flag),'a')
    else:
        encoder_tokens = open('{}_encoder.txt'.format(train_val_flag),'a+')
        decoder_tokens = open('{}_decoder.txt'.format(train_val_flag),'a+')
        target_tokens = open('{}_target.txt'.format(train_val_flag),'a+')
    eof = False
    while not eof:
      tokens = []
      corrupted_tokens = []
      counter = 0
      while counter < chunk_size:
        counter += 1
        next_token = tokenized_stream.readline()[:-1]
        next_corr_token = corr_tokenized_stream.readline()[:-1]
        if not next_token:
          eof = True
          break
        tokens.append(next_token)
        corrupted_tokens.append(next_corr_token)
      if shuffle:
          
          shuffle_tokens = list(zip(tokens, corrupted_tokens))
          np.random.shuffle(shuffle_tokens)
          tokens,corrupted_tokens = [list(pack) for pack in zip(*shuffle_tokens)]
      for corrupted_token,token in zip(corrupted_tokens,tokens):
          encoder = add_speling_erors( corrupted_token, error_rate)
          encoder += EOS * (maxlen - len(encoder)) + '\n' # Padded to maxlen.
          encoder_tokens.writelines(encoder)
          decoder = SOS + token
          decoder += EOS * (maxlen - len(decoder))+'\n'
          decoder_tokens.writelines(decoder)
          target = decoder[1:]
          target += EOS * (maxlen - len(target))
          target_tokens.writelines(target)  
          assert(len(encoder[:-1]) == len(decoder[:-1]) == len(target))
    return '{}_encoder.txt'.format(train_val_flag),'{}_decoder.txt'.format(train_val_flag),'{}_target.txt'.format(train_val_flag)


def datagen(encoder_iter, decoder_iter, target_iter):
    """Utility function to load data into required model format."""
    inputs = zip(encoder_iter, decoder_iter)
    while(True):
        encoder_input, decoder_input = next(inputs)
        target = next(target_iter)
        yield ([encoder_input, decoder_input], target)


def decode_sequences(input_length,inputs, targets, input_ctable, target_ctable,
                     maxlen, reverse, encoder_model, decoder_model,
                     nb_examples, sample_mode='argmax', random=True):
    input_tokens = []
    target_tokens = []
    
    if random:
        indices = np.random.randint(0, input_length, nb_examples)
    else:
        indices = range(nb_examples)
        
    for index in indices:
        input_tokens.append(get_nth_line(inputs,index))
        target_tokens.append(get_nth_line(targets,index))
    
    input_sequences = batch(input_tokens, maxlen, input_ctable,
                            nb_examples, reverse)
    input_sequences = next(input_sequences)
    
    # Procedure for inference mode (sampling):
    # 1) Encode input and retrieve initial decoder state.
    # 2) Run one step of decoder with this initial state
    #    and a start-of-sequence character as target.
    #    Output will be the next target character.
    # 3) Repeat with the current target character and current states.

    # Encode the input as state vectors.    
    enc_output, enc_h, enc_c = encoder_model.predict(input_sequences)
    states_value = [enc_h,enc_c]
    # Create batch of empty target sequences of length 1 character.
    target_sequences = np.zeros((nb_examples, 1, target_ctable.size))
    # Populate the first element of target sequence
    # with the start-of-sequence character.
    target_sequences[:, 0, target_ctable.char2index[SOS]] = 1.0

    # Sampling loop for a batch of sequences.
    # Exit condition: either hit max character limit
    # or encounter end-of-sequence character.
    decoded_tokens = [''] * nb_examples
    for _ in range(maxlen):
        # `char_probs` has shape
        # (nb_examples, 1, nb_target_chars)
        char_probs, h, c = decoder_model.predict(
            [enc_output,target_sequences] + states_value)

        # Reset the target sequences.
        target_sequences = np.zeros((nb_examples, 1, target_ctable.size))

        # Sample next character using argmax or multinomial mode.
        sampled_chars = []
        for i in range(nb_examples):
            if sample_mode == 'argmax':
                next_index, next_char = target_ctable.decode(
                    char_probs[i], calc_argmax=True)
            elif sample_mode == 'multinomial':
                next_index, next_char = target_ctable.sample_multinomial(
                    char_probs[i], temperature=0.5)
            else:
                raise Exception(
                    "`sample_mode` accepts `argmax` or `multinomial`.")
            decoded_tokens[i] += next_char
            sampled_chars.append(next_char) 
            # Update target sequence with index of next character.
            target_sequences[i, 0, next_index] = 1.0

        stop_char = set(sampled_chars)
        if len(stop_char) == 1 and stop_char.pop() == EOS:
            break
            
        # Update states.
        states_value = [h, c]
    
    # Sampling finished.
    input_tokens   = [re.sub('[%s]' % EOS, '', token)
                      for token in input_tokens]
    target_tokens  = [re.sub('[%s]' % EOS, '', token)
                      for token in target_tokens]
    decoded_tokens = [re.sub('[%s]' % EOS, '', token)
                      for token in decoded_tokens]
    return input_tokens, target_tokens, decoded_tokens


def restore_model(path_to_full_model, hidden_size):
    """Restore model to construct the encoder and decoder."""
    model = load_model(path_to_full_model, custom_objects={
        'truncated_acc': truncated_acc, 'truncated_loss': truncated_loss})
    
    encoder_inputs = model.input[0] # encoder_data
    encoder_lstm1 = model.get_layer('encoder_lstm_1')
    encoder_lstm2 = model.get_layer('encoder_lstm_2')
    
    encoder_outputs = encoder_lstm1(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs] + encoder_states)

    decoder_inputs = model.input[1] # decoder_data
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    decoder_enc_output = Input(shape=(None,hidden_size))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    dot_layer1 = model.get_layer('dot_layer1')
    attention = dot_layer1([decoder_outputs, decoder_enc_output])
    activation_layer = model.get_layer('activation_layer')
    attention = activation_layer(attention)
    dot_layer2 = model.get_layer('dot_layer2')
    context = dot_layer2([attention, decoder_enc_output])
    conc_layer = model.get_layer('conc_layer')
    decoder_outputs = conc_layer([context, decoder_outputs])
    decoder_softmax = model.get_layer('decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_outputs)
    decoder_model = Model(inputs=[decoder_enc_output,decoder_inputs] + decoder_states_inputs,
                          outputs=[decoder_outputs] + decoder_states)
    return encoder_model, decoder_model