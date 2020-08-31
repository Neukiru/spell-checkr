import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import CharacterTable, transform
from utils import restore_model, decode_sequences
from utils import read_text, tokenize

error_rate = 0.6
reverse = True
model_path = 'drive/My Drive/my_model.h5'
hidden_size = 512
sample_mode = 'argmax'
data_path = 'deep-spell-checkr/data'
books = ['nietzsche.txt', 'pride_and_prejudice.txt', 'shakespeare.txt', 'war_and_peace.txt']

test_sentence = "The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."


if __name__ == '__main__':
    text  = read_text(data_path, books)
    vocab = tokenize(text)
    vocab = list(filter(None, vocab))
    vocab,corr_vocab = utils.add_segmentation(vocab,0.15)

   
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    maxlen = max([len(token) for token in vocab]) + 2
    train_encoder, train_decoder, train_target = transform(
    vocab,corr_vocab,maxlen, error_rate=error_rate, shuffle=False)

    tokens = tokenize(test_sentence)
    tokens = list(filter(None, tokens))
    tokens,corr_tokens = utils.add_segmentation_errors(tokens,0.2)

    
    misspelled_tokens, _, target_tokens = transform(
    tokens,corr_tokens,maxlen, error_rate=error_rate, shuffle=False)
    nb_tokens = len(tokens)
    
    

    input_chars = set(' '.join(train_encoder))
    target_chars = set(' '.join(train_decoder))
    input_ctable = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)
    
    encoder_model, decoder_model = restore_model(model_path, hidden_size)
    
    input_tokens, target_tokens, decoded_tokens = decode_sequences(
        misspelled_tokens, target_tokens, input_ctable, target_ctable,
        maxlen, reverse, encoder_model, decoder_model, nb_tokens,
        sample_mode=sample_mode, random=False)
    
    print('-')
    print('Input sentence:  ', ' '.join([token for token in input_tokens]))
    print('-')
    print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
    print('-')
    print('Target sentence: ', ' '.join([token for token in target_tokens]))
