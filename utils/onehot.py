import numpy as np
import re
import string
import torch


class OnehotEncoding():
    """Class to transform the sentences in One Hot vectors
    """
    def __init__(self):
        """OnehotEncoding constructor
        """
        self.dico_len, self.index = self.__get_index()
        self.onehot_sentences, self.sent_len = self.__get_onehot_training_text()
     
    def __get_onehot_training_text(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        texts = open('../data/sentences.txt').readlines()
        sentences = [re.sub('\n', '', text) for text in texts]

        # Process sentences
        max_len = np.max([len(line) for line in sentences])
        synsthesis_sentences = torch.zeros((len(sentences), max_len, self.dico_len)).to(device)
        #synsthesis_sent_bool = torch.zeros((len(sentences), max_len)).to(device)
        for idx in range(len(sentences)):
            synsthesis_sentences[idx][0:len(sentences[idx])] = self.sent_emb(sentences[idx])
            #synsthesis_sent_bool[idx][0:len(sentences[idx])] = 1
   
        #return synsthesis_sentences, synsthesis_sent_bool
        return synsthesis_sentences, len(synsthesis_sentences[0])

    def sent_emb(self, sent):
        """Create One Hot vectors of a sentence
        
        Arguments:
            sent {string} -- sentence
        
        Returns:
            Tensor -- corresponding One Hot vectors
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sent_tens = torch.zeros((len(sent), self.dico_len)).to(device)
        for (i, let) in enumerate(sent):
            sent_tens[i] = self.letter_emb(let)
        return sent_tens

    def letter_emb(self, letter):
        """Create One Hot vectors of a character
        
        Arguments:
            letter {string} -- a character
        
        Returns:
            Tensor -- corresponding One Hot vector
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
        let_tens = torch.zeros((1, self.dico_len)).to(device)
        if letter in self.index.keys():
            let_tens[0, self.index[letter]] = 1
        else:
            let_tens[0, self.index['!']] = 1
        return let_tens

    def __get_index(self):
        """Create the index of the embedding
        
        Returns:
            n_letters {int} -- nomber of character of the dictionnary used for the embedding
            index {dict} -- dictionnary, key is a letter, value is the index of the letter in the One Hot encoding
        """
        all_letters = string.ascii_letters + " ?.,;'\"-!:"
        # for i in range(10):
        #     all_letters += str(i) 
        
        n_letters = len(all_letters)
        index = {all_letters[i]:i for i in range(n_letters)}
        return n_letters, index
