import torch
import numpy as np
from torch.utils.data import Dataset

from utils.onehot import OnehotEncoding


class HandwrittingData(Dataset):
    """
    Dataset customised for the unconditional model
    """

    def __init__(self):
        """HandwrittingData constructor
        
        Attributes:
            strokes {Tensor} -- Complete database of strokes recorded to train the model modified so that all sentences have the same length
            bools {Tensor} -- Tensor which indicates whether the corresponding value in the tensor stroke is an original value or a 0 added for length uniformity
        """
        strokes = np.load("../data/strokes-py3.npy", allow_pickle=True)

        max_len = np.max([len(s) for s in strokes])
        predict_strokes = np.zeros((len(strokes), max_len, 3))
        predict_bool = np.zeros((len(strokes), max_len))
        for idx in range(len(strokes)):
            predict_strokes[idx][0 : len(strokes[idx])] = strokes[idx]
            predict_bool[idx][0 : len(strokes[idx])] = 1

        self.__cuda = torch.cuda.is_available()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.strokes = (
            torch.from_numpy(predict_strokes).type(torch.FloatTensor).to(self.__device)
        )
        self.bools = (
            torch.from_numpy(predict_bool).type(torch.FloatTensor).to(self.__device)
        )
        self.len = len(predict_strokes)

    def __getitem__(self, index):
        """to support the indexing such that dataset[i] can be used to get ith sample
        
        Arguments:
            index {int}
        
        Returns:
            strokes[index] -- ith sample from strokes
            bools[index] -- ith sample from bools
        """
        return self.strokes[index], self.bools[index]

    def __len__(self):
        """Number of elements in the dataset
        
        Returns:
            len -- number of elements in the dataset
        """
        return self.len


class HandwrittingDataSynthesis(Dataset):
    """
    Dataset customised for the conditional model
    """

    def __init__(self, onehot_encoder):

        """HandwrittingDataSynthesis(Dataset) constructor
        
        Arguments:
            onehot_encoder {OnehotEncoding} -- instance of OnehotEncoding

        Attributes:
            strokes {Tensor} -- Complete database of strokes recorded to train the model modified so that all sentences have the same length
            bools {Tensor} -- Tensor which indicates whether the corresponding value in the tensor stroke is an original value or a 0 added for length uniformity
            sentences {Tensor} -- Sentences corresponding to the strokes and encoded in onehot

            index -- index used for the encoding of the sentences
            dico_len -- number of characters used in the index
            sent_len -- standardised lenght of sentences
            stroke_len -- standardised lenght of strokes
            len -- number of elements in the dataset
        """
        self.__cuda = torch.cuda.is_available()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #onehot_encoder = OnehotEncoding()
        synsthesis_sentences_full = onehot_encoder.onehot_sentences

        self.index = onehot_encoder.index
        self.dico_len = onehot_encoder.dico_len
        self.sent_len = onehot_encoder.sent_len

        strokes_full = np.load("../data/strokes-py3.npy", allow_pickle=True)

        assert len(strokes_full) == len(synsthesis_sentences_full)

        strokes = []
        synsthesis_sentences = []

        for ind in range(len(strokes_full)):
            if len(strokes_full[ind])<800:
                strokes.append(strokes_full[ind])
                synsthesis_sentences.append(synsthesis_sentences_full[ind])
                #print(sentences[ind])

        assert len(strokes) == len(synsthesis_sentences)

        # print(f"len dataset: {len(strokes)}")

        # Process strokes to standardize lengths
        max_len = np.max([len(s) for s in strokes])

        # print(f"max_len: {max_len}")
        synsthesis_strokes = np.zeros((len(strokes), max_len, 3))
        synsthesis_bool = np.zeros((len(strokes), max_len))
        for idx in range(len(strokes)):
            synsthesis_strokes[idx][0 : len(strokes[idx])] = strokes[idx]
            synsthesis_bool[idx][0 : len(strokes[idx])] = 1

        self.strokes = (
            torch.from_numpy(synsthesis_strokes)
            .type(torch.FloatTensor)
            .to(self.__device)
        )
        self.bools = (
            torch.from_numpy(synsthesis_bool).type(torch.FloatTensor).to(self.__device)
        )

        self.sentences = synsthesis_sentences
        self.len = len(synsthesis_strokes)
        self.stroke_len = len(synsthesis_strokes[0])
        print(f"embedding done")

    def __getitem__(self, index):
        """to support the indexing such that dataset[i] can be used to get ith sample
        
        Arguments:
            index {int}
        
        Returns:
            strokes[index] -- ith sample from strokes
            bools[index] -- ith sample from bools
            sentences[index] -- ith sample from sentence
        """
        return self.strokes[index], self.bools[index], self.sentences[index]

    def __len__(self):
        """Number of elements in the dataset
        
        Returns:
            len -- number of elements in the dataset
        """
        return self.len
