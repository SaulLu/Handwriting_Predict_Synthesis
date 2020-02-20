import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from models.networks import LstmHandPredictor, LstmHandSynthesis
from models.dataloaders import HandwrittingData, HandwrittingDataSynthesis
from models.dummy import generate_conditionally
from utils import plot_stroke_list


class HandPredictor:
    """
    Class to train the unconditional network
    """

    def __init__(self, hidden_size, batch_size=100, num_gauss=20):
        """HandPredictor constructor
        
        Arguments:
            hidden_size {int} -- number of hidden cells for one hidden layer
        
        Keyword Arguments:
            batch_size {int} -- number of batch used for the training (default: {100})
            num_gauss {int} -- number of bivariate Gaussian modelised (default: {20})
        
        Attributes:
            model -- instance of the model
            batch_size
        """
        self.batch_size = batch_size

        self.model = LstmHandPredictor(hidden_size, batch_size, num_gauss)
        self.__optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.__hidden_size = hidden_size
        self.__num_gauss = num_gauss

        self.__cuda = torch.cuda.is_available()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.__load_data()

        if self.__cuda:
            self.model.to(self.__device)

        print(f"Cuda is available : {self.__cuda}")

    def __load_data(self):
        train_data = HandwrittingData()
        self.__train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=False
        )

    def train(self, n_epoch):
        """Model training
        
        Arguments:
            n_epoch {int} -- number of epoch
        """
        total_loss = []
        for epoch in range(n_epoch):
            self.model.reset()
            for idx_batch, (data_batch, bool_batch) in enumerate(self.__train_loader):

                data_batch = Variable(data_batch, requires_grad=False).to(self.__device)
                bool_batch = Variable(bool_batch, requires_grad=False).to(self.__device)

                outputs = self.model(data_batch)

                loss = self.__compute_log_likelihood(
                    outputs, data_batch, bool_batch
                ).float()

                if idx_batch % 10 == 0:
                    print(f"Epoch :{epoch} / {n_epoch} | Current loss:{loss}")

                total_loss.append(loss)

                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()

            self.model.reset()

        torch.save(
            self.model.state_dict(),
            "../models/files/tests/mod_predict_"
            + time.strftime("%Y%m%d-%H%M%S")
            + "_hidden_"
            + str(self.__hidden_size)
            + ".pt",
        )

    def __bivariate_gaussian(self, sig1, sig2, mu1, mu2, x1, x2, rho):
        """Calculation of value(s) of a given bivariate Gaussian for given input(s)
        
        Arguments:
            sig1 {Tensor} -- 1st standard deviations
            sig2 {Tensor} -- 2nd standard deviations
            mu1 {Tensor} -- 1st mean
            mu2 {Tensor} -- 2nd mean
            x1 {Tensor} -- relative deviation of the next point along x
            x2 {Tensor} -- relative deviation of the next point along y
            rho {Tensor} -- correlation
       
        Returns:
            Tensor -- value(s) of a given bivariate Gaussian
        """
        Z1 = ((x1 - mu1) / sig1) ** 2
        Z2 = ((x2 - mu2) / sig2) ** 2
        Z3 = 2 * rho * (x1 - mu1) * (x2 - mu2) / (sig1 * sig2)
        Z = Z1 + Z2 - Z3

        pi_const = torch.tensor([np.pi]).to(self.__device)

        return torch.exp(-Z / (2 * (1 - rho ** 2))).to(self.__device) / (
            2 * pi_const * sig1 * sig2 * torch.sqrt(1 - rho ** 2)
        )

    def __compute_log_likelihood(self, outputs, data, boolean):
        """Calculation of the sequence loss
        
        Arguments:
            outputs {list[Tensor]} -- list containing the tensors of the parameters of the n_gauss bivariate gaussians distributions
            data {Tensor} -- strokes modified so that all sentences have the same length
            boolean {Tensor} -- Tensor which indicates whether the corresponding value in the tensor stroke is an original value or a 0 added for length uniformity
        
        Returns:
            Tensor -- Tensor containing the loss value
        """
        end_loc, pi_loc, mu_loc, sigma_loc, rho_loc = outputs

        mu1_loc, mu2_loc = mu_loc[:, :, :20], mu_loc[:, :, 20:]
        sig1_loc, sig2_loc = (
            sigma_loc[:, :, :20] + 10e-10,
            sigma_loc[:, :, 20:] + 10e-10,
        )

        x1_loc = data[:, 1:, 1].unsqueeze(2).to(self.__device)
        x2_loc = data[:, 1:, 2].unsqueeze(2).to(self.__device)
        x3_loc = data[:, 1:, 0].to(self.__device)

        end_loc = end_loc[:, :-1, -1].to(self.__device)
        pi_loc = pi_loc[:, :-1, :].to(self.__device)
        mu1_loc = mu1_loc[:, :-1, :].to(self.__device)
        mu2_loc = mu2_loc[:, :-1, :].to(self.__device)
        sig1_loc = sig1_loc[:, :-1, :].to(self.__device)
        sig2_loc = sig2_loc[:, :-1, :].to(self.__device)
        rho_loc = rho_loc[:, :-1, :].to(self.__device)

        boolean = boolean[:, :-1].to(self.__device)

        gauss = pi_loc * self.__bivariate_gaussian(
            sig1_loc, sig2_loc, mu1_loc, mu2_loc, x1_loc, x2_loc, rho_loc
        )
        gauss = torch.sum(gauss, dim=2).to(self.__device)

        log_lik = torch.sum(
            -boolean * torch.log(gauss + 10e-10)
            - boolean * torch.log(end_loc + 10e-10) * (x3_loc)
            - boolean * torch.log(1 - end_loc + 10e-10) * (1 - x3_loc)
        )

        return log_lik


class HandSynthesis:
    """Class to train the conditional network
    """

    def __init__(
        self, hidden_size, onehot_encoder, batch_size=100, num_gauss_multi=20, num_gauss_conv=10, path_previous_model=None
    ):
        """HandSynthesis constructor
        
        Arguments:
            hidden_size {int} -- number of hidden cells for one hidden layer
            onehot_encoder {OnehotEncoding} -- instance of OnehotEncoding
        
        Keyword Arguments:
            batch_size {int} -- number of batch used for the training (default: {100})
            num_gauss_multi {int} -- number of bivariate gaussian functions used to predict x_1 and x_2 (default: {20})
            num_gauss_conv {int} -- number of gaussian functions for the window (default: {10})
            path_previous_model {str} -- path to the model to continue training by default start from scratch (default: {None})
        """
        self.batch_size = batch_size
        self.__load_data(onehot_encoder) #del 
        self.model = LstmHandSynthesis(
            hidden_size,
            batch_size,
            num_gauss_multi,
            num_gauss_conv,
            self.dico_len,
            self.sent_len,
            self.stroke_len,
        )
        self.__optimizer = optim.Adam(self.model.parameters(), lr=8e-4, weight_decay=1e-8)

        self.__hidden_size = hidden_size
        self.__num_gauss_conv = num_gauss_conv
        self.__num_gauss_multi = num_gauss_multi

        self.__cuda = torch.cuda.is_available()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Cuda is available : {self.__cuda}")

        if self.__cuda:
            self.model.to(self.__device)
        if path_previous_model:
            self.resume_training(path_previous_model)

    def __load_data(self, onehot_encoder):
        train_data = HandwrittingDataSynthesis(onehot_encoder)
        self.dico_len = train_data.dico_len
        self.sent_len = train_data.sent_len
        self.stroke_len = train_data.stroke_len
        self.__train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
    
    def resume_training(self, path_to_model):
        checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, n_epoch, text_to_plot = 'welcome to lyrebird'):
        """Model training
        
        Arguments:
            n_epoch {int} -- number of epoch
            text_to_plot {string} -- text used to test the model during training
        """
        total_loss = []
        for epoch in range(n_epoch):
            for idx_batch, (data_batch, bool_batch, sentence_batch) in enumerate(
                self.__train_loader
            ):

                data_batch = Variable(data_batch, requires_grad=False).to(self.__device)
                bool_batch = Variable(bool_batch, requires_grad=False).to(self.__device)
                sentence_batch = Variable(sentence_batch, requires_grad=False).to(
                    self.__device
                )

                outputs = self.model(data_batch, sentence_batch)

                loss = self.__compute_log_likelihood(
                    outputs, data_batch, bool_batch
                ).float()/torch.sum(bool_batch).float()

                # print(f"torch.sum(bool_batch).float() : {torch.sum(bool_batch).float()}")

                if idx_batch % 10 == 0:
                    print(f"Epoch :{epoch} / {n_epoch} | Current loss:{loss}")
                    # if idx_batch % 100 == 0:
                    #     phi = self.model.attention_var['phi'][:,:,:,0]
                    #     plot_phi(phi[0,:,:])

                total_loss.append(loss)

                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()

            if epoch % 5 == 0 or epoch == 0:
                path_file = "../models/files/temp/mod_synthesis_" \
                          + time.strftime("%Y%m%d_%H%M%S") \
                          + "_hidden_" \
                          + str(self.__hidden_size) \
                          + ".pt"

                torch.save(self.model.state_dict(), path_file)

                stroke_list = generate_conditionally(text=text_to_plot, random_seed=1, model_path = path_file, bias = 1., plot_attention=True)

                path_img = "../models/files/img/mod_synthesis_" \
                          + time.strftime("%Y%m%d_%H%M%S") \
                          + "_hidden_" \
                          + str(self.__hidden_size) \
                          + ".jpeg"
 
                plot_stroke_list(stroke_list, path_img)

        torch.save(
            self.model.state_dict(),
            "../models/files/tests/mod_synthesis_"
            + time.strftime("%Y%m%d_%H%M%S")
            + "_hidden_"
            + str(self.__hidden_size)
            + ".pt",
        )

    def __bivariate_gaussian(self, sig1, sig2, mu1, mu2, x1, x2, rho):
        """Calculation of value(s) of a given bivariate Gaussian for given input(s)
        
        Arguments:
            sig1 {Tensor} -- 1st standard deviations
            sig2 {Tensor} -- 2nd standard deviations
            mu1 {Tensor} -- 1st mean
            mu2 {Tensor} -- 2nd mean
            x1 {Tensor} -- relative deviation of the next point along x
            x2 {Tensor} -- relative deviation of the next point along y
            rho {Tensor} -- correlation
       
        Returns:
            Tensor -- value(s) of a given bivariate Gaussian
        """
        Z1 = ((x1 - mu1) / sig1) ** 2
        Z2 = ((x2 - mu2) / sig2) ** 2
        Z3 = 2 * rho * (x1 - mu1) * (x2 - mu2) / (sig1 * sig2)

        Z = Z1 + Z2 - Z3

        pi_const = torch.tensor([np.pi]).to(self.__device)

        return torch.exp(-Z / (2 * (1 - rho ** 2))).to(self.__device) / (
            2 * pi_const * sig1 * sig2 * torch.sqrt(1 - rho ** 2)
        )

    def __compute_log_likelihood(self, outputs, data, boolean):
        """Calculation of the sequence loss
        
        Arguments:
            outputs {list[Tensor]} -- list containing the tensors of the parameters of the n_gauss bivariate gaussians distributions
            data {Tensor} -- strokes modified so that all sentences have the same length
            boolean {Tensor} -- Tensor which indicates whether the corresponding value in the tensor stroke is an original value or a 0 added for length uniformity
        
        Returns:
            Tensor -- Tensor containing the loss value
        """
        end_loc, pi_loc, mu_loc, sigma_loc, rho_loc = outputs

        mu1_loc, mu2_loc = mu_loc[:, :, :20], mu_loc[:, :, 20:]
        sig1_loc, sig2_loc = (
            sigma_loc[:, :, :20] + 10e-10,
            sigma_loc[:, :, 20:] + 10e-10,
        )

        x1_loc = data[:, 1:, 1].unsqueeze(2).to(self.__device)
        x2_loc = data[:, 1:, 2].unsqueeze(2).to(self.__device)
        x3_loc = data[:, 1:, 0].to(self.__device)

        end_loc = end_loc[:, :-1, -1].to(self.__device)
        pi_loc = pi_loc[:, :-1, :].to(self.__device)
        mu1_loc = mu1_loc[:, :-1, :].to(self.__device)
        mu2_loc = mu2_loc[:, :-1, :].to(self.__device)
        sig1_loc = sig1_loc[:, :-1, :].to(self.__device)
        sig2_loc = sig2_loc[:, :-1, :].to(self.__device)
        rho_loc = rho_loc[:, :-1, :].to(self.__device)

        boolean = boolean[:, :-1].to(self.__device)

        gauss = pi_loc * self.__bivariate_gaussian(
            sig1_loc, sig2_loc, mu1_loc, mu2_loc, x1_loc, x2_loc, rho_loc
        )
        gauss = torch.sum(gauss, dim=2).to(self.__device)

        log_lik = torch.sum(
            -boolean * torch.log(gauss + 10e-10)
            - boolean * torch.log(end_loc + 10e-10) * (x3_loc)
            - boolean * torch.log(1 - end_loc + 10e-10) * (1 - x3_loc)
        )

        return log_lik