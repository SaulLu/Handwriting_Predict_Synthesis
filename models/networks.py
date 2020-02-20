import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class LstmHandPredictor(nn.Module):
    """Class for the unconditional network
    """

    def __init__(self, hidden_size, batch_size=100, num_gauss=20):
        """LstmHandPredictor constructor
        
        Arguments:
            hidden_size {int} -- number of hidden cells for one hidden layer
        
        Keyword Arguments:
            batch_size {int} -- number of batch used for the training (default: {100})
            num_gauss {int} -- number of bivariate Gaussian modelised (default: {20})
        
        Attributes:
            lstm1 -- first layer of LSTM
            lstm2 -- Second layer of LSTM
            fc -- Fully connected layer
            hidden_size
            batch_size
            num_gauss
        """
        super(LstmHandPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(
            input_size=3 + hidden_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=1 + num_gauss * 6)

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_gauss = num_gauss

        self.__cuda = torch.cuda.is_available()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        init = Variable(
            torch.zeros((1, self.batch_size, self.hidden_size), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)
        self.previous = {1: (init, init), 2: (init, init)}

    def reset(self):
        """Reset to 0 the tensors storing the values of the lstm outputs in the previous sequence.
        """
        init = Variable(
            torch.zeros((1, self.batch_size, self.hidden_size), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)
        self.previous = {1: (init, init), 2: (init, init)}

    def forward(self, stroke):

        lstm1_output, (hid1, c1) = self.lstm1(stroke, self.previous[1])
        self.previous[1] = hid1.detach(), c1.detach()

        lstm2_input = torch.cat([stroke, lstm1_output], dim=-1)
        lstm2_output, (hid2, c2) = self.lstm2(lstm2_input, self.previous[2])
        self.previous[2] = hid2.detach(), c2.detach()

        fc_input = torch.cat([lstm1_output, lstm2_output], dim=-1)
        fc_output = self.fc(fc_input)
        param_output = self.__get_output(fc_output)

        return param_output

    def __get_output(self, fc_output):
        """Splitting of the output into Gaussian parameters simulating x_1 and x_2
        
        Arguments:
            fc_output {Tensor} -- Output of the last layer, the fully connected layer, of the network
        
        Returns:
            tensor storing parameters of the n_gauss multivariate gaussian distributions
            end -- end of stroke probability
            pi -- mixture weights
            mu -- two means
            sigma -- two standard deviations
            rho -- correlation
        """
        n = self.num_gauss
        cutting = [1, n, 2 * n, 2 * n, n]
        end, pi, mu, sigma, rho = torch.split(fc_output, cutting, dim=2)

        end = 1 / (1 + torch.exp(end))
        pi = torch.softmax(pi, dim=-1)
        sigma = torch.exp(sigma)
        rho = torch.tanh(rho)
        mu = mu

        return end, pi, mu, sigma, rho


class LstmHandSynthesis(nn.Module):
    """Class for the conditional network
    """

    def __init__(
        self,
        hidden_size,
        batch_size=100,
        num_gauss_multi=20,
        num_gauss_conv=10,
        dico_len=72,
        sent_len=64,
        stroke_len=1191,
        usage='train'
    ):
        """LstmHandSynthesis constructor

        Arguments:
            hidden_size {int} -- number of hidden cells for one hidden layer
        
        Keyword Arguments:
            batch_size {int} -- number of batch used for the training (default: {100})
            num_gauss_multi {int} -- number of bivariate gaussian functions used to predict x_1 and x_2 (default: {20})
            num_gauss_conv {int} -- number of gaussian functions for the window (default: {10})
            dico_len {int} -- number of characters in the lexicon used for one hot encoding (default: {72})
            sent_len {int} -- standardized length of encoded sentences (default: {64})
            stroke_len {int} -- standardized length of strokes (default: {1191})
            usage {str} -- 'train' or 'generate'
        """
        super(LstmHandSynthesis, self).__init__()
        self.lstm1_cell = nn.LSTMCell(input_size=3 + dico_len, hidden_size=hidden_size)
        self.lstm2 = nn.LSTM(
            input_size=3 + hidden_size + dico_len,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.lstm3 = nn.LSTM(
            input_size=3 + hidden_size + dico_len,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.window_layer = WindowLayer(
            hidden_size=hidden_size, num_gauss_conv=num_gauss_conv, batch_size=batch_size
        )

        self.fc = nn.Linear(
            in_features=hidden_size * 3, out_features=1 + num_gauss_multi * 6
        )

        self.dico_len = dico_len
        self.sent_len = sent_len
        self.stroke_len = stroke_len
        self.num_gauss_multi = num_gauss_multi
        self.num_gauss_conv = num_gauss_conv

        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.__cuda = torch.cuda.is_available()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__usage = usage

        self.reset()

        init_cell = Variable(
            torch.zeros((self.batch_size, self.hidden_size), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)

        init_net = Variable(
            torch.zeros((1, self.batch_size, self.hidden_size), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)

        self.previous = {
            1: (init_cell, init_cell),
            2: (init_net, init_net),
            3: (init_net, init_net),
        }

        self.attention_var ={
            'wind':[],
            'kappa': [],
            'phi':[]
            }

    def reset(self):
        """Reset to 0 the tensors storing the values of the window and kappa in the previous sequence.
        """
        self.window_layer.reset()
        self.wind_prev = Variable(
            torch.zeros((self.batch_size, self.dico_len), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)

        init_cell = Variable(
            torch.zeros((self.batch_size, self.hidden_size), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)

        init_net = Variable(
            torch.zeros((1, self.batch_size, self.hidden_size), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)

        self.previous = {
            1: (init_cell, init_cell),
            2: (init_net, init_net),
            3: (init_net, init_net),
        }


    def forward(self, stroke, sentence_hot, bias = 0.):
        lstm1_output = []
        windows_outputs = []
        wind_list = []
        kappa_list = []
        phi_list = []

        for time in range(stroke.size()[1]):

            try:
                lstm1_cell_input = torch.cat([stroke[:, time, :], self.wind_prev], dim=-1)
            except:
                print(f"stroke[:, time, :]: {stroke[:, time, :].size()}")
                print(f"self.wind_prev: {self.wind_prev.size()}")
            hid1, c1 = self.lstm1_cell(lstm1_cell_input, self.previous[1])
            self.previous[1] = hid1.detach(), c1.detach()
            lstm1_output.append(self.previous[1][0])

            window_input = self.previous[1][0]
            wind_prev, kappa, phi = self.window_layer(window_input, sentence_hot)
            self.wind_prev = wind_prev.detach()
            windows_outputs.append(wind_prev.detach())

            wind_list.append(wind_prev.detach().cpu().numpy())
            kappa_list.append(kappa.detach().cpu().numpy())
            phi_list.append(phi.detach().cpu().numpy())

            # if time % 100 == 0:
            #     print(f"time: {time}")
            #     print(f"kappa: {self.window_layer.kappa_prev.size()}")

        windows_outputs = torch.stack(windows_outputs, dim=1)
        lstm1_output = torch.stack(lstm1_output, dim=1)

        lstm2_input = torch.cat([stroke, lstm1_output, windows_outputs], dim=-1)
        lstm2_output, (hid2, c2) = self.lstm2(lstm2_input, self.previous[2])
        self.previous[2] = hid2.detach(), c2.detach()

        lstm3_input = torch.cat([stroke, lstm2_output, windows_outputs], dim=-1)
        lstm3_output, (hid3, c3) = self.lstm3(lstm3_input, self.previous[3])
        self.previous[3] = hid3.detach(), c3.detach()

        fc_input = torch.cat([lstm1_output, lstm2_output, lstm3_output], dim=-1)
        fc_output = self.fc(fc_input)
        param_output = self.__get_output(fc_output, bias)

        self.attention_var['wind'] = np.swapaxes(np.array(wind_list), 0, 1)
        self.attention_var['kappa'] = np.swapaxes(np.array(kappa_list), 0, 1)
        self.attention_var['phi'] = np.swapaxes(np.array(phi_list), 0, 1)

        if self.__usage == 'train':
          self.reset()

        return param_output

    def __get_output(self, fc_output, bias=0.):
        """Splitting of the output into Gaussian parameters simulating x_1 and x_2
        
        Arguments:
            fc_output {Tensor} -- Output of the last layer, the fully connected layer, of the network
        
        Keyword Arguments:
            bias {float} -- value of the bias used to calculate the coefficients (default: {0.})

        Returns:
            tensor storing parameters of the n_gauss multivariate gaussian distributions
            end -- end of stroke probability
            pi -- mixture weights
            mu -- two means
            sigma -- two standard deviations
            rho -- correlation
        """
        n = self.num_gauss_multi
        cutting_lay_2 = [1, n, 2 * n, 2 * n, n]

        end, pi, mu, sigma, rho = torch.split(fc_output, cutting_lay_2, dim=2)

        end = 1 / (1 + torch.exp(end))
        pi = torch.softmax(pi*(1+bias), dim=-1)
        sigma = torch.exp(sigma- bias)
        rho = torch.tanh(rho)
        mu = mu

        return end, pi, mu, sigma, rho


class WindowLayer(nn.Module):
    """Class for the window layer
    """

    def __init__(self, hidden_size, num_gauss_conv, batch_size):
        """WindowLayer constructor
        
        Arguments:
            hidden_size {int} -- number of hidden cells for one hidden layer
            num_gauss_conv {int} -- number of gaussian functions for the window
            batch_size {int} -- number of item in a batch

        Attributes:
            linear -- linear layer
            num_gauss_conv -- number of gaussian functions for the window
            kappa_prev -- value of kappa at the previous step
        """
        super(WindowLayer, self).__init__()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear = nn.Linear(
            in_features=hidden_size, out_features=3 * num_gauss_conv
        )
        self.num_gauss_conv = num_gauss_conv
        self.batch_size = batch_size
        self.kappa_prev = Variable(
            torch.zeros((self.batch_size, self.num_gauss_conv), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)

        
    def reset(self):
        self.kappa_prev = Variable(
            torch.zeros((self.batch_size, self.num_gauss_conv), dtype=torch.float),
            requires_grad=False,
        ).to(self.__device)

    def forward(self, window_input, sentence_hot):
        linear_output = self.linear(window_input)

        alpha, beta, kappa = self.__get_inter_output(linear_output)

        wind, phi = self.__compute_conv_wind(alpha, beta, kappa, sentence_hot)
        return wind, kappa, phi

    def __get_inter_output(self, linear_output):
        """Splitting of the output of the linear layer into gaussian parameters simulating the window
        
        Arguments:
            linear_output {Tensor} -- output of the linear layer
        
        Returns:
            alpha
            beta
            kappa
        """

        n = self.num_gauss_conv
        cutting_lay_1 = [n, n, n]

        alpha, beta, kappa = torch.split(linear_output, cutting_lay_1, dim=1)

        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        kappa = self.kappa_prev + torch.exp(kappa) * 0.05

        self.kappa_prev = kappa.detach()
        return alpha, beta, kappa

    def __compute_conv_wind(self, alpha, beta, kappa, sentence_hot):
        """Calculate the window values
        
        Arguments:
            alpha {Tensor} -- parameter controlling the the importance of the window within the mixture
            beta {Tensor} -- parameter controlling the width of the window
            kappa {Tensor} -- parameter controlling the location of the window
            sentence_hot {Tensor} -- sentences embeded in onehot vectors
        
        Returns:
            wind -- tensor of the window values
        """
        u = (
            Variable(torch.arange(sentence_hot.size()[1]), requires_grad=False)
            .to(self.__device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        kappa = kappa.unsqueeze(2)
        alpha = alpha.unsqueeze(2)
        beta = beta.unsqueeze(2)

        phi_k = self.__discrete_conv(alpha, beta, kappa, u)

        phi = torch.sum(phi_k, dim=1).unsqueeze(2)

        wind = torch.sum(phi * sentence_hot, dim=1)

        return wind, phi

    def __discrete_conv(self, alpha, beta, kappa, u):
        return alpha * torch.exp(-beta * (kappa - u) ** 2)