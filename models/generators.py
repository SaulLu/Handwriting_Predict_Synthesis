import numpy as np
import torch

from models.networks import LstmHandPredictor, LstmHandSynthesis
from utils.onehot import OnehotEncoding
from utils import plot_phi

def generate_unconditional(
    hidden_size, model_path, num_gauss=20, seed=1, timestamps=300
):
    """Generation of unconditional strokes without preconditions following the model that has been trained on the dataset strokes-py3
    
    Arguments:
        hidden_size {int} -- number of hidden cells for one hidden layer
        model_path {str} -- path to the model
    
    Keyword Arguments:
        num_gauss {int} -- number of bivariate Gaussian modelised(default: {20})
        seed {int} -- seed for random (default: {1})
        timestamps {int} -- temporal lenght of the stroke generated (default: {300})
    
    Returns:
        stroke -- Tensor with size (timestamps, 3)
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rs = np.random.RandomState(seed + 1)

    hand_writer = LstmHandPredictor(hidden_size, 1, num_gauss=num_gauss).to(device)
    hand_writer.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage)
    )
    hand_writer.eval()

    init = torch.zeros((1, 1, 3)).to(device)

    strokes = np.zeros((timestamps, 3))

    for time in range(timestamps):
        end, pi, mu, sigma, rho = hand_writer(init)

        lift = rs.binomial(1, end[:, :, -1].cpu().detach().numpy())

        strokes[time, 0] = lift

        mu1_loc, mu2_loc = mu[:, :, :20], mu[:, :, 20:].detach()
        sig1_loc, sig2_loc = sigma[:, :, :20].detach(), sigma[:, :, 20:].detach()

        gauss = np.zeros((num_gauss, 2))

        for j in range(num_gauss):

            s1 = sig1_loc[:, :, j].cpu().detach().numpy().squeeze(0)
            s2 = sig2_loc[:, :, j].cpu().detach().numpy().squeeze(0)
            ro = rho[:, :, j].cpu().detach().numpy().squeeze(0)
            mu1 = mu1_loc[:, :, j].cpu().detach().numpy().squeeze(0)
            mu2 = mu2_loc[:, :, j].cpu().detach().numpy().squeeze(0)

            cov = np.array([[s1 ** 2, s1 * s2 * ro], [s1 * s2 * ro, s2 ** 2]]).reshape(
                2, 2
            )

            mean = np.array([mu1, mu2]).reshape(2)

            rs.multivariate_normal(mean, cov)

            gauss[j] = pi[:, :, j].cpu().detach().numpy() * rs.multivariate_normal(
                mean, cov
            )

        strokes[time, 1:] = np.sum(gauss, axis=0)

        init = (
            torch.from_numpy(strokes[time])
            .type(torch.FloatTensor)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

    return strokes


def generate_conditional(
    hidden_size,
    model_path,
    num_gauss_multi=20,
    num_gauss_conv=10,
    seed=1,
    timestamps=300,
    sentence="hello word",
    bias=1.,
    plot_attention=False
):
    """generation of a stroke corresponding to a handwriting of the input text thanks to the model trained on the dataset strokes-py3

    Arguments:
        hidden_size {[type]} -- number of hidden cells for one hidden layer
        model_path {[type]} -- path to the model

    Keyword Arguments:
        num_gauss_multi {int} -- number of bivariate gaussian functions used to predict x_1 and x_2 (default: {20})
        num_gauss_conv {int} -- number of gaussian functions for the window  (default: {10})
        seed {int} -- seed for random  (default: {1})
        timestamps {int} -- temporal lenght of the stroke generated (default: {300})
        sentence {str} -- input sentence to transform (default: {'hello word'})
        bias {float} -- value of the bias used to calculate the coefficients (default: {1.})
        plot_attention {bool} -- plot the phis (default: {False})
    
    Returns:
        stroke -- Array with size (timestamps, 3)
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    onehot_encoder = OnehotEncoding()

    sent_tens = onehot_encoder.sent_emb(sentence).unsqueeze(0).to(device)
    rs = np.random.RandomState(seed)

    hand_writer = LstmHandSynthesis(
        hidden_size,
        1,
        num_gauss_conv=num_gauss_conv,
        num_gauss_multi=num_gauss_multi,
        dico_len=onehot_encoder.dico_len,
        sent_len=onehot_encoder.sent_len,
        usage='generate'
    ).to(device)
    hand_writer.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage)
    )
    hand_writer.eval()

    init = torch.zeros((1, 1, 3)).to(device)

    phis = []
    keep_going = True
    time = 0
    first = True
    begin_end = False

    strokes = np.zeros((timestamps, 3))

    strokes = np.array([])

    while keep_going:
        new_stroke = np.zeros((1, 3))
        end, pi, mu, sigma, rho = hand_writer(init, sent_tens,  bias=bias)
        phi = hand_writer.attention_var['phi'][0,0,:,0]
        phis.append(phi)

        lift = rs.binomial(1, end[:, :, -1].cpu().detach().numpy())

        new_stroke[0, 0] = lift

        mu1_loc, mu2_loc = mu[:, :, :20], mu[:, :, 20:]
        sig1_loc, sig2_loc = sigma[:, :, :20], sigma[:, :, 20:]

        gauss = np.zeros((num_gauss_multi, 2))

        for j in range(num_gauss_multi):
            s1 = sig1_loc[:, :, j].cpu().detach().numpy().squeeze(0)
            s2 = sig2_loc[:, :, j].cpu().detach().numpy().squeeze(0)
            ro = rho[:, :, j].cpu().detach().numpy().squeeze(0)
            mu1 = mu1_loc[:, :, j].cpu().detach().numpy().squeeze(0)
            mu2 = mu2_loc[:, :, j].cpu().detach().numpy().squeeze(0)

            cov = np.array([[s1 ** 2, s1 * s2 * ro], [s1 * s2 * ro, s2 ** 2]]).reshape(
                2, 2
            )

            mean = np.array([mu1, mu2]).reshape(2)

            gauss[j] = pi[:, :, j].cpu().detach().numpy() * rs.multivariate_normal(
                mean, cov
            )

        new_stroke[0, 1:] = np.sum(gauss, axis=0)
        try:
            strokes = np.append(strokes, new_stroke, axis=0)
        except:
            strokes = new_stroke

        init = (
            torch.from_numpy(strokes[time])
            .type(torch.FloatTensor)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        if time > timestamps: 
            keep_going = False
        
        if (time>1 and phi[-1] == np.max(phi)):
            begin_end = True
        
        if begin_end and first and (phi[-1] < 1E-3) :
            first = False
            break
        
        time +=1
        
    if plot_attention:
        plot_phi(phis, sentence)

    return strokes
