import numpy

from models.generators import generate_unconditional, generate_conditional

strokes = numpy.load('../data/strokes-py3.npy', allow_pickle=True)
stroke = strokes[0]


def generate_unconditionally(random_seed=1):
    """
    Input:
      random_seed - integer

    Output:
      stroke - numpy 2D-array (T x 3)
    """
    config = {
        "model_path": '../models/files/model_generate_unconditionally.pt',
        "hidden_size": 900,
        "num_gauss": 20,
        "timestamps": 1000,
        "seed": random_seed
        }

    stroke = generate_unconditional(**config)
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1, model_path = '../models/files/model_generate_conditionally.pt', bias = 1., plot_attention=False):
    """Creating a list of strokes corresponding to each word of the sentence given as argument
    
    Keyword Arguments:
        text {str} -- sentence to transform in a stroke (default: {'welcome to lyrebird'})
        random_seed {int} -- value of the seed (default: {1})
        model_path {str} -- path to the saved weight of a model (default: {'../models/files/model_generate_conditionally.pt'})
        bias {[type]} -- value of the bias used to calculate the coefficient of the probability distributions (default: {1.})
        plot_attention {bool} -- show the display of the attention windows (True) or not (False) (default: {False})
    """
    stroke_list = []

    text_list = text.split(" ")
    for word in text_list:
        word = word + " "
        config = {
            "model_path": model_path,
            "hidden_size": 400,
            "num_gauss_multi": 20,
            "timestamps": 900,
            "seed": random_seed,
            "sentence": word,
            'bias': bias,
            'plot_attention': plot_attention
            }
        stroke_list.append(generate_conditional(**config))
    return stroke_list

def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'


