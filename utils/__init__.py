import numpy as np
from matplotlib import pyplot


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()

def plot_stroke_list(stroke_list, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()
    offset = 0.

    x_max = None
    x_min = None
    y_max = None
    y_min = None

    for stroke in stroke_list:
        x = np.cumsum(stroke[:, 1])
        y = np.cumsum(stroke[:, 2])

        if x_max:
            
            x_max = max(x_max, offset + x.max())
            x_min = min(x_min, x.min())
            y_min = min(y_min, y.min())
            y_max = max(y_max, y.max())
        
        else:
            x_max = offset + x.max()
            x_min = x.min()
            y_min = y.min()
            y_max = y.max()

        cuts = np.where(stroke[:, 0] == 1)[0]
        start = 0

        for cut_value in cuts:
            if start - 1 > 0 :
                diff = x[start] - x[start - 1]
                if diff>12.:
                    # to cut the concatenation if there is a 2nd word
                    break
            ax.plot(offset + x[start:cut_value], y[start:cut_value],
                    'k-', linewidth=3)
            # ax.plot(offset + x[start], y[start],
            #         'ob', linewidth=3)
            # ax.plot(offset + x[cut_value], y[cut_value],
            #         'og', linewidth=3)
            start = cut_value + 1
        ax.axis('equal')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    
        offset += x[cut_value] + 2.

    size_x = x_max - x_min + 1.
    size_y = y_max - y_min + 1.

    pyplot.tick_params(
        axis='y',      
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    f.set_size_inches(5. * size_x / size_y, 5.)


    if save_name :
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.show()
    pyplot.close()

def plot_phi(phi, sentence=None, zoom=False):
    phis= phi/(np.sum(phi, axis = 0, keepdims=True))
    pyplot.figure(figsize=(15,6))
    pyplot.imshow(phis, cmap='BuPu', interpolation='nearest', aspect='auto')
    pyplot.xlabel('text')
    if sentence:
        pyplot.xticks(ticks = np.arange(len(sentence)), labels=[char for char in sentence])
    pyplot.ylabel('stroke')
    pyplot.show()
    
    if zoom:
        pyplot.figure(figsize=(15,6))
        pyplot.imshow(phis, cmap='BuPu', interpolation='nearest', aspect='auto')
        pyplot.xlabel('text')
        if sentence:
            pyplot.xticks(ticks = np.arange(len(sentence)), labels=[char for char in sentence])
        pyplot.ylabel('stroke')
        pyplot.ylim(0,400)
        pyplot.show()

