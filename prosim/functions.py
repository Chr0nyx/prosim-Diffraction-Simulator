"""
Functions module - functions.py
=====

Provides : useful functions for workflow

  1. progress_bar() : Print computation progress
  2. notify_user :  Play system notification
  3. stripe : "Cut" one dim array from 2d array 
  4. split : Split array from center to end
  5. absolute_square : Evaluate absolute square of array
  6. power_spectrum : Evaluate power spectrum
  7. normalize_beam : Normalize beam from propagation.propagate
  8. evaluate_beam_radius : Evaluate beam radius from propagation.propagate
  9. abs_error : Evaluate absolute error for given array
  10. rel_error : Evaluate relative error for given array
  11. make_single_xy_graph : Create pgf for xy-graph with multiple datasets
  12. make_multiple_xy_graphs : Create pgf for multiple xy-graphs each with one dataset
  13. make_multiple_colorgraphs : Create pgf for mutliple matplot imshow-graphs
  14. show_single_xy_graph : Print single xy-graph with multiple datasets
  15. show_multiple_colorgraphs :  Print multiple matplot imshow graphs

"""
from plyer import notification
import numpy as np

def progress_bar(current: int, total: int, barLength=20, title="computation"):
    """Print computation progress for loops etc.

    Attributes:
    =====
    current (int, scalar) : loop index 
    
    total (int, scalar) : loop end

    barLength (int, scalar) : displayed bar length in cli.

    title (string) : displayed title for computation
    """
    # Calculate the progress in percent
    percent = float(current) * 100 / total

    # define the output arrow
    arrow = '=' * int(percent/100 * barLength-1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    # print the progress bar
    if percent == 100:
        print(title,'progress: [%s%s] %d %%' % (arrow, spaces, percent))
    else:
        print(title,'progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def notify_user(title: str, message: str):
    """Lanuch system notification to inform user about completed computation.

    Attributes:
    =====
    title (string) : displayed title

    message (string) : displayed message
    """

    notification.notify(title = title, message = message, 
                                app_icon = None,  timeout = 10,)

def stripe(array: complex):
    """Return a strip from two-dimensional array-like `[[],[],[]]`.

    Attributes:
    =====
    array (complex, array-like) : target array (2D)
    """
    l = int(len(array)/2)
    return array[l][0:l]        

def split(array: complex):
    """Return a split array-like ´[a,b,c]´.

    len(array) even: '[a,b,c,d] -> [c,d]'
    len(array) odd: '[a,b,c,d,e,f,g] -> [d,e,f,g]

    Attributes:
    =====
    array (complex, array-like) : target array (1D)
    """
    return  np.array_split(array, 2)[1]

def absolute_square(array: complex):
    """Return absolute square of array-like.

    Attributes:
    =====
    array (complex, array-like) : target array (2D)
    """
    return array.real**2+array.imag**2

def power_spectrum(spectrum: complex):
    """Return power spectrum of argument array.

    Attributes:
    =====
    spectrum (complex, array-like) : target array (2D) representing spectrum
    """

    return absolute_square(spectrum)/np.max(absolute_square(spectrum))
    

def normalize_beam(beam: complex):
    """Return normalized beam.
    Normalization is caclulated at each z_i in z. Plot therefore has no physical
    meaning. Only visualizes growth of beam radius.

    Attributes:
    =====
    beam (complex, array-like) : target array (2D) representing beam \
                                 as returned by `calculate_continous()` or \
                                 `calculate_pulse()` from `propagation.py`.

    """
    Nx = len(beam)  
    Nz = len(beam[0])

    Irz_norm = np.zeros((Nx, Nz), dtype=float)

    for i in range(0, Nz):  
        progress_bar(i, Nz, title='normalize')

        max_abs_square = np.max(absolute_square( beam[:, i]))
        Irz_norm_i = absolute_square(beam[:, i])/max_abs_square
        Irz_norm[:, i] = Irz_norm_i

    return Irz_norm

def evaluate_beam_radius(beam: complex, dx: float):
    """Return beam radius as function of propagation distance.

    Attributes:
    =====
    beam (complex, array-like) : target array (2D) representing beam \
                                 as returned by `calculate_continous()` or \
                                 `calculate_pulse()` from `propagation.py`.

    dx (float, scalar) : discrete spatial sample from wave object used to
                          evaluate beam (wave.dx).
    """
    Nx = len(beam)  
    Nz = len(beam[0])

    wz_num = np.zeros(Nz, dtype=float)

    for i in range(0, Nz):  
        progress_bar(i, Nz, 20, 'beam-radius')

        E_i = beam[:, i]
        max_abs_square = np.max(absolute_square(E_i))
        Irz_norm_i = absolute_square(E_i)/max_abs_square

        for j, Irz_norm_j in enumerate(Irz_norm_i[::-1]):
            Irz_norm_i_max = np.max(Irz_norm_i)

            if Irz_norm_j < Irz_norm_i_max/np.exp(2):
                wz_num[i] = j*dx    
                break

    return wz_num

def abs_error(measured: float, actual: float):
    """Return difference between measured and actually numpy array.

    Attributes:
    =====
    measured (float, array-like) : numeric result

    actual (float, array-like) : "analytic" result
    """  
    return measured-actual

def rel_error(measured: float, actual: float):
    """Return relative difference between measured and actually numpy array.

    Attributes:
    =====
    measured (float, array-like) : numeric result

    actual (float, array-like) : "analytic" result
    """    
    return abs_error(measured, actual)/(actual)*100


def make_single_xy_graph(path: str, x: float, y: float, scale: float, style: str, legend: str, xlabel: str, ylabel: str, title: str, width_in: float,  height_in: float):
    """Create single xy-graph with multiple datasets (x,y) in plot and safe
    to specified directory.

    Attributes:
    =====
    path (string) : destination path with file name e.g.: /home/user/filename

    x (float, array-like) : dataset for x-axis

    y (float, array-like) : dataset for y-axis

    scale (float, scalar) : factor to scale unit in xy-labels

    legend (string, array-like) : name respective datasets 

    xlabel (string) : label (usually phys. unit) for x-axis.

    ylabel (string) : label (usually phys. unit) for y-axis.
    
    title (string) : title of entire plot

    width_in (float, scalar) : plot width in inches
    
    height_in (float, scalar): plot height in inches
    """    

    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex",'font.family': 'serif',
                                'text.usetex': True, 'pgf.rcfonts': False,})
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in))

    for i in range(0, len(x)):
        ax.plot(x[i]*scale, y[i]*scale, style[i], linewidth='0.7')


    ax.grid(which='major', axis='both')
    ax.legend(legend)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    plt.savefig(path+'.pgf')
    print('Plot has been saved to specified path.\n')

    return 


def make_multiple_xy_graphs(path, x, y, style, xlabel, ylabel, title, width_in, height_in, row=1, col=1, id=['a'],scale=[1]):
    """Create multiple xy-graphs with single dataset (x,y) in each plot and safe
    to specified directory.

    Attributes:
    =====
    path (string) : destination path with file name e.g.: /home/user/filename

    x (float, array-like) : data for x-axis.

    y (float, array-like) : data for y-axis.

    style (string, list) : style for datasets. 

    xlabel (string, array-like) : label (usually phys. unit) for x-axis.

    ylabel (string, array-like) : label (usually phys. unit) for y-axis.

    title (string) : Title for entire plot.
    
    width_in (float, scalar) : Width of entire plot in inch.

    height_in (float, scalar) : Height of entire plot in inch.

    row (int, scalar) : number of rows 

    col (int, scalar) : number of cols
                        number of figures = cols * rows
    
    id (string, list) : Character to identify respective plot.

    scale (float, array-like) : scale to fit specified unit in label
    """   
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    })
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText

    fig, axes = plt.subplots(row, col, figsize=(width_in, height_in))
    axes = axes.flatten() #flatten array to iterate easily

    for i, ax in enumerate(axes):
        ax.plot(x[i]*scale[i], y[i]*scale[i], style[i], linewidth='0.7')
        ax.grid(which='major', axis='both')
        ax.set_xlabel(xlabel[i])
        ax.set_ylabel(ylabel[i])
        ax.set_title(title)
        at = AnchoredText(id[i], prop=dict(size=10), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

    fig.tight_layout()
    plt.savefig(path+'.pgf')
    print('Plot has been saved to specified path.\n')


    return 

def make_multiple_colorgraphs(path, arrays, extent, xlabel, ylabel, xlim, ylim, title, width_in, height_in, cmap='binary', row=1, col=1, id=['']):
    """Create multiple colorzied graphs with array dataset in each figure.

    Attributes:
    =====
    path (string) : destination path with file name e.g.: /home/user/filename

    arrays (float, array-like) : consists of 2D-arrays to be printed.

    extent (float) : extent for plot (has to be the same for every figure).
                      See matplotlib -> imshow -> extent attribute

    xlabel (string, array-like) : label (usually phys. unit) for x-axis.

    ylabel (string, array-like) : label (usually phys. unit) for y-axis.

    xlim (float, list) : limit for x-axis.

    ylim (float, list) : limit for y-axis.

    title (string) : Title for entire plot.
    
    width_in (float, scalar) : Width of entire plot in inch.

    height_in (float, scalar) : Height of entire plot in inch.

    cmap (string) : Color code specifier.

    row (int, scalar) : number of rows 

    col (int, scalar) : number of cols
                        number of figures = cols * rows
    
    id (string, list) : Character(s) to identify respective plot.
    """   
     
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    })
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText

    fig, axes = plt.subplots(row, col, figsize=(width_in, height_in))
    axes = axes.flatten()#flatten array to iterate easily


    for i, ax in enumerate(axes):
        p=ax.imshow(arrays[i], cmap=cmap, extent=extent, aspect='auto')
        fig.colorbar(p, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-xlim[i],xlim[i])
        ax.set_ylim(-ylim[i],ylim[i])
        ax.set_title(title)
        at = AnchoredText(id[i], prop=dict(size=10), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

    

    fig.tight_layout()
    plt.savefig(path+'.pgf')
    print('Plot has been saved to specified path.\n')

    return 


def make_single_colorgraph(path, array, extent, xlabel, ylabel, title, width_in, height_in, cmap='binary'):
    """Create multiple colorzied graphs with array dataset in each figure.

    Attributes:
    =====
    path (string) : destination path with file name e.g.: /home/user/filename

    arrays (float, array-like) : consists of 2D-arrays to be printed.

    extent (float) : extent for plot (has to be the same for every figure).
                      See matplotlib -> imshow -> extent attribute

    xlabel (string, array-like) : label (usually phys. unit) for x-axis.

    ylabel (string, array-like) : label (usually phys. unit) for y-axis.

    xlim (float, list) : limit for x-axis.

    ylim (float, list) : limit for y-axis.

    title (string) : Title for entire plot.
    
    width_in (float, scalar) : Width of entire plot in inch.

    height_in (float, scalar) : Height of entire plot in inch.

    cmap (string) : Color code specifier.
    """   
     
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    })
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in))

    p=ax.imshow(array, cmap=cmap, extent=extent, aspect='auto')
    fig.colorbar(p, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xlim(-xlim,xlim)
    #ax.set_ylim(-ylim,ylim)
    ax.set_title(title)    

    fig.tight_layout()
    plt.savefig(path+'.pgf')
    print('Plot has been saved to specified path.\n')

    return 


def show_single_xy_graph(x, y, scale, style, legend, xlabel, ylabel, title, width_in, height_in, row=1, col=1):
    """Print single xy-graph with multiple datasets (x,y) in plot.

    Attributes:
    =====
    x (float, array-like) : dataset for x-axis

    y (float, array-like) : dataset for y-axis

    scale (float, scalar) : factor to scale unit in xy-labels

    legend (string, array-like) : name respective datasets 

    xlabel (string) : label (usually phys. unit) for x-axis.

    ylabel (string) : label (usually phys. unit) for y-axis.
    
    title (string) : title of entire plot

    width_in (float, scalar) : plot width in inches
    
    height_in (float, scalar): plot height in inches
    """    

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(row, col, figsize=(width_in, height_in))

    for i in range(0, len(x)):
        ax.plot(x[i]*scale, y[i]*scale, style[i], linewidth='0.7')


    ax.grid(which='major', axis='both')
    #ax.legend(legend)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    plt.show()

    return


def show_multiple_colorgraphs(arrays, extent, xlim, ylim, xlabel, ylabel, title, width_in, height_in, cmap='binary', row=1, col=1):
    """Print multiple colorzied graphs with array dataset in each figure.

    Attributes:
    =====
    arrays (float, array-like) : consists of 2D-arrays to be printed.

    extent (float) : extent for plot (has to be the same for every figure).
                      See matplotlib -> imshow -> extent attribute

    xlabel (string, array-like) : label (usually phys. unit) for x-axis.

    ylabel (string, array-like) : label (usually phys. unit) for y-axis.

    xlim (float, list) : limit for x-axis.

    ylim (float, list) : limit for y-axis.

    title (string) : Title for entire plot.
    
    width_in (float, scalar) : Width of entire plot in inch.

    height_in (float, scalar) : Height of entire plot in inch.

    cmap (string) : Color code specifier.

    row (int, scalar) : number of rows 

    col (int, scalar) : number of cols
                        number of figures = cols * rows
    """   

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(row, col, figsize=(width_in, height_in))
    axes = axes.flatten()#flatten array to iterate easily


    for i, ax in enumerate(axes):
        p=ax.imshow(arrays[i], cmap=cmap, extent=extent, aspect='auto')
        fig.colorbar(p, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-xlim[i],xlim[i])
        ax.set_ylim(-ylim[i],ylim[i])
        ax.set_title(title)
    

    fig.tight_layout()
    plt.show()


    return