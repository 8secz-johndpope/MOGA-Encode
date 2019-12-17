from matplotlib import pyplot as plt
import logging
import config as cfg
logger = logging.getLogger('gen-alg')


def plot_front_from_pop(pop, title):
    '''
    Unpacks population data and sends it to the plot_front function
    '''
    front = pop.get_f()
    ml_perf = []
    comp_ratio = []
    for f in front:
        ml_perf.append(-f[0])
        comp_ratio.append(-f[1])
    plot_front(ml_perf, comp_ratio, title)

def plot_front_from_fitness(fitness, title):
    '''
    Unpacks fitness data and sends it to the plot_front function
    '''
    ml_perf = []
    comp_ratio = []
    for f in fitness:
        ml_perf.append(-f[0])
        comp_ratio.append(-f[1])
    plot_front(ml_perf, comp_ratio, title)

def plot_front(ml_perf, comp_ratio, title):
    '''
    Draws a scatter plot from ml_perf and comp_ratio data points.
    The plot is saved into a directory unique for the current optimization session.
    '''
    logger.debug("Plotting front of " + title)
    plot_name = "Fronteer of " + title

    # Draw plot
    fig, ax = plt.subplots()
    ax.scatter(comp_ratio, ml_perf, marker=".")
    ax.axhline(y = cfg.ML_PERFORMANCE_BASELINE, color="r")  # Add baseline
    
    # Plot styling
    ax.set_title(plot_name)
    ax.set_ylabel("ML-performance")
    ax.set_xlabel("Compression ratio")
    ax.grid(True)
    ax.set_xscale('log')
    #ax.set_xlim(xmin=1, xmax=300)
    ax.set_ylim(ymin=0., ymax=1.0)

    # Save plot
    fig.savefig(cfg.PLOT_PATH + cfg.timestamp +  "/pf-" + title +".png", dpi=300, format='png')
    
