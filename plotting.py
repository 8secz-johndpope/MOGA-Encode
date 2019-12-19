from matplotlib import pyplot as plt
import logging
import config as cfg
logger = logging.getLogger('gen-alg')


def plot_front(title, fitness, ndf=[]):
    '''
    Unpacks fitness data and sends it to the plot_front function
    '''
    logger.debug("Plotting front of " + title)

    ml_perf = []
    comp_ratio = []
    color = []
    for i in range(0, len(fitness)):
        f = fitness[i]
        ml_perf.append(-f[0])
        comp_ratio.append(-f[1])
        color.append("b") if (i in ndf) else color.append("r")        

    # Draw plot
    fig, ax = plt.subplots()
    ax.scatter(comp_ratio, ml_perf, c=color, marker="1")
    ax.axhline(y = cfg.ML_PERFORMANCE_BASELINE, color="black",  linestyle="dotted")  # Add baseline
   
    # Plot styling
    plot_name = "Fronteer of " + title
    ax.set_title(plot_name)
    ax.set_ylabel("ML-performance")
    ax.set_xlabel("Compression ratio")
    ax.grid(False)
    ax.set_ylim(ymin=0., ymax=1.0)
    #ax.set_xlim(xmin=1, xmax=300)

    # Save plot
    fig.savefig(cfg.PLOT_PATH + cfg.timestamp +  "/pf-" + title +"-linear"+".png", dpi=300, format='png')

    ax.set_xscale('log')
    ax.set_xlabel("Compression ratio (logarithmic scale)")
    fig.savefig(cfg.PLOT_PATH + cfg.timestamp +  "/pf-" + title +"-log"+".png", dpi=300, format='png')

    plt.close(fig)


if(__name__ == "__main__"):
    '''
    Create simple plot for styling purposes
    '''
    cfg.timestamp = ""
    title = "Test"
    fitness = [[-0.9,-10],[-0.7,-25],[-0.35,-20],[-0.2,-30], [-0.25,-50]]
    ndf = [0,1,4]
    plot_front(title, fitness, ndf)
