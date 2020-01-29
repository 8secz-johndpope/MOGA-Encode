from matplotlib import pyplot as plt
import pygmo as pyg
import logging, csv, argparse
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
    plot_name = "Frontier of " + title
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

def plot_from_csv(filepath, title, perf_index, comp_index, add_ndf):
    '''
    Read data from csv file and redirect fitness data to plot_front function.
    '''

    fitness = []
    ndf = []

    with open(filepath) as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',')
        for row in csv_data:
            fitness.append( [float(row[perf_index]), float(row[comp_index])] )

    if add_ndf: ndf = pyg.non_dominated_front_2d(fitness)

    plot_front(title, fitness, ndf)


if(__name__ == "__main__"):
    '''
    Create plots from files
    '''
    cfg.timestamp = ""

    parser = argparse.ArgumentParser(description='Plot front from file data')
    parser.add_argument('filepath', help="path to the csv file containing fitness data")
    parser.add_argument('-n', '--name', default="Plot", help="name of plot")
    parser.add_argument('-p', '--perf', type=int, default=2, help="index of ml-perf column")
    parser.add_argument('-c', '--comp', type=int, default=3, help="index of comp-rate column")
    parser.add_argument('--ndf', action="store_true", help="set flag to separate dominated and non dominated fits")

    args = parser.parse_args()
    plot_from_csv(args.filepath, args.name, args.perf, args.comp, args.ndf)
