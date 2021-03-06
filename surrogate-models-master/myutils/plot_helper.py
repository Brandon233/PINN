__author__ = "Juri Bieler"
__version__ = "0.0.1"
__email__ = "juribieler@gmail.com"
__status__ = "Development"

# ==============================================================================
# description     :helper class for fancy latex ready plots
# date            :2018-07-23
# version         :0.01
# notes           :
# python_version  :3.6
# ==============================================================================


import numpy as np


class PlotHelper:

    def __init__(self, axis_labels, fancy=False, font_size=12, ax=None, pgf=False):
        self._use_pgf = pgf
        if pgf:
            import matplotlib
            matplotlib.use('pgf')
            pgf_with_custom_preamble = {
                'pgf.rcfonts': False
            }
            matplotlib.rcParams.update(pgf_with_custom_preamble)
        import matplotlib.pyplot as plt
        from matplotlib import rc
        #plt.style.use('tableau-colorblind10')
        #plt.style.use('ggplot')

        self.fig = None
        self.FONT_SIZE = font_size
        self.font = {'family': 'sans-serif', 'size': self.FONT_SIZE}
        self.labelpad = 8
        if len(axis_labels) == 2:
            if ax is None:
                self.fig, self.ax = plt.subplots()
            else:
                self.ax = ax
            self.ax.set_xlabel(axis_labels[0], fontdict=self.font)
            self.ax.set_ylabel(axis_labels[1], fontdict=self.font)
            rc('xtick', labelsize=self.FONT_SIZE)
            rc('ytick', labelsize=self.FONT_SIZE)
        elif len(axis_labels) == 3:
            from mpl_toolkits.mplot3d import Axes3D
            if ax is None:
                self.fig = plt.figure()
                self.ax = self.fig.gca(projection='3d')
            else:
                self.ax = ax
            self.ax.set_xlabel(axis_labels[0], fontdict=self.font, labelpad=self.labelpad)
            self.ax.set_ylabel(axis_labels[1], fontdict=self.font, labelpad=self.labelpad)
            self.ax.set_zlabel(axis_labels[2], fontdict=self.font, labelpad=self.labelpad)
            rc('xtick', labelsize=self.FONT_SIZE)
            rc('ytick', labelsize=self.FONT_SIZE)
            #rc('ztick', labelsize=self.FONT_SIZE)
            #self.ax.xaxis._axinfo['label']['space_factor'] = 4
        elif len(axis_labels) == 0:
            self.fig = plt.figure()
            self.ax = None
        else:
            raise ValueError('length of axis_labels should be 2(D) or 3(D)')
        if self.ax != None:
            self.ax.tick_params(labelsize=self.FONT_SIZE, length=6, width=2)
        if fancy or pgf:
            rc('text', usetex=True)
            import matplotlib as mpl
            # mpl.rcParams['text.latex.unicode'] = True
        rc('font', **self.font)

    def finalize(self, width=6, height=4, legendLoc=1, legendNcol=1, bbox_to_anchor=None, tighten_layout=True, show_legend=True):
        legend = None
        if self.ax != None and show_legend:
            if bbox_to_anchor is not None:
                legend = self.ax.legend(loc=legendLoc, ncol=legendNcol, bbox_to_anchor=bbox_to_anchor, fancybox=True)
            else:
                legend = self.ax.legend(loc=legendLoc, ncol=legendNcol, fancybox=True)
        if self.fig is not None:
            self.fig.set_size_inches(width, height)
        if self.ax != None:
            self.ax.autoscale_view(tight=True)
        if tighten_layout:
            import matplotlib.pyplot as plt
            plt.tight_layout()
        return legend

    def save(self, file_path, transparent=True):
        import matplotlib.pyplot as plt
        plt.savefig(file_path, transparent=transparent)
        if self._use_pgf and '.pdf' in file_path:
            plt.savefig(file_path.replace('pdf', 'pgf'), transparent=transparent)

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    def animate(self):
        import matplotlib.pyplot as plt
        for angle in np.linspace(0, 360, 1000):
            self.ax.view_init(30, angle)
            plt.draw()
            #print(str(angle))
            plt.pause(.001)

    def plot_function_3d(self, f, fx, fy, label, color='b', scale=[1., 1., 1.], offset=[0., 0., 0.]):
        fz = np.zeros((len(fy), len(fx)))
        for iX in range(0, len(fx)):
            for iY in range(0, len(fy)):
                coords = [fx[iX], fy[iY]]
                fz[iY][iX] = f(coords)

        plotX, plotY = np.meshgrid(fx, fy)
        surf = self.ax.plot_wireframe((plotX * scale[0])+offset[0],
                                      (plotY * scale[1])+offset[1],
                                      (fz * scale[2])+offset[2],
                                      color=color,
                                      label=label,
                                      rcount=20,
                                      ccount=20,
                                      linewidths=1,
                                      alpha=0.5)
        return surf

    '''
    def plot_function_2D(self, f, fx, label, color='b'):
        fz = np.zeros((len(fx), 1))
        for iX in range(0, len(fx)):
            fz[iX][0] = f(fx[iX])

        plotX, plotY = np.meshgrid(fx, fy)
        surf = self.ax.plot_wireframe(plotX, plotY, fz, color=color, label=label, rcount=20, ccount=20,
                                       linewidths=1,
                                       alpha=0.5)  # , rstride=1, cstride=1)#, cmap=cm.coolwarm) # ,linewidth=0, antialiased=False
        return surf
    '''