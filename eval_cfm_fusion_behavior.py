#This script contains code for replicating Figure 2 from the paper "Bayesian Classifier Fusion with an Explicit Model of Correlation"
#For two examples with different marginal distributions two 3-dimensional categorical base distributions are fused with the Correlated
#Fusion Model assuming successively higher correlations between the corresponding base classifiers. The fused distributions are
#additionally compared to the result distributions of the two meta classifiers that can be obtained when using the Independent Fusion
#Model with K = 1 for each individual classifier.


from __future__ import division
import numpy as np

import matplotlib as mpl
rc_fonts = {
    "text.usetex": True,
    'text.latex.preview': True, # Gives correct legend alignment.
    #'mathtext.default': 'regular',
    'mathtext.fontset': 'cm',
    'font.family':'Times New Roman',
    'text.latex.preamble': [r"""\usepackage{bm}"""],
}
mpl.rcParams.update(rc_fonts)
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


fs = 11.5

plt.rc('xtick', labelsize = fs)
plt.rc('ytick', labelsize = fs)

np.set_printoptions(precision=3)


#import code of IFM and CFM
import independent_fusion_model
import correlated_fusion_model






#fuse base distributions x1 and x2 with J dimensions using the CFM with marginal parameters alpha and assuming different correlation levels
#additionally create the meta classifiers' result distributions
#x1: categorical base distribution x1
#x2: categorical base distribution x2
#J: number of classes / dimensions of the categorical distributions
#alpha: marginal parameters of modeling correlated Dirichlet distributions, shape (2,J,J)
def create_fused_and_meta_dists(x1, x2, J, alpha):

    #correlations that will be assumed for fusing with CFM
    correlations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9, 1.0]
    #container for storing all fused distributions
    all_fused = np.zeros((len(correlations), J))

    #loop over all correlations
    for i in range(len(correlations)):
        r = correlations[i]
        #set correlation parameters delta according to r:
        if r == 0:
            delta = np.array([[0.1,0.1,0.1], [0.1,0.1,0.1], [0.1,0.1,0.1]])  #generates a correlation of approx 0
        elif r == 1:
            #set delta depending on the marginal parameters alpha
            if alpha[0,0,0] == 101:
                delta = np.array([[100.9,99.9,99.9], [99.9,100.9,99.9], [99.9,99.9,100.9]])  #generates a correlation of approx 1
            else:
                delta = np.array([[101.9,99.9,99.9], [99.9,101.9,99.9], [99.9,99.9,101.9]])  #generates a correlation of approx 1
        else:
            delta = r * alpha[0] # generates a correlation of approx r


        #fuse
        fused_cat, samples_t = correlated_fusion_model.sample_t(J, x1, x2, alpha, delta)

        #console output
        print('r = %.1f' %r)
        print('fused distribution:')
        print(fused_cat)

        #add fused distribution to container
        all_fused[i] = fused_cat


    #meta classifier results

    #use the Independent Fusion Model as meta classifier individually for
    # x1 given marginal Dirichlet parameters of classifier 1 (alpha[0])
    # x2 given marginal Dirichlet parameters of classifier 2 (alpha[1])
    meta1 = independent_fusion_model.fuse_analytical(1, J, np.array([x1]), np.array([alpha[0]]))
    meta2 = independent_fusion_model.fuse_analytical(1, J, np.array([x2]), np.array([alpha[1]]))


    #merge base distributions, fused distributions, and meta classifier's results
    dists = np.vstack((x1, x2, all_fused, meta1, meta2))

    #return all distributions (base distributions, fused distributions, meta classifiers' results)
    return dists



#plot the base distributions, fused distributions for different correlations, and meta classifiers' results as bar plots
#ax: the axis to plot
#dists: the distributions to be plotted
#offset: the offset of the bars (to be able to plot the bars of the two examples side by side)
#colors: the colors of the bars
def plot_results(dists, ax, offset, colors):
    #width of bars
    width = 0.4

    nr_dists = dists.shape[0]

    #set x ticks of the categorical distributions in the plot
    x_ticks = np.array([0,1,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5, 14,15])

    #plot categorical distributions as bar plots
    p1 = ax.bar(x_ticks + offset, dists[:,0], width = width, color = colors[0], edgecolor = 'black')
    p2 = ax.bar(x_ticks + offset, dists[:,1], width = width, bottom = dists[:,0], color = colors[1], edgecolor = 'black')
    p3 = ax.bar(x_ticks + offset, dists[:,2], width = width, bottom = dists[:,0] + dists[:,1], color = colors[2], edgecolor = 'black')

    #x lim, ticks, and labels
    ax.set_xlim([-0.7,15.7])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r'$\bm{x_i^1}$', r'$\bm{x_i^2}$', 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                            r'$\bm{m_i^1}$', r'$\bm{m_i^2}$'], fontsize = fs)
    ax.set_xlabel(' ', fontsize = fs, labelpad = 1.2)
    ax.text(-0.1,-0.3,'base', fontsize = fs, multialignment = 'center')
    ax.text(4.65,-0.3,'fused dependent on $r$', fontsize = fs, multialignment = 'center')
    ax.text(13.9,-0.3,'meta', fontsize = fs, multialignment = 'center')
    
    #y lim, ticks, and labels
    ax.set_ylim([0,1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('categ. distributions', fontsize = fs)

    #for creating the legend, return p1, p2, p3
    return p1, p2, p3






####################
#reproduce figure 2#


#set model parameters applying to both examples 1 and 2

#nr of classes
J = 3

#base distributions to be fused
x1 = np.array([0.6,0.2,0.2])
x2 = np.array([0.6,0.2,0.2])



#set up plot
fig, ax = plt.subplots()
fig.set_size_inches(6,2.5)




#first example (IOP marginals, blue bars in figure 2)

#marginal parameters of modeling correlated Dirichlet distribution
alpha1 = np.array([[[101,100,100], [100,101,100], [100,100,101]],      #marginal parameters of classifier 1 for t_i = 1, t_i = 2, t_i = 3
                    [[101,100,100], [100,101,100], [100,100,101]]])    #marginal parameters of classifier 2 for t_i = 1, t_i = 2, t_i = 3


#fuse x1 and x2 with the CFM using marginals alpha1 and different correlation levels, and create meta classifiers' result distributions
dists1 = create_fused_and_meta_dists(x1, x2, J, alpha1)


#set offset and bar colors for plot
offset1 = -0.2
colors1 = ['lightskyblue', 'royalblue', 'navy']


#plot the resulting distributions
p1, p2, p3 = plot_results(dists1, ax, offset1, colors1)



#second example (marginals leading to more uncertainty reduction, orange bars in figure 2)

#marginal parameters of modeling correlated Dirichlet distribution
alpha2 = np.array([[[102,100,100], [100,102,100], [100,100,102]],      #marginal parameters of classifier 1 for t_i = 1, t_i = 2, t_i = 3
                    [[102,100,100], [100,102,100], [100,100,102]]])    #marginal parameters of classifier 2 for t_i = 1, t_i = 2, t_i = 3


#fuse x1 and x2 with the CFM using marginals alpha2 and different correlation levels, and create meta classifiers' result distributions
dists2 = create_fused_and_meta_dists(x1, x2, J, alpha2)


#set offset and bar colors for plot
offset2 = 0.2
colors2 = ['orange', "#FF6F00", 'sienna']


#plot the resulting distributions
p4, p5, p6 = plot_results(dists2, ax, offset2, colors2)




#set legend
ax.legend([(p1, p4), (p2, p5), (p3, p6)], [r'$p(t_i = 1)$', r'$p(t_i = 2)$', r'$p(t_i = 3)$'], loc = 'lower right', fontsize = fs,
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=2, pad = 0)})


#show plot
plt.tight_layout()
plt.show()