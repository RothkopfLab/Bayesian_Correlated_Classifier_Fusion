#This script contains the functionality of the Independent Fusion Model, which is a Bayesian model
#for the fusion of independent classifiers that output categorical distributions.


from __future__ import division
import pyjags
import numpy as np
import scipy.stats

np.set_printoptions(precision=3)

from utils import create_true_labels, iop



#JAGS definition of the generative model of the Independent Fusion Model
jags_model_code = '''
model {

    for (i in 1:I) {

        t[i] ~ dcat(p)

        for (k in 1:K) {

            #choose the respective parameters for classifier k according to the true label t[i] of example i
            a[i,k,1:J] <- alpha[k,t[i],1:J]

            #model classifier k's output distribution with a Dirichlet distribution
            x[i,k,1:J] ~ ddirch(a[i,k,1:J])

        }
    }

    #uninformed prior for alpha
    for (k in 1:K) {
        for (j in 1:J) {
            for (l in 1:J) {
                alpha[k,j,l] ~ dgamma(0.001,0.001)
            }
        }
    }

    #define parameter p of uninformed prior of the true class labels
    for (j in 1:J) {
        p[j] <- 1/J
    } 
}
'''
        


#data generation
#sample simulated classifier outputs while observing parameters alpha and the true class labels
#K: number of classifiers
#J: number of classes
#alpha: parameters of modeling Dirichlet distributions, shape (K,J,J)
#true_labels: list of true class labels of the classifier outputs to be generated
def sample_x(K, J, alpha, true_labels):
    I = len(true_labels) #nr of examples
    model = pyjags.Model(jags_model_code, data = dict(K = K,
                                                        J = J,
                                                        alpha = alpha,
                                                        t = true_labels,
                                                        I = I), chains = 1, adapt = 1000)
    samples = model.sample(1, vars = ['x']) #we just draw one sample containing I examples
    values = samples['x'].squeeze()
    return values



#parameter inference
#sample parameters alpha while observing I classifier outputs x and their true labels
#K: number of classifiers
#J: number of classes
#x: observed categorical classifier outputs of all classifiers, shape (I,K,J)
#true_labels: list of true class labels of the observed classifier outputs
#nr_samples: number of samples we draw for inferring the parameters
#returns mean of sampled alphas and raw samples
def sample_alpha(K, J, x, true_labels, nr_samples = 1000):
    I = len(true_labels) #nr of examples
    model = pyjags.Model(jags_model_code, data = dict(K = K,
                                                        J = J,
                                                        x = x,
                                                        t = true_labels,
                                                        I = I), chains = 1, adapt = 500)
    samples = model.sample(nr_samples, vars = ['alpha'])
    samples = samples['alpha'].squeeze()

    #compute mean of alpha samples
    mean_alpha = np.mean(samples, axis = 3)

    #return mean of alphas and raw samples
    return mean_alpha, samples



#fusion by Gibbs Sampling
#sample true labels t from observed classifier output distributions x and given model parameters alpha for fusion
#K: number of classifiers
#J: number of classes
#x: categorical output distribution of classifiers 1,...,K, shape (K,J)
#alpha: parameters of the modeling Dirichlet distributions, shape (K,J,J)
#nr_samples: number of samples we draw for inferring t
#returns the fused categorical distribution computed from the samples and the respective raw samples
def sample_t(K, J, x, alpha, nr_samples = 10000):
    I = 1 #we only cosider one example here

    x = x.reshape((1, K, J))

    model = pyjags.Model(jags_model_code, data = dict(K = K,
                                                        J = J,
                                                        x = x,
                                                        alpha = alpha,
                                                        I = I), chains = 1, adapt = 10000)
    samples = model.sample(nr_samples, vars = ['t'])
    samples = samples['t'].squeeze()

    #compute categorical distribution from samples
    unique, counts = np.unique(samples, return_counts=True)
    count_dict = dict(zip(unique, counts))
    label_counts = np.zeros(J)
    for i in range(1, J + 1): #compensate that JAGS works 1-based
        if i in count_dict.keys():
            label_counts[i-1] = count_dict[i]
    fused_cat = label_counts / nr_samples

    #return fused categorical and raw samples
    return fused_cat, samples



#fusion with analytical formula
#fuse classifier outputs x given model parameters alpha using the analytical formula given in equation (2)
#K: number of classifiers
#J: number of classes
#x: categorical output distribution of classifiers 1,...,K, shape (K,J)
#alpha: parameters of the modeling Dirichlet distributions, shape (K,J,J)
#returns the fused categorical distribution
def fuse_analytical(K, J, x, alpha):
    fused_cat = np.zeros(J)
    for j in range(J):
        prod = 1
        for k in range(K):
            prod *= scipy.stats.dirichlet.pdf(x[k], alpha[k,j])
        fused_cat[j] = prod
    return fused_cat / np.sum(fused_cat)




if __name__ == "__main__":


    #additional imports only needed for testing code below
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
    import sys


    
    ######################
    #set model parameters#

    
    K = 2 # nr of classifiers
    J = 3 # nr of classes

    #Dirichlet model parameters
    alpha = np.array([[[3,1,1], [1,3,1], [1,1,3]],      #parameters of classifier 1 for t_i = 1, t_i = 2, t_i = 3
                        [[3,1,1], [1,3,1], [1,1,3]]])   #parameters of classifier 2 for t_i = 1, t_i = 2, t_i = 3

    
    #create simulated true labels
    nr_examples_per_class = 500 
    true_labels = create_true_labels(J, nr_examples_per_class)



    #############################
    #read command line arguments#


    #to decide which functionality should be tested:
    #data generation: argument is 'data_generation'
    #parameter inference: argument is 'inference'
    #fusion: argument is 'fusion'
    run_args = sys.argv
    if len(run_args) > 1:
        run_mode = run_args[1]
    #if no argument is given, all functionalities are tested
    else:
        run_mode = 'all'

    
    
    #######################################
    #generate simulated classifier outputs#

    if run_mode == 'all' or run_mode == 'data_generation' or run_mode == 'inference':

        print('### Data Generation ###')

        
        samples_x = sample_x(K, J, alpha, true_labels)

        print('Generated data of shape:')
        print(samples_x.shape)


        #plot generated classifier outputs if K=2

        if K == 2:

            fig,ax = plt.subplots(J,J, sharex=True, sharey=True)
            fig.set_size_inches(10,10)
            fig.suptitle(r'Generated Classifier Outputs $\bm{x_i^1}$ and $\bm{x_i^2}$ of 2 Independent Classifiers'
                            '\n' '- the categorical distributions are plotted columnwise per dimension -'
                            '\n' r'- the rows show the classifier models for different true class labels $t_i$ -',
                            fontsize = 18)


            for j in range(J):
                for l in range(J):
                    ax[l,j].scatter(samples_x[l*nr_examples_per_class:(l+1)*nr_examples_per_class, 0, j],
                                    samples_x[l*nr_examples_per_class:(l+1)*nr_examples_per_class, 1, j],
                                    alpha = 0.1)

            #plot cosmetics
            for i in range(J):
                ax2 = ax[i,J-1].twinx()
                ax2.set_yticks([])
                ax2.set_ylabel('$t_i = %d$'%(i+1), rotation = 0, labelpad = 25, fontsize = 15)
                for j in range(J):
                    ax[i,j].set_xlim([0,1])
                    ax[i,j].set_ylim([0,1])
                    ax[i,j].set_xticks([0,0.5,1])
                    ax[i,j].set_yticks([0,0.5,1])

                    ax[i,j].set_xticklabels([0,0.5,1], fontsize = 15)
                    ax[i,j].set_yticklabels([0,0.5,1], fontsize = 15)


            ax[J-2,0].set_ylabel(r'$\bm{x_i^2}$', labelpad = 15, fontsize = 18)
            ax[J-1,1].set_xlabel(r'$\bm{x_i^1}$', labelpad = 15, fontsize = 18)


            plt.show()


    

    ########################
    #infer parameters alpha#

    if run_mode == 'all' or run_mode == 'inference':

        print('### Parameter Inference ###')


        #use previously generated data as training data
        train_data = samples_x

        alpha_inf, samples_alpha = sample_alpha(K, J, train_data, true_labels)

        print('inferred alphas:')
        print(alpha_inf)



    ########
    #fusion#

    if run_mode == 'all' or run_mode == 'fusion':

        print('### Fusion ###')


        #input distributions to be fused
        dist1 = np.array([0.6,0.2,0.2])
        dist2 = np.array([0.8,0.1,0.1])

        dists = np.vstack([dist1, dist2])

        #fuse by sampling
        fused_cat, samples_t = sample_t(K, J, dists, alpha)

        print('fused distribution sampling:')
        print(fused_cat)

        #fuse with analytical formula
        fused_cat_analytical = fuse_analytical(K, J, dists, alpha)

        print('fused_distribution analytical:')
        print(fused_cat_analytical)

        

        #compare the fused result to Independent Opinion Pool fusion
        fused_iop = iop(dists)
        print('IOP result:')
        print(fused_iop)


    