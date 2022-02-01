#This script contains the functionality of the Correlated Fusion Model, which is a Bayesian model
#for the fusion of correlated classifiers that output categorical distributions.
#For simplicity, here we only show it for the fusion of 2 classifiers.


from __future__ import division
import pyjags
import numpy as np

np.set_printoptions(precision=3)



#JAGS definition of the generative model of the Correlated Fusion Model
jags_model_code = '''
model {

    for (i in 1:I) {

        t[i] ~ dcat(p)

        #choose the respective parameters according to the true label t[i] of example i
        d[i,1:J] <- delta[t[i],1:J]
        alpha1[i,1:J] <- alpha[1,t[i],1:J] - d[i,1:J]
        alpha2[i,1:J] <- alpha[2,t[i],1:J] - d[i,1:J]
        
        #generate gamma variates A1,A2,D
        for (j in 1:J) {
            A1[i,j] ~ dgamma(alpha1[i,j], 1)
            A2[i,j] ~ dgamma(alpha2[i,j], 1)
            D[i,j] ~ dgamma(d[i,j], 1)
            
        }

        #generate x1,x2 from A1,A2,D
        for (j in 1:J) {
            x1[i,j] <- (A1[i,j] + D[i,j]) / (sum(A1[i,1:J]) + sum(D[i,1:J]))
            x2[i,j] <- (A2[i,j] + D[i,j]) / (sum(A2[i,1:J]) + sum(D[i,1:J]))
        }

        #Gaussian trick to enable observing deterministic variables when inferring delta and t
        for (j in 1:J) {
            x1_star[i,j] ~ dnorm(x1[i,j], precision)
            x2_star[i,j] ~ dnorm(x2[i,j], precision)
        }

    }

    #uninformed prior for delta
    for (j in 1:J) {
        for (l in 1:J) {
            delta[j,l] ~ dgamma(0.001,0.001)
        }
    }

    #define parameter p of uninformed prior of the true class labels
    for (j in 1:J) {
        p[j] <- 1/J
    } 

}
'''
        


#data generation
#sample simulated classifier outputs x1,x2 while observing parameters alpha and delta and the true class labels
#J: number of classes
#alpha: marginal parameters of modeling correlated Dirichlet distributions, shape (2,J,J)
#delta: correlation parameters of modeling correlated Dirichlet distribution, shape (J,J)
#true_labels: list of true class labels of the classifier outputs to be generated
def sample_x(J, alpha, delta, true_labels):
    I = len(true_labels) #nr of examples
    model = pyjags.Model(jags_model_code, data = dict(J = J,
                                                alpha = alpha,
                                                delta = delta,
                                                t = true_labels,
                                                I = I,
                                                precision = 100000), chains = 1, adapt = 1000)
    samples = model.sample(1, vars = ['x1', 'x2']) #we just draw one sample containing I examples
    values_x1 = samples['x1'].squeeze()
    values_x2 = samples['x2'].squeeze()
    return values_x1, values_x2



#parameter inference
#sample parameters delta while observing I classifier outputs x1,x2, their true labels and
#marginal parameters alpha
#J: number of classes
#x1: observed categorical classifier outputs of classifier 1, shape (I, J)
#x2: observed categorical classifier outputs of classifier 2, shape (I, J)
#true_labels: list of true class labels of the observed classifier outputs
#alpha: marginal parameters of modeling correlated Dirichlet distribution, assumed to be known here, shape (2,J,J)
#nr_samples: number of samples we draw for inferring the parameters
#nr_adapt: number of samples used as burn-in
#delta_init: initial value for the delta parameters
#returns mean of sampled deltas and raw samples
def sample_delta(J, x1, x2, true_labels, alpha, nr_samples = 40000, nr_adapt = 20000, delta_init = 0.5):
    I = len(true_labels) #nr of examples
    model = pyjags.Model(jags_model_code, data = dict(J = J,
                                                I = I,
                                                x1_star = x1,
                                                x2_star = x2,
                                                t = true_labels,
                                                alpha = alpha,
                                                precision = 10000),
                                                init = dict(delta = np.ones((J,J)) * delta_init),
                                                chains = 1, adapt = nr_adapt)
    samples = model.sample(nr_samples, vars = ['delta'])
    samples = samples['delta'].squeeze()

    mean_delta = np.mean(samples, axis = 2)

    #return mean of deltas and raw samples
    return mean_delta, samples


#fusion
#sample true labels t from observed classifier output distributions x1, x2 and given
#model parameters alpha and delta for fusion of x1 and x2
#J: number of classes
#x1: categorical output distribution of classifier 1, shape (J,)
#x2: categorical output distribution of classifier 2, shape (J,)
#alpha: marginal parameters of the modeling correlated Dirichlet distributions, shape (2,J,J)
#delta: correlation parameters of the modeling correlated Dirichlet distributions, shape (J,J)
#nr_samples: number of samples we draw for inferring t per chain
#nr_adapt: number of samples used as burn-in per chain
#nr_chains: number of parallel chains that are sampled
#           to speed up fusion, increase according to available hardware while lowering nr_samples and nr_adapt
#           nr_samples * nr_chains is the total number of samples drawn
#           nr_adapt * nr_chains is the total number of burn-in samples drawn
#progress_bar: Boolean that determines if the JAGS progressbar is shown
#returns the fused categorical distribution computed from the samples and the respective raw samples
def sample_t(J, x1, x2, alpha, delta, nr_samples = 500000, nr_adapt = 500000, nr_chains = 7, progress_bar = True):
    I = 1 #we only cosider one example here
    x1 = x1.reshape((1,J))
    x2 = x2.reshape((1,J))
    model = pyjags.Model(jags_model_code, data = dict(J = J,
                                                x1_star = x1,
                                                x2_star = x2,
                                                alpha = alpha,
                                                delta = delta,
                                                I = I,
                                                precision = 100000),
                                                chains = nr_chains, threads = nr_chains, chains_per_thread = 1,
                                                adapt = nr_adapt, progress_bar = progress_bar)
    samples = model.sample(nr_samples, vars = ['t'])
    samples = samples['t'].squeeze()


    #compute fused distribution from samples of t

    #if only one chain was sampled, reshape samples array to (nr_samples, 1):
    if len(samples.shape) == 1:
        samples = samples.reshape((samples.shape[0], 1))

    #store the fused categoricals for each chain
    fused_cats = np.zeros((nr_chains, J))

    for i in range(nr_chains):
        #compute a categorical distribution from the sampled true labels t
        unique, counts = np.unique(samples[:,i], return_counts=True)
        count_dict = dict(zip(unique, counts))
        label_counts = np.zeros(J)
        for j in range(1, J + 1): #compensate that JAGS works 1-based
            if j in count_dict.keys():
                label_counts[j-1] = count_dict[j]
        fused_cats[i] = label_counts / nr_samples

    #optional: print fused categoricals of each chain for checking convergence
    #print('fused dists of each chain separately:')
    #print(fused_cats)

    #resulting fused distribution is the mean of the fused distributions of each chain
    fused_cat = np.mean(fused_cats, axis = 0)

    #return fused distribution and respective samples of t
    return fused_cat, samples




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


    from utils import create_true_labels, iop
    import independent_fusion_model



    ######################
    #set model parameters#


    J = 3 #nr of classes

    #create simulated true labels
    nr_examples_per_class = 1000
    true_labels = create_true_labels(J, nr_examples_per_class)

    #marginal parameters of modeling correlated Dirichlet distribution
    alpha = np.array([[[12,8,8], [8,12,8], [8,8,12]],       #marginal parameters of classifier 1 for t_i = 1, t_i = 2, t_i = 3
                        [[12,8,8], [8,12,8], [8,8,12]]])    #marginal parameters of classifier 2 for t_i = 1, t_i = 2, t_i = 3

    #correlation parameters of modeling correlated Dirichlet distribution for t_i = 1, t_i = 2, t_i = 3
    delta = np.array([[6,4,4], [4,6,4], [4,4,6]])  #generates a correlation of approx 0.5
    #delta = np.array([[0.1,0.1,0.1], [0.1,0.1,0.1], [0.1,0.1,0.1]])  #generates a correlation of approx 0
    #delta = np.array([[11.9,7.9,7.9], [7.9,11.9,7.9], [7.9,7.9,11.9]])  #generates a correlation of approx 1
    #note that very small (<0.1) and very large (>alpha-0.1) values for delta can lead to problems with sampling when fusing



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


        samples_x1, samples_x2 = sample_x(J, alpha, delta, true_labels)


        #plot generated classifier outputs

        fig,ax = plt.subplots(J,J, sharex=True, sharey=True)
        fig.set_size_inches(10,10)
        fig.suptitle(r'Generated Classifier Outputs $\bm{x_i^1}$ and $\bm{x_i^2}$ of 2 Correlated Classifiers'
                        '\n' '- the categorical distributions are plotted columnwise per dimension -'
                        '\n' r'- the rows show the classifier models for different true class labels $t_i$ -',
                        fontsize = 18)


        for j in range(J):
            for l in range(J):
                ax[l,j].scatter(samples_x1[l*nr_examples_per_class:(l+1)*nr_examples_per_class, j],
                                samples_x2[l*nr_examples_per_class:(l+1)*nr_examples_per_class, j],
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


        #print correlations
        print('correlations per true label and dimension:')
        for i in range(J):
            print('true label = %d:' %(i+1))
            cors = np.zeros(J)
            for j in range(J):
                cors[j] = np.corrcoef(samples_x1[i*nr_examples_per_class:(i+1)*nr_examples_per_class,j],
                                        samples_x2[i*nr_examples_per_class:(i+1)*nr_examples_per_class,j])[0,1]
            print(cors)


        plt.show()



    ##################################
    #infer parameters alpha and delta#

    if run_mode == 'all' or run_mode == 'inference':

        print('### Parameter Inference ###')


        #infer alpha with Independent Fusion Model


        #reshape generated data to training data for independent fusion model
        train_data = np.swapaxes(
                            np.vstack((np.reshape(samples_x1, (1,samples_x1.shape[0], samples_x1.shape[1])),
                                        np.reshape(samples_x2, (1,samples_x2.shape[0], samples_x2.shape[1])))),
                            0,1)

        alpha_inf, samples_alpha = independent_fusion_model.sample_alpha(2, J, train_data, true_labels)

        print('inferred alphas:')
        print(alpha_inf)


        #infer delta assuming inferred alpha to be observed

        x1 = samples_x1
        x2 = samples_x2

        delta_inf, samples_delta = sample_delta(J, x1, x2, true_labels, alpha_inf, nr_samples = 40000, nr_adapt = 20000)

        print('inferred deltas:')
        print(delta_inf)

    

    ########
    #fusion#

    if run_mode == 'all' or run_mode == 'fusion':

        print('### Fusion ###')


        #input distributions to be fused
        dist1 = np.array([0.6,0.2,0.2])
        dist2 = np.array([0.6,0.2,0.2])

        #fuse
        fused_cat, samples_t = sample_t(J, dist1, dist2, alpha, delta)

        print('fused distribution:')
        print(fused_cat)



        #compare the fused result to Independent Fusion Model fusion
        dists = np.vstack([dist1, dist2])
        fused_indf, _ = independent_fusion_model.sample_t(2, J, dists, alpha)

        print('Independent Fusion Model result:')
        print(fused_indf)


        #compare the fused result to Independent Opinion Pool fusion
        fused_iop = iop(dists)
        print('IOP result:')
        print(fused_iop)