import whales as w
import numpy as np
from itertools import chain
import sklearn.decomposition as sk_decomp
import sklearn.linear_model as sk_lm
import factors as f  
from sklearn import metrics
from progressbar import RealProgressBar, withProgress

N = len(w.all_cases) # N = 30000
def flatten_list_of_arrays(l):
    return np.array(list(chain.from_iterable(l)))

def load_for_pca(n, training=True):
    return w.get_log_spectrogram(n, training=training, 
                               spec_opts=w.short_specgram).flatten()

def model(z):
    return 1/(1+np.exp(-z))    

def train_model(m, seed):
    np.random.seed(seed)
    ordering = np.random.permutation(N)

    training_0 = w.all_cases[1:][w.labels[:-1]==0]
    training_1 = w.all_cases[1:][w.labels[:-1]==1]

    print "Loading data for PCA:"
    pca_data, pca_labels = w.load_data_subset(
        cases=w.all_cases[ordering[:10000]], 
        load_function=load_for_pca)

    print "\nCalculating principle components:"
    pca = sk_decomp.RandomizedPCA(
        n_components=m, 
        whiten=True, 
        random_state=1)

    pca.fit(pca_data)

    def load_with_pca(n, training=True):
        data = load_for_pca(n, training=training)
        return pca.transform(data)[:m]

    print "Loading data for training logistic_0:"
    train_0_X, train_0_y = w.load_data_subset(
        cases=training_0,
        load_function=load_with_pca)

    logistic_0 = sk_lm.LogisticRegression(C=1e4)
    logistic_0.fit(train_0_X, train_0_y)

    print "\nLoading data for training logistic_1:"
    train_1_X, train_1_y = w.load_data_subset(
        cases=training_1,
        load_function=load_with_pca,
        training=True)

    logistic_1 = sk_lm.LogisticRegression(C=1e4)
    logistic_1.fit(train_1_X, train_1_y)

    ordering = np.random.permutation(len(w.all_cases))
    p_1 = sum(w.labels[w.all_cases[ordering[:5000]]])/float(5000)
    p_0 = 1-p_1

    return logistic_0, logistic_1, p_0, p_1, load_with_pca

def classify_sequence(cases, logistic_0, logistic_1, p_0, p_1, load_with_pca, training=True):
    result = np.zeros(len(cases))

    m0 = cases[0]

    data = load_with_pca(m0, training=training)

    prior = f.Factor(["w_%d"%(m0-1)], [p_0, p_1])

    l0_p = model(logistic_0.decision_function(data)).flatten()[0]
    l1_p = model(logistic_1.decision_function(data)).flatten()[0]

    psi = f.Factor(["w_%d"%(m0-1), "w_%d"%m0], [[1-l0_p, l0_p],[1-l1_p, l1_p]])

    w_marginal = (psi*prior).marginalise(["w_%d"%(m0-1)])

    result[0] = w_marginal.toarray()[1]

    for i, m in enumerate(withProgress(cases[1:], RealProgressBar())):
        data = load_with_pca(m, training=training)

        l0_p = model(logistic_0.decision_function(data)).flatten()[0]
        l1_p = model(logistic_1.decision_function(data)).flatten()[0]

        psi = f.Factor(["w_%d"%(m-1), "w_%d"%m], [[1-l0_p, l0_p],[1-l1_p, l1_p]])

        w_marginal = (w_marginal*psi).marginalise(["w_%d"%(m-1)])

        result[i+1] = w_marginal.toarray()[1]

    return result

def calculate_performance(cases, results):
    labels = w.labels[cases]

    fpr, tpr, thresholds = metrics.roc_curve(labels.flatten(), results.flatten())
    auc = metrics.auc(fpr, tpr)

    return auc
