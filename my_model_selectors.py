import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
   
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores


        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        best_BIC_score = float('inf')
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):    
            try:
            #model = GaussianHMM(n_components=n, n_iter=1000).fit(self.X, self.lengths)
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                p = n*n + 2*n*len(self.X[0]) - 1
                BIC = (-2 * logL) + (p * np.log(n))
                if BIC < best_BIC_score:
                    best_BIC_score = BIC
                    best_model = model
            except:
                pass
        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
             
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        n_components = range(self.min_n_components, self.max_n_components+1)
        best_n = self.random_state
        best_DIC = float('-inf')
        best_model=None
        counter=0
        for n in n_components:
            counter+=1
            try:
                
                model = GaussianHMM(n,  n_iter=1000).fit(self.X, self.lengths)
                original_prob = model.score(self.X,self.lengths)

                sum_prob_others =0.0

                for word in self.words:
                    if word== self.this_word:
                        continue
                    X_other , lengths = self.hwords[word]
                    logL = model.score(X_other, lengths_other)
                    sum_prob_others+=logL
                avg_prob_others = sum_prob_others/counter
                DIC = original_prob - avg_prob_others

                if DIC > bes_DIC:
                    best_DIC= DIC
                    best_n=n
            except:
                pass
        best_model = GaussianHMM(best_n,  n_iter=1000).fit(self.X, self.lengths)
        return best_model
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        bestD , bestN = float('-inf'), 0
        other_words = list ( self.words.keys())
        other_words.remove(self.this_word)

        M= len(self.hwords)

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL= model.score( self.X, self.lengths)
                other_logL= 0
                for word in other_words:
                    X_o , l_o = self.hwords[word]
                    other_logL += model.score(X_o, l_o)
                dic_val = logL - (1/(1-M))*other_logL
                if dic_val > bestD:
                    bestD= dic_val
                    bestN= n
            except:
                pass
            return self.base_model(bestN)
                    
                


                


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        """
        best_score = float('-inf')
        best_model=None
        split_method= KFold(n_splits=min(3,len(self.lengths)))
        model= GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)


        for components in range(self.min_n_components,self.max_n_components+1):
            score=0
            total=0
            try:
                
                for cv_train_idx , cv_test_idx in split_method.split(self.sequences):
                    folds+=1
                    X_test , length_test = combine_sequence(cv_test_idx,self.sequences)
                    X_train , length_train = combine_sequence(cv_train_idx,self.sequences)
                    model = self.base_model(components)
                    score +=model.score(X_test, length_test)

                    total+=1
                    score=score/total
                if score > best_score:
                    best_score = score
                    best_comp=components
            except:
                pass
        return self.base_model(best_comp)
        """
        best_n = 0
        best_model = None
        best_score = float('-inf')
    
        if len(self.sequences) > 1: #Only proceed with CV if there is more than one sequence
            split_method = KFold(n_splits=min(3,len(self.sequences))) 
            # Loop through possible number of states in HMM model
            for n in range(self.min_n_components, self.max_n_components + 1):
                try:
                    split__iter_cnt = 1
                    total_score = 0.0
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # Above just selects indices for the folds - we need to split the data explicitly below
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
            
                        # Train the model on the training data
                        model = GaussianHMM(n_components=n, covariance_type='diag', n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X_train, self.lengths_train)
            
                        # Score the model on the test data
                        logL = model.score(X_test, lengths_test)
            
                        # As we perform this for each cross validation of the data set, we sum the test set log loss
                        total_score += logL
                             
                        # Keep track of number of times we have split the data
                        split_iter_cnt += 1
        
                    # Once all folds are completed, we get an average log loss per fold for that particular model
                    avg_score = total_score / split_iter_cnt
               
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_n = n
                
                except:
                    pass
                             
                if best_n > 0:
                    best_model = GaussianHMM(n_components=best_n, covariance_type='diag', n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
             

        if len(self.sequences) == 1: # Prcoess but not using CV method
            for n in range(self.min_n_components, self.max_n_components+1):
                try:
                    model = GaussianHMM(n_components=n, covariance_type='diag', n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)
                    if logL > best_score:
                        best_score = logL
                        best_model = model
                except:
                    pass
                             
        if best_model is None:
             return self.base_model(self.n_constant)           
        return best_model
    
