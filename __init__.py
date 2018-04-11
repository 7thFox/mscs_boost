r"""This module contains ensemble classifiers and regressors based on 
the SciKitLearn framework.

    - :class:`BootBagRegressor`, for averaging regression models via a 
      Bootstrap Bagging.  This is an example for reference.

    - :class:`BootBagClassifier`, for averaging classification models via a 
      Bootstrap Bagging.  This is an example for reference.

    - :class:`BoostedRegressor`, for boosting any regression method.
      This is to be completed by the student.

    - :class:`BoostedClassifier`, for boosting any classification method
      This is to be completed by the student.


Each class has the required methods for a SciKitLearn-compatible class:
      - __init__()
      - fit(x, y)  Fit the model using X, y as training data.
      - get_params()  Get parameters for this estimator.  Because these are
                      ensemble methods, the `deep` option will also display the
                      details of the underlying methods.
      - predict(x)  Predict output y using input X.
      - score(x, y)  Returns the score of the prediction for testing data X,Y. 
                     For regression, this is FVE.  For classification, it is
                     percent correct.  
 
Copyright
---------
- This file is part of https://github.com/abrahamdavidsmith/mscs_boost/ 
- copyright, 2018 by Abraham D. Smith. 
- CC0 license.  See https://creativecommons.org/choose/zero/
"""

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import scipy.linalg as la
import numpy as np


class BootBagRegressor(BaseEstimator, RegressorMixin):
    r""" This class is for the classic ensemble method known as "bagging" or 
    "bootstrap aggregating" or just "averaging".  When instantiating the
    regressor, the user must specify what underlying method will be used in the
    ensemble.

     Examples
     --------
     >>> import numpy as np
     >>> train_x = np.linspace(0, 4*np.pi, 100).reshape(100,1)
     >>> train_y = np.sin(train_x).reshape(100,)
     >>> test_x = np.linspace(0, 4*np.pi, 49).reshape(49,1) # 49 and 100 are relatively prime
     >>> test_y = np.sin(test_x).reshape(49,)

     >>> from sklearn.linear_model import LinearRegression
     >>> linbag = BootBagRegressor(LinearRegression,
     ...                           num_bootstraps=5,
     ...                           bootstrap_size=20,
     ...                           rand_seed=2018,
     ...                           model_args={'fit_intercept':True})
     >>> linbag.fit(train_x,train_y)
     >>> linbag.score(test_x, test_y)
     0.1348009...
     
     >>> from sklearn.tree import DecisionTreeRegressor
     >>> treebag = BootBagRegressor(DecisionTreeRegressor,
     ...                           num_bootstraps=5,
     ...                           bootstrap_size=20,
     ...                           rand_seed=2018,
     ...                           model_args={'max_depth': 3, 'random_state': 2019})
     >>> treebag.fit(train_x,train_y)
     >>> treebag.score(test_x, test_y)
     0.864821...
    
   """

    def __init__(self, base_model_class, num_bootstraps=None, bootstrap_size=None,
                 rand_seed=None, model_args={}):
       
        self.num_bootstraps = num_bootstraps
        self.bootstrap_size = bootstrap_size

        # we'll build and store the bootstrap samples here, in lists.
        self._bootstrap_subsets = []
        self._bootstrap_models = []

        self.model_args = model_args
        self.base_model_class = base_model_class

        # set up a pseudo-random-number generator with a known seed, for
        # reproducable tests.  "None" will be seeded at random.
        self.random_state = np.random.RandomState(rand_seed)

        # set up the sub-models now, so we can manipulate them before fitting
        for i in range(self.num_bootstraps):
            this_model = self.base_model_class(**self.model_args) 
            self._bootstrap_models.append(this_model)

        pass


    def fit(self, X, Y):
        r""" Fit a model to this training data.  This is the step that 
        runs the actual estimators.  The sub-models are stored in the 
        `_self_bootstrap_models` attribute.

        Parameters
        ----------
        X   - an N-by-p `numpy.ndarray` of training predictors
        Y   - a length N `numpy ndarray` of training responses

        """
        assert len(X.shape) == 2, f"The predictor data X should be 2-dimensional, N-by-p.  Yours is {X.shape}"
        assert len(Y.shape) == 1, f"The response data Y should be 1-dimensional, length N.  Yours is {Y.shape}"
        assert X.shape[0] == Y.shape[0], "Sample sizes of Predictor X and Reponse Y do not match!"


        N = X.shape[0]
        
        for i in range(self.num_bootstraps):
            this_model = self._bootstrap_models[i]
            this_sample = self.random_state.choice(N, self.bootstrap_size) 
            self._bootstrap_subsets.append(this_sample)
            this_X = X[this_sample,:]
            this_Y = Y[this_sample]
            this_model.fit(this_X, this_Y)

        pass 
            
    def get_params(self, deep=False):
        r""" Provide dictionary of fit parameters to the user.

        Parameters 
        ----------
        deep    - (boolean) if True, then return details of each model in the
                  ensemble (default: False)

        Returns
        -------
        a dictionary of parameters

        
        Examples
        --------
        The `deep` option can be used to examine and change arguments to component
        models.

        >>> train_x = np.linspace(0, 4*np.pi, 100).reshape(100,1)
        >>> train_y = np.sin(train_x).reshape(100,)
        >>> test_x = np.linspace(0, 4*np.pi, 49).reshape(49,1) # 49 and 100 are relatively prime
        >>> test_y = np.sin(test_x).reshape(49,)
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> treebag = BootBagRegressor(DecisionTreeRegressor,
        ...                           num_bootstraps=5,
        ...                           bootstrap_size=20,
        ...                           rand_seed=2018,
        ...                           model_args={'max_depth': 3})
        >>> for i, m in enumerate(treebag.get_params(deep=True)['model_list']):
        ...     m.set_params(max_depth=i+1, random_state=2**i)
        DecisionTreeRegressor(criterion='mse', max_depth=1, max_features=None,
                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                   min_impurity_split=None, min_samples_leaf=1,
                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                   presort=False, random_state=1, splitter='best')
        DecisionTreeRegressor(criterion='mse', max_depth=2, max_features=None,
                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                   min_impurity_split=None, min_samples_leaf=1,
                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                   presort=False, random_state=2, splitter='best')
        DecisionTreeRegressor(criterion='mse', max_depth=3, max_features=None,
                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                   min_impurity_split=None, min_samples_leaf=1,
                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                   presort=False, random_state=4, splitter='best')
        DecisionTreeRegressor(criterion='mse', max_depth=4, max_features=None,
                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                   min_impurity_split=None, min_samples_leaf=1,
                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                   presort=False, random_state=8, splitter='best')
        DecisionTreeRegressor(criterion='mse', max_depth=5, max_features=None,
                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                   min_impurity_split=None, min_samples_leaf=1,
                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                   presort=False, random_state=16, splitter='best')
        >>> treebag.fit(train_x,train_y)
        >>> treebag.score(test_x, test_y)
        0.791667...
        """

        param_dict = { 'num_bootstraps': self.num_bootstraps,
                       'bootstrap_size': self.bootstrap_size,
                       'base_model_class': self.base_model_class,
                       'model_args': self.model_args,
                       'random_state': self.random_state }

        if deep:
            param_dict['model_list'] = self._bootstrap_models
            param_dict['subset_list'] = self._bootstrap_subsets
            param_dict['model_arg_list'] = []
            for i in range(self.num_bootstraps):
                m = self._bootstrap_models[i]
                param_dict['model_arg_list'].append( m.get_params() )

        return param_dict

    def predict(self, x):
        r""" Apply the ensemble model to the testing data x. 
        
        Parameters
        ----------
        x   - an N-by-p `numpy.ndarray` of testing predictors
        
        Returns
        -------
        y  - an N-by-1 `numpy.ndarray` of responses.
        """ 
        
        # evaluate each model, and average them.
        sub_answers = [ m.predict(x) for m in self._bootstrap_models ]
        sub_answers = np.array(sub_answers) 
        sub_answers.shape = (self.num_bootstraps, x.shape[0],)
        output = sub_answers.mean(axis=0)
        output.shape = (x.shape[0],)

        return output


    def residual(self, x, y):
        r""" Compute the residual vector for testing data x,y 
       
        Parameters
        ----------
         - x   an N-by-p `numpy.ndarray` of testing predictors
         - y   a length-N `numpy.ndarray` of testing responses.
        """
        yhat = self.predict(x)
        yhat.shape = y.shape
        return y - yhat

    def score(self, x, y):
        r""" Compute fraction-of-variance-explained (FVE or R^2) of the model
        on this testing data.  
        
        Notes
        -----
        If the data was not normalized, then the variance of y might
        be misunderstood.   If you want to fix this, then play with the
        normalization methods.
       
        Parameters
        ----------
         - x   an N-by-p `numpy.ndarray` of testing predictors
         - y   a length-N `numpy.ndarray` of testing responses.
        """
        r = self.residual(x,y)
        RR = (r.T.dot(r)).flatten()
        YY = (y.T.dot(y)).flatten()
        assert RR.shape == (1,) and YY.shape == (1,), "Shapes/dimensions of responses are mixed up."

        return 1. - RR[0]/YY[0]


class BootBagClassifier(BaseEstimator, ClassifierMixin):
    r""" This class is for the classic ensemble method known as "bagging" or 
     "bootstrap aggregating" or just "averaging".  When instantiating the
     classifier, the user must specify what underlying method will be used in the
     ensemble.

     Examples
     --------
     >>> import numpy as np
     >>> train_x = np.linspace(0.1,10000,10000).reshape(10000,1)
     >>> train_y = np.sign(np.sin(np.sqrt(train_x))).reshape(10000,)
     >>> test_x = np.array( [ (np.pi*0.5)**2, (np.pi*1.5)**2, (np.pi*2.5)**2, (np.pi*3.5)**2 ]).reshape(4,1)
     >>> test_y = np.array( [ 1, -1, 1, -1], dtype='int')
     >>> from sklearn.neighbors import KNeighborsClassifier
     >>> knnbag = BootBagClassifier(KNeighborsClassifier,
     ...                           num_bootstraps=5,
     ...                           bootstrap_size=20,
     ...                           rand_seed=2018,
     ...                           model_args={'n_neighbors': 3})
     >>> knnbag.fit(train_x,train_y)
     >>> knnbag.score(test_x, test_y)
     0.5

     >>> from sklearn.tree import DecisionTreeClassifier
     >>> treebag = BootBagClassifier(DecisionTreeClassifier,
     ...                           num_bootstraps=5,
     ...                           bootstrap_size=20,
     ...                           rand_seed=2018,
     ...                           model_args={'max_depth': 3, 'random_state': 1})
     >>> treebag.fit(train_x,train_y)
     >>> treebag.score(test_x, test_y)
     0.5

   """

    def __init__(self, base_model_class, num_bootstraps=None, bootstrap_size=None,
                 rand_seed=None, model_args={}):
       
        self.num_bootstraps = num_bootstraps
        self.bootstrap_size = bootstrap_size

        # we'll build and store the bootstrap samples here, in lists.
        self._bootstrap_subsets = []
        self._bootstrap_models = []

        self.model_args = model_args
        self.base_model_class = base_model_class

        # set up a pseudo-random-number generator with a known seed, for
        # reproducable tests.  "None" will be seeded at random.
        self.random_state = np.random.RandomState(rand_seed)

        # set up the sub-models now, so we can manipulate them before fitting
        for i in range(self.num_bootstraps):
            this_model = self.base_model_class(**self.model_args) 
            self._bootstrap_models.append(this_model)

        pass


    def fit(self, X, Y):
        r""" Fit a model to this training data.  This is the step that 
        runs the actual estimators.  The sub-models are stored in the 
        `_self_bootstrap_models` attribute.

        Parameters
        ----------
        X   - an N-by-p `numpy.ndarray` of training predictors
        Y   - an N-by-1 `numpy ndarray` of training responses

        """
        assert len(X.shape) == 2, f"The predictor data X should be 2-dimensional, N-by-p.  Yours is {X.shape}"
        assert len(Y.shape) == 1, f"The response data Y should be 1-dimensional, length N.  Yours is {Y.shape}"
        assert X.shape[0] == Y.shape[0], "Sample sizes of Predictor X and Reponse Y do not match!"


        N = X.shape[0]
        
        for i in range(self.num_bootstraps):
            this_model = self._bootstrap_models[i]
            this_sample = self.random_state.choice(N, self.bootstrap_size) 
            self._bootstrap_subsets.append(this_sample)
            this_X = X[this_sample,:]
            this_Y = Y[this_sample]
            this_model.fit(this_X, this_Y)
            

        pass 
            
    def get_params(self, deep=False):
        r""" Provide dictionary of fit parameters to the user.

        Parameters 
        ----------
        deep    - (boolean) if True, then return details of each model in the
                  ensemble (default: False)
                  As in the BootBagRegressor, the `deep` option can be used to examine
                  and change arguments to component models.

        Returns
        -------
        a dictionary of parameters

        

        """

        param_dict = { 'num_bootstraps': self.num_bootstraps,
                       'bootstrap_size': self.bootstrap_size,
                       'base_model_class': self.base_model_class,
                       'model_args': self.model_args,
                       'random_state': self.random_state }

        if deep:
            param_dict['model_list'] = self._bootstrap_models
            param_dict['subset_list'] = self._bootstrap_subsets
            param_dict['model_arg_list'] = []
            for i in range(self.num_bootstraps):
                m = self._bootstrap_models[i]
                param_dict['model_arg_list'].append( m.get_params() )

        return param_dict

    def predict(self, x):
        r""" Apply the ensemble model to the testing data x. 
        
        Parameters
        ----------
        x   - an N-by-p `numpy.ndarray` of testing predictors
        
        Returns
        -------
        y  - an N-by-1 `numpy.ndarray` of responses.
        """ 
        
        # evaluate each model, and average them.
        sub_answers = [ m.predict(x) for m in self._bootstrap_models ]
        sub_answers = np.array(sub_answers) 
        sub_answers.shape = (self.num_bootstraps, x.shape[0])
        
        # count and vote.
        winners = []
        for i in range(x.shape[0]):
            classes, votes = np.unique(sub_answers[:, i], return_counts=True)
            winner_i = classes[np.argmax(votes)]
            winners.append(winner_i)

        output = np.array(winners)

        return output


    def score(self, x, y):
        r""" Compute the fraction of matches.
        
        Parameters
        ----------
         - x   an N-by-p `numpy.ndarray` of testing predictors
         - y   an N-by-1 `numpy.ndarray` of testing responses.
        """
        return np.count_nonzero(self.predict(x) == y)*1.0/x.shape[0]


class BoostedRegressor(BootBagRegressor):
    r""" Dear Student --- Implement a BOOST algrithm based on 8.2 on page 323
    of ISL.  

    Note that the one in the book is described for TREEs only, but you can
    generalize step (a) to be "fit a ___ model with ___ model_args" to use any
    model you wish, such as kNN or tree or linear or any other regression.

    The slow-learning parameter called "lambda" in the book is called
    'shrinkage' here.

    You don't need to re-write the residual() or score() commands, as those are
    inherited.
    """
    
    def __init__(self, base_model_class, shrinkage=0.01, num_bootstraps=None, bootstrap_size=None,
                 rand_seed=None, model_args={}):
        raise NotImplementedError("Dear Student -- please write this!")

    def fit(self, X, Y):
        raise NotImplementedError("Dear Student -- please write this!")
        
    def predict(self, x):
        raise NotImplementedError("Dear Student -- please write this!")
 
    def get_params(self, deep=False):
        raise NotImplementedError("Dear Student -- please write this! (just add self.shrinkage to the dictionary.)")

class BoostedClassifier(BootBagClassifier):
    r""" Dear Student --- *Invent* a BOOST algorithm based on 8.2 on page 323
    of ISL.   That algorithm is for regression, but here we want
    classification.  Here's the idea: Step (1) instead of building new models
    using the RESIDUAL, build new models that the training points that are
    mis-classified so far.  Step (2) instead of adding f += lambda*f_i, think
    about adding partial votes that are worth less than before.  So, if the
    classes (Red, Blue, Green) have (200,200,100) votes so far, and a new model
    makes (30,10,10) "residual" votes with lambda=0.05, then the new vote will
    be (201.5, 200.5, 100.5), and Red wins by a hair.

    Note that the algorithm in the book is described for TREEs only, but you can
    generalize step (a) to be "fit a ___ model with ___ model_args" to use any
    model you wish, such as kNN or trees or SVM or any other classification.

    The slow-learning parameter called "lambda" in the book is called
    'shrinkage' here.

    You don't need to re-write the residual() or score() commands, as those are
    inherited.
    """
    
    def __init__(self, base_model_class, shrinkage=0.01, num_bootstraps=None, bootstrap_size=None,
                 rand_seed=None, model_args={}):
        raise NotImplementedError("Dear Student -- please write this!")

    def fit(self, X, Y):
        raise NotImplementedError("Dear Student -- please write this!")
        
    def predict(self, x):
        raise NotImplementedError("Dear Student -- please write this!")
 
    def get_params(self, deep=False):
        raise NotImplementedError("Dear Student -- please write this! (just add self.shrinkage to the dictionary.)")


