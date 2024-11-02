from bayes_opt import BayesianOptimization,UtilityFunction
from .active_dc_model import ActiveDCModel
import numpy as np
class BayesianADCModel(ActiveDCModel):
    def __init__(self,loss_data_loader,expected_loss_class:str="RandomForestRegressor",\
        nums_each_iter:int=100,beta:float=0.25, iters:int=5,kappa=0.1,xi=0.0):
        
        self.loss_data_loader = loss_data_loader
        self.expected_loss_class = expected_loss_class
        self.nums_each_iter = nums_each_iter
        self.beta = beta
        self.iters = iters
        self.kappa = kappa
        self.xi = xi
        self.utility = UtilityFunction(kind="ucb", kappa=self.kappa, xi=self.xi)
        print("initializing BayesianADCModel...")
        self.region_classifier = None
        self.optimizer = None
        
    def fit_region_model(self):
        # fit h(x)
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds=self.loss_data_loader.get_bounds(),
            verbose=2,
            random_state=1,
            allow_duplicate_points=True
        )
       
        all_x, all_l, all_t = self.loss_data_loader.get_x_l_t_pairs()

        all_b = (all_l - self.expected_loss_model.predict(all_x)) * all_t

        idx = np.random.permutation(len(all_x))
        all_x = all_x[idx]
        all_l = all_l[idx]
        all_t = all_t[idx]
        all_b = all_b[idx]
        
        for i in range(len(all_x)):
            self.optimizer.register(params=all_x[i], target=all_b[i])
        
        self.optimizer.fit_gp()

        all_ucb = self.optimizer.get_ucb(all_x, self.utility)
        print(all_ucb)
        # sort ucb
        all_ucb = np.sort(all_ucb)
        print("all ucb after sorted", all_ucb)
        self.cutoff = np.quantile(all_ucb, 1-self.beta)
        print("cutoff",self.cutoff) 
        
        
    def is_in_region_S(self,x):
        '''
        check if x is in region S:
        compare h(x)+βα(x) with cutoff
        '''
        if self.optimizer is None:
            return True
        return self.optimizer.get_ucb(x.reshape(1,-1), self.utility)[0] >= self.cutoff
    def train_and_collect(self):
        for i in range(self.iters):
            self.collect_data()
            self.fit_expected_loss_model()
            self.fit_region_model()
            filename = f"bayesian_adc_model_{i}.png"
            self.plot_region_model(filename)
            
        
        
            
        