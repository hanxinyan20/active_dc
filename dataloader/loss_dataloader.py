import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
import sys
sys.path.append("../..")
from model.helpers import train_logreg, train_linear_reg, train_decision_tree_reg, train_random_forest_reg, train_decision_tree, train_random_forest
class LossDataLoader:
    def __init__(self, source_x, source_y, target_x, target_y, source_train_size:float=0.5, source_valid_in_train_size:float=0.25, pred_model_class:str="RandomForestClassifier"):
        
        # split source data into training validation and test sets;
        # training + valid is used to fit ourtcome model f(x); after fitting, these data should never be used again
        # source_test is used to identify region
        print("initializing LossDataLoader...")
        print("prediction model class is ",pred_model_class)
        self.source_x = source_x
        self.source_y = source_y
        self.source_train_x, self.source_region_x, self.source_train_y, self.source_region_y = \
            train_test_split(self.source_x, self.source_y, test_size=source_train_size, random_state=42)
        train_x, valid_x, train_y, valid_y = \
            train_test_split(self.source_train_x, self.source_train_y, test_size=source_valid_in_train_size, random_state=42)
            
        self.latent_target_x = target_x
        self.latent_target_y = target_y
        # initialize collected target data as empty set
        self.collected_target_x = np.array([]).reshape(0, target_x.shape[1])
        self.collected_target_y = np.array([])
        
        print("collect_target_x",self.collected_target_x,self.collected_target_x.shape)
        
        # train prediction model
        # three types of regression models: RidgeRegression, DecisionTreeRegressor, RandomForestRegressor
        # three types of classification models: LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
        assert pred_model_class in \
            ["RidgeRegression", "DecisionTreeRegressor", "RandomForestRegressor", \
                "LogisticRegression",  "RandomForestClassifier", "DecisionTreeClassifier"], \
                    "Model class not supported."
        self.pred_model_class = pred_model_class
        print("training prediction model...")
        if self.pred_model_class == "RidgeRegression":
            self.pred_model = train_linear_reg(train_x, train_y, valid_x, valid_y)
        elif self.pred_model_class == "DecisionTreeRegressor":
            self.pred_model = train_decision_tree_reg(train_x, train_y, valid_x, valid_y)
        elif self.pred_model_class == "RandomForestRegressor":
            self.pred_model = train_random_forest_reg(train_x, train_y, valid_x, valid_y)
        elif self.pred_model_class == "LogisticRegression":
            self.pred_model = train_logreg(train_x, train_y, valid_x, valid_y)
        elif self.pred_model_class == "RandomForestClassifier":
            self.pred_model = train_random_forest(train_x, train_y, valid_x, valid_y)
        elif self.pred_model_class == "DecisionTreeClassifier":
            self.pred_model = train_decision_tree(train_x, train_y, valid_x, valid_y)    
        else:
            raise ValueError(f"Pred Model class {self.pred_model_class} not supported.")
        print("finish training prediction model")

        if self.pred_model_class in ["LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier"]:
            self.loss_func = log_loss
            print("loss function is log_loss")
        else:
            self.loss_func = mean_squared_error
            print("loss function is mean_squared_error")

        # calculate loss for source data
        print("calculating loss for source data...")
        if self.pred_model_class in ["LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier"]:
            pred_source_region_y = self.pred_model.predict_proba(self.source_region_x)
            self.source_region_loss = np.array([log_loss([y_true], [y_pred], labels=[False,True]) for y_true, y_pred in zip(self.source_region_y, pred_source_region_y)])
        else:
            pred_source_region_y = self.pred_model.predict(self.source_region_x)
            self.source_region_loss = np.array([mean_squared_error([y_true], [y_pred]) for y_true, y_pred in zip(self.source_region_y, pred_source_region_y)])
        # transform self.source_test_y to binary
        

        
        print("finish initializing LossDataLoader")
        
    def merge_new_target_data(self, new_target_x, new_target_y):
        self.collected_target_x = np.concatenate([self.collected_target_x, new_target_x], axis=0)
        self.collected_target_y = np.concatenate([self.collected_target_y, new_target_y], axis=0)
        if self.pred_model_class in ["LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier"]:
            pred_target_y = self.pred_model.predict_proba(self.collected_target_x)
        else:
            pred_target_y = self.pred_model.predict(self.collected_target_x)
        self.collected_target_loss = np.array([self.loss_func([y_true], [y_pred],labels=[False,True]) for y_true, y_pred in zip(self.collected_target_y, pred_target_y)])
    
    def collect_new_target_data(self, in_region, nums:int):
        # in_region is a function that returns a boolean mask, indicating whether the data point is in the region
        # nums is the number of new data points to collect
        print(self.latent_target_x.shape)
        indices_to_collect = [i for i, row in enumerate(self.latent_target_x) if in_region(row)]

        indices_to_collect = np.random.choice(indices_to_collect, nums, replace=False)
        # print("indices_to_collect",indices_to_collect)
        new_target_x = self.latent_target_x[indices_to_collect]
        # print("new_target_x",new_target_x.shape)
        new_target_y = self.latent_target_y[indices_to_collect]
        # TODO: can remove collected data from latent target data 
        # self.latent_target_x = np.delete(self.latent_target_x, indices_to_collect, axis=0)
        # self.latent_target_y = np.delete(self.latent_target_y, indices_to_collect, axis=0)
        
        return new_target_x, new_target_y
    
    def get_x_y_pairs(self):
        all_x = np.concatenate([self.source_region_x, self.collected_target_x])
        all_y = np.concatenate([self.source_region_y, self.collected_target_y])
        return all_x, all_y

    def get_x_l_t_pairs(self):
        all_x = np.concatenate([self.source_region_x, self.collected_target_x])
        all_l = np.concatenate([self.source_region_loss, self.collected_target_loss])
        all_t = np.concatenate([np.zeros(len(self.source_region_x)), np.ones(len(self.collected_target_x))])
        return all_x, all_l, all_t
                                            