
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from .helpers import *
class ActiveDCModel:
    def __init__(self, loss_data_loader, region_model_class:str="DecisionTreeRegressor", expected_loss_class:str="RandomForestRegressor", \
            nums_each_iter:int=100, beta:float=0.25, iters:int=10):
        self.loss_data_loader = loss_data_loader
        self.region_model_class = region_model_class
        self.expected_loss_class = expected_loss_class
        self.nums_each_iter = nums_each_iter
        self.beta = beta
        self.iters = iters
        print("initializing ActiveDCModel...")
        # print params
        print("region model class is ",region_model_class)
        print("expected loss model class is ",expected_loss_class)
        print("nums_each_iter is ",nums_each_iter)
        print("beta is ",beta)
        # initialize region model and cutoff
        self.region_model = None    
        
        
    
    def collect_data(self):
        new_target_x, new_target_y = \
            self.loss_data_loader.collect_new_target_data(self.is_in_region_S, self.nums_each_iter)    
        self.loss_data_loader.merge_new_target_data(new_target_x, new_target_y)
    
    def fit_expected_loss_model(self):
        # fit f(x) = E[L|x]
        all_x, all_l, all_t = self.loss_data_loader.get_x_l_t_pairs()
        train_x, valid_x, train_l, valid_l = \
            train_test_split(all_x, all_l, test_size=0.25, random_state=42)
        
        if self.expected_loss_class == "RidgeRegression":
            self.expected_loss_model = train_linear_reg(train_x, train_l, valid_x, valid_l)
        elif self.expected_loss_class == "DecisionTreeRegressor":
            self.expected_loss_model = train_decision_tree_reg(train_x, train_l, valid_x, valid_l)
        elif self.expected_loss_class == "RandomForestRegressor":
            self.expected_loss_model = train_random_forest_reg(train_x, train_l, valid_x, valid_l)
        else:
            raise ValueError(f"Excepted loss model class {self.expected_loss_class} not supported.")
        
    def fit_region_model(self):
        # fit h(x) = (L-f(x))*t
        all_x, all_l, all_t = self.loss_data_loader.get_x_l_t_pairs()
        # b = (L-f(x))*t
        all_b = (all_l - self.expected_loss_model.predict(all_x)) * all_t
        all_x_train, all_x_valid, all_b_train, all_b_valid = \
            train_test_split(all_x, all_b, test_size=0.25, random_state=42)

        if self.region_model_class == "DecisionTreeRegressor":
            self.region_model = train_decision_tree_reg(all_x_train, all_b_train, all_x_valid, all_b_valid)
        elif self.region_model_class == "RandomForestRegressor":
            self.region_model = train_random_forest_reg(all_x_train, all_b_train, all_x_valid, all_b_valid)
        elif self.region_model_class == "RidgeRegression":
            self.region_model = train_linear_reg(all_x_train, all_b_train, all_x_valid, all_b_valid)
        else:
            raise ValueError(f"Region model class {self.region_model_class} not supported.")
    
    def identify_region(self):
        print("identifying region...")
        all_x, all_l, all_t = self.loss_data_loader.get_x_l_t_pairs()
        all_scores = self.region_model.predict(all_x)
        self.cutoff = np.quantile(all_scores, 1-self.beta)
        
    def is_in_region_S(self,x):
        if self.region_model is None:
            return True
        score = self.region_model.predict(x.reshape(1,-1))
        # print("score shape is:",score.shape)
        score = score[0]
        if score > self.cutoff:
            return True
        else:
            return False
    
    def plot_region_model(self, filename):
        all_x, all_l, all_t = self.loss_data_loader.get_x_l_t_pairs()
        # all_scores = self.expected_loss_model.predict(all_x)
        labels = (all_l > self.cutoff).astype(int)

        # 使用分类标签训练分类模型
        self.region_classifier = DecisionTreeClassifier(max_depth=5)
        self.region_classifier.fit(all_x, labels)
        plt.figure(figsize=(60, 30))
        tree.plot_tree(self.region_classifier, filled=True)
        plt.savefig(filename)
        
        
    def train_and_collect(self):
        for i in range(self.iters):
            self.collect_data()
            self.fit_expected_loss_model()
            self.fit_region_model()
            self.identify_region()
            self.plot_region_model("plot/tree" + f"b{self.beta}_n{self.nums_each_iter}_i{i}.png")
            # exit(1)
        return self.region_model, self.cutoff    
    
    def finetune(self):
        model_class = self.loss_data_loader.pred_model_class
        all_x = np.concatenate([self.loss_data_loader.source_train_x, self.loss_data_loader.collected_target_x], axis=0)
        all_y = np.concatenate([self.loss_data_loader.source_train_y, self.loss_data_loader.collected_target_y], axis=0)
        train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.25, random_state=42)
        print("finetuning prediction model...")
        if model_class == "RidgeRegression":
            self.pred_model = train_linear_reg(train_x, train_y, valid_x, valid_y)
        elif model_class == "DecisionTreeRegressor":
            self.pred_model = train_decision_tree_reg(train_x, train_y, valid_x, valid_y)
        elif model_class == "RandomForestRegressor":
            self.pred_model = train_random_forest_reg(train_x, train_y, valid_x, valid_y)
        elif model_class == "LogisticRegression":
            self.pred_model = train_logreg(train_x, train_y, valid_x, valid_y)
        elif model_class == "RandomForestClassifier":
            self.pred_model = train_random_forest(train_x, train_y, valid_x, valid_y)
        elif model_class == "DecisionTreeClassifier":
            self.pred_model = train_decision_tree(train_x, train_y, valid_x, valid_y)
        else:
            raise ValueError(f"Pred Model class {model_class} not supported.")
        return self.pred_model
        
         
        