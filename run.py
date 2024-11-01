
import argparse
from dataloader import RawDataLoader, LossDataLoader
from model.active_dc_model import ActiveDCModel


parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default="active_dc",choices=["active_dc"],help="choose ours or predlines")
parser.add_argument('--dataset',type=str,default="ACS2018_1yr",choices=["ACS2018_1yr"])
parser.add_argument('--pred_model_class',type=str,default="RandomForestClassifier",choices=["RandomForestClassifier"],help="g(x), L=loss(y,g(x))")
parser.add_argument('--expected_loss_model_class',type=str,default="DecisionTreeRegressor",choices=["DecisionTreeRegressor"],help="f(x), which fits E[L|x] ")
parser.add_argument('--region_model_class',type=str,default="DecisionTreeRegressor",choices=["decision_tree"],help="h(x), which fits (L-f(x))*t")

args = parser.parse_args()
print(args)

'''
prepare data:
1. load raw_data(X,Y,T) # T=1 if from target distribution, T=0 if from source distribution
2. train g(x) on data(X,Y,T=0)
3. get data(X,L,T) using g(x)

run algorithm:
1. initialize region S, test_data
    repeat:
    2. collect new_test_data(X,L,T=1) in S, update test_data with test_data + new_test_data
    3. fit f(x) on training_data + test_data
    4. fit h(x) on training_data + test_data
    5. store old S in S', update S                           
    6. if stopping condition is met, break
7. return S
'''
if __name__ == "__main__":
    model_type = args.model
    dataset_name = args.dataset
    source_state = "CA"
    target_state = "MI"
    raw_data_loader = RawDataLoader(dataset_name=dataset_name, source_state=source_state, target_state=target_state)
    
    pred_model_class = args.pred_model_class
    loss_data_loader = LossDataLoader(raw_data_loader.source_x, 
                                      raw_data_loader.source_y, 
                                      raw_data_loader.target_x, 
                                      raw_data_loader.target_y, 
                                      pred_model_class=pred_model_class)
    
    region_model_class = args.region_model_class
    expected_loss_class = args.expected_loss_model_class
    active_dc_model = ActiveDCModel(loss_data_loader, region_model_class=region_model_class, expected_loss_class=expected_loss_class)
    active_dc_model.train_and_collect()
    
    
    