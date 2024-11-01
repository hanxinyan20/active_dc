from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
import numpy as np
# data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
# ca_data = data_source.get_data(states=["CA"], download=False)
# mi_data = data_source.get_data(states=["MI"], download=False)
# ca_features, ca_labels, _ = ACSIncome.df_to_numpy(ca_data)
# mi_features, mi_labels, _ = ACSIncome.df_to_numpy(mi_data)

# model = LogisticRegression(max_iter=1000)
# # Train on CA data
# model.fit(ca_features, ca_labels)
# print("finish training")
# # Test on MI data
# score = model.score(mi_features, mi_labels)
# print(f"Model accuracy: {score}")

class RawDataLoader:
    def __init__(self, dataset_name:str = "ACS2018_1yr", **kwargs):
        self.dataset = dataset_name
        if self.dataset == "ACS2018_1yr":
            try:
                data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
                self.source_data_df = data_source.get_data(states=[kwargs.get("source_state", [])], download=False)
                self.target_data_df = data_source.get_data(states=[kwargs.get("target_state", [])], download=False)
                self.source_x, self.source_y,_ = ACSIncome.df_to_numpy(self.source_data_df)
                # source只保留10000条数据
                self.source_x = self.source_x[:10000]
                self.source_y = self.source_y[:10000]
                
                # print("self.source_x.shape",self.source_x.shape)
                # print("self.source_y.shape",self.source_y.shape)
                # source_y is now a 1d array, each element is a boolean value
                # transform source_y to a 2d array, each element is a 1d array with 2 binary elements, [1,0] or [0,1]
                # self.source_y = np.array([[1,0] if y else [0,1] for y in self.source_y])
                # print(self.source_y[0])
                self.target_x, self.target_y,_ = ACSIncome.df_to_numpy(self.target_data_df)
                # self.target_y = np.array([[1,0] if y else [0,1] for y in self.target_y])
            except Exception as e:
                raise ValueError(f"No downloaded data: {e}")
        else:
            raise ValueError(f"Dataset {self.dataset} is not supported.")
        
