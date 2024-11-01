from folktables import ACSDataSource, ACSEmployment

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["AL"], download=False)
print(acs_data.columns.values.tolist())
features, label, group = ACSEmployment.df_to_numpy(acs_data)
