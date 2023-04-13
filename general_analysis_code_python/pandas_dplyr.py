import pandas as pd
import numpy as np
def filter_at(result_df,at):
    #at is a list of dictionary, each element of this dict represent an observation
    result_dfs = []

    for observation_dict in at:
        tmp_df = result_df.copy()
        for key,value in observation_dict.items():
            tmp_df = tmp_df[tmp_df[key]==value]
        result_dfs.append(tmp_df)
    result_df = pd.concat(result_dfs)
    return result_df

def concatenate_time_by_variables(df,variables):
### Sometimes we want to concatenate time series by variables,
### for example, we have a dataframe with columns: section, rate, rep, data
### we want to concatenate data by rate, so that the 2 regression share the same weights.

    # df is a dataframe with columns: many variables, (data or feature)
    # variables is a list of variables to combine

    #assert df is not empty
    assert not df.empty, 'df can not be empty'
    #assert variables is a list or tuple
    assert isinstance(variables,(list,tuple)), 'variables must be a list or tuple'
    column_names = df.columns
    assert 'data' in column_names or 'feature' in column_names, 'data or feature must be in the columns'
    time_variable = ('data' if 'data' in column_names else 'feature')
    tmp_df = df.copy()
    column_names = [x for x in column_names if x not in ('data','feature')]
    #make sure observations can be identified by column_names
    obs = tmp_df.groupby(column_names).apply(lambda x: len(x[time_variable])).reset_index()
    assert  (obs[0].values == 1).all(), 'observations can not be identified by rest of the columns'
    tmp_variables = variables.copy()
    for variable in variables:
        
        assert variable in column_names, f'{variable} must be in the columns'
        id_vars = [x for x in column_names if x not in tmp_variables]

        #sort by variable
        tmp_df = tmp_df.sort_values(by=variable)

        tmp_df = tmp_df.groupby(id_vars).apply(lambda x: np.concatenate(x[time_variable].values,axis=0)).reset_index()
        tmp_df.rename(columns={0:time_variable},inplace=True)
        tmp_variables = [x for x in variables if x != variable]
        column_names = [x for x in column_names if x != variable]
    return tmp_df

def replicate_data_feature(feature_df,data_df):
#Sometimes different observations of data share the same feature,
#Like a experiment is replicated twice, or it is done by different people.
#In this case, there are more group variables in data than in feature.
#We want to replicate the feature so that the number of observations in feature and data are the same.

#Or feature have more group variables than data. This is beacure we want to test which feature can fit data better.
#We want to replicate the data so that the number of observations in feature and data are the same.

#Besides, in this way, we can make sure that the order of feature is the same for the same of data.
    #feature_df is a dataframe with columns: group variables, feature
    #data_df is a dataframe with columns: group variables, data

    #make sure observations can be identified by column_names of data
    column_names_data = data_df.columns
    column_names_feature = feature_df.columns
    assert 'data' in column_names_data, 'data must be in the columns of data'
    assert 'feature' in column_names_feature, 'feature must be in the columns of feature'
    column_names_data = [x for x in column_names_data if x != 'data']
    column_names_feature = [x for x in column_names_feature if x != 'feature']
    obs = data_df.groupby(column_names_data).apply(lambda x: len(x['data'])).reset_index()
    assert  (obs[0].values == 1).all(), 'observations can not be identified by rest of the columns of data'

    same_group_variables = [x for x in column_names_data if x in column_names_feature]

    #join data_df and feature_df
    data_feature_df = data_df.merge(feature_df,how='outer',on=same_group_variables)

    #delete column data
    feature_df = data_feature_df.drop(columns=['data'])

    #delete column feature
    data_df = data_feature_df.drop(columns=['feature'])
    
    return feature_df,data_df

