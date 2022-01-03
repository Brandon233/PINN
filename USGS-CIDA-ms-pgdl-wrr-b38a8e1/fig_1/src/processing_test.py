import os
raw_data_path = 'fig_1/tmp/mendota/shared/raw_data'
pretrain_inputs_path = 'fig_1/tmp/mendota/pretrain/inputs'
pretrain_model_path = 'fig_1/tmp/mendota/pretrain/model'
train_inputs_path = 'fig_1/tmp/mendota/train/inputs'
train_model_path = 'fig_1/tmp/mendota/train/model'
predictions_path = 'fig_1/tmp/mendota/train/out'
if not os.path.isdir(raw_data_path): os.makedirs(raw_data_path)

if not os.path.isdir(pretrain_inputs_path): os.makedirs(pretrain_inputs_path)

if not os.path.isdir(pretrain_model_path): os.makedirs(pretrain_model_path)

if not os.path.isdir(train_inputs_path): os.makedirs(train_inputs_path)

if not os.path.isdir(train_model_path): os.makedirs(train_model_path)

if not os.path.isdir(predictions_path): os.makedirs(predictions_path)

import pandas as pd

# define the filenames again if already downloaded from ScienceBase in a previous python session
train_obs_file=os.path.join(raw_data_path, 'me_similar_training.csv')
test_obs_file=os.path.join(raw_data_path, 'me_test.csv')

# read, subset, and write the training data for a single experiment
train_obs = pd.read_csv(train_obs_file)
train_obs_subset = train_obs[(train_obs['exper_id'] == 'similar_980') & (train_obs['exper_n'] == 1)].reset_index()[['date','depth','temp']]
train_obs_subset.to_feather(os.path.join(train_inputs_path, 'labels_train.feather'))

# read, subset, and write the testing data for a single experiment
test_obs = pd.read_csv(test_obs_file)
test_obs_subset = test_obs[(test_obs['exper_type'] == 'similar') & (test_obs['exper_n'] == 1)].reset_index()[['date','depth','temp']]
test_obs_subset.to_feather(os.path.join(train_inputs_path, 'labels_test.feather'))