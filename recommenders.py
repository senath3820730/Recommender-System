import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

from surprise import KNNBaseline
from surprise import SVD
from surprise import SlopeOne



def read_train():

    df_train = pd.read_csv('train.tsv',sep='\t',
                            names=['RowID','BeerID','ReviewerID','BeerName','BeerType','rating'])
    return df_train

def read_val():

    df_val = pd.read_csv('val.tsv',sep='\t',
                            names=['RowID','BeerID','ReviewerID','BeerName','BeerType','rating'])
    return df_val

def read_test():

    df_test = pd.read_csv('test.tsv',sep='\t',
                            names=['RowID','BeerID','ReviewerID','BeerName','BeerType'])
    return df_test

## loads the pandas data-frame as a surprise dataset
def load_df(curr_df, test_flag):
    reader = Reader(rating_scale=(0, 5))
    clean_df = curr_df.drop(['RowID','BeerName','BeerType'],axis=1)
    if (test_flag == False):
        return Dataset.load_from_df(clean_df[['ReviewerID','BeerID','rating']],reader)
    else:
        ## As test set has no ratings, a column for ratings with dummy values is added 
        clean_df['rating'] = np.nan
        return Dataset.load_from_df(clean_df[['ReviewerID', 'BeerID', 'rating']],reader)

## builds a full trainset and testset from the datasets obtained 
def get_set_from_df(data, datav, data_test):
    trainset = data.build_full_trainset()
    NA,valset = train_test_split(datav, test_size=1.0)
    NA_t,test_set = train_test_split(data_test, test_size=1.0)
    return (trainset, valset, test_set)

## creates predictions using the model that is passed 
def create_predictions(trainset, valset, test_set, model, row_id_pred, validate_flag, model_name):
    ## if flag is set the predictions are done on the validation set and the MAE is shown 
    if (validate_flag):
        print('Now predicting values for the validation set using '+ model_name +'...')
        val_predictions = model.fit(trainset).test(valset)
        print('MAE score for '+ model_name +' on the validation set: ', accuracy.mae(val_predictions))

    print('Now predicting values for the test set using '+ model_name +'...')
    predictions = model.fit(trainset).test(test_set)
    ## sorts the data by user id and item id so that the rowID column can be inserted here
    predicted_test_df = pd.DataFrame(predictions).sort_values(['uid','iid']).reset_index().drop('index',axis=1)
    ## joins the RowID from the test set that was sorted by Reviewer ID  and Beer ID
    predicted_test_df.insert(0, 'RowID', row_id_pred['RowID'])
    ## only extracts the rowID and predicted ratings 
    final_test_pred= predicted_test_df.sort_values('RowID')[['RowID','est']]
    pred_path = model_name + '.tsv'
    final_test_pred.to_csv(pred_path, sep = '\t', index=False, header = False)
    print('Test predictions file using ' + model_name + ' created.')

## trains the model with preset values or by performing a grid search, depending on the search_flag parameter
def model_train(trainset, valset, test_set, test_df, search_flag, validate_flag):
    
    ## sorts the test dataframe by ReviewerID and BeerID so that the RowID column can be extracted to use in the predictions file
    sorted_test_df = test_df.sort_values(['ReviewerID','BeerID'])
    row_id_pred = sorted_test_df['RowID'].reset_index().drop('index',axis=1)

    if (search_flag == False):
        ## KNNBaseline algorithm with preset parameters.

        bsl_options = {'method': 'als',
                    'n_epochs': 30,
                    'reg_u': 8,
                    'reg_i': 11
                    }
        model = KNNBaseline(k = 160, bsl_options = bsl_options)
        create_predictions(trainset, valset, test_set, model, row_id_pred, validate_flag, 'KNNBaseline')

        ## SlopeOne algorithm with preset parameters.
        model = SlopeOne()
        create_predictions(trainset, valset, test_set, model, row_id_pred, validate_flag, 'SlopeOne')

        ## SVD algorithm with preset parameters.
        model = SVD(n_factors=70,
                    n_epochs=55,
                    biased=True,
                    lr_all=0.005,
                    reg_all=0.05)
        create_predictions(trainset, valset, test_set, model, row_id_pred, validate_flag, 'SVD')
    
    else:
        ## sets of possible values to pass into the parameters 
        print('Running grid search on the KNNBaseline algorithm...')
        num_epochs = [5, 10, 20, 30]
        reg_user = [8,12,16,20]
        reg_item = [2,6,11,15]
        sim_names = ['msd', 'cosine', 'pearson', 'pearson_baseline']
        k_vals = [100, 160, 200]
        best = 10
        for epochs in num_epochs:
            for r_u in reg_user:
                for r_i in reg_item:
                    for sim_n in sim_names:
                        for k_val in k_vals:
                            bsl_options = {'method': 'als',
                                        'n_epochs': epochs,
                                        'reg_u': r_u,
                                        'reg_i': r_i
                                        }        
                            sim_options = {'name': sim_n,
                                        }
                            model = KNNBaseline(k = k_val, bsl_options = bsl_options, sim_options = sim_options)
                            predictions = model.fit(trainset).test(valset)
                            ## best score is compared with the current score to check for best.
                            if accuracy.mae(predictions)<best:
                                best_model = model
                                best = accuracy.mae(predictions)
        print(best_model)
        create_predictions(trainset, valset, test_set, best_model, row_id_pred, False, 'KNNBaseline')  

        ## SlopeOne does not have any parameters to set 
        model = SlopeOne()
        create_predictions(trainset, valset, test_set, model, row_id_pred, validate_flag, 'SlopeOne')

        print('Running grid search on the SVD algorithm...')
        num_factors = [70, 110, 150, 200]
        num_epochs= [55, 75, 105, 155]
        lr_all_lis = [0.005, 0.075, 0.01]
        reg_all_lis = [0.05, 0.1, 0.15]
        best = 10
        for num_f in num_factors:
            for num_ep in num_epochs:
                for lr_a in lr_all_lis:
                    for reg_a in reg_all_lis:
                        model = SVD(n_factors=num_f,
                                    n_epochs=num_ep,
                                    biased=True,
                                    lr_all=lr_a,
                                    reg_all=reg_a)
                        predictions = model.fit(trainset).test(valset)
                        if accuracy.mae(predictions)<best:
                            best_model = model
                            best = accuracy.mae(predictions)
        print(best_model)
        create_predictions(trainset, valset, test_set, best_model, row_id_pred, False, 'SVD')



if __name__ == "__main__":
    # sweep set to false to run with preset values.
    sweep = False
    # set validate_flag to True in order to make predictions on the validation set.
    validate_flag = False
    test_df = read_test()

    data = load_df(read_train(), False)
    datav = load_df(read_val(), False)
    data_test = load_df(test_df, True)
    (trainset, valset, test_set) = get_set_from_df(data, datav, data_test)
    model_train(trainset, valset, test_set, test_df, sweep, validate_flag)

