# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class Model(object):
    
    irrelevant_cols = ['reltol', 'method', 'bsim4', 'gmin(S)',  
                       'abstolV(V)', 'abstolI(A)', 'numcpu',  
                       'maxstep(s)']
    one_hot_cols = []
    targets = ['transtotaltime(s)', 'peakresmem(MBytes)', 'totalused(%)']
    linear_cols = ['errpreset', 'nodes', 'perfmode']
    categorical_cols = [
        'amd',
        'intel',
    ]
    
    def __init__(self, 
                 df=None, 
                 model_type='intel',
                 target='time',
                 ) -> None:
        self.df = df.dropna(axis=0, how='all')
        self.model_type = model_type
        self.target = target
        
        self.best_model = None
        self.best_accuracy = 0

    # input has format
    # [{
    # errpreset: conservative, 
    # perfmode: Spectre, 
    # nodes: 503,
    # coreType: AMD EPYC 7601 32-Core Processor,
    # }]
    # returns unnormalized X and y
    def process_inputs(self, test_df):
        X = test_df.copy()
           
        # drop the irrelevant columns
        if all(col in X.columns for col in self.irrelevant_cols):
            X = X.drop(self.irrelevant_cols, axis=1)
        X = X.dropna(axis=1)

        # map errpreset and perfmode to numerical values
        X['errpreset'] = X['errpreset'].map({'conservative': 0,
                                            'conservative_sigglobal': 0,
                                            'moderate': 0.5,
                                            'liberal': 1})

        X['perfmode'] = X['perfmode'].map({'Spectre': 0,
                                            'APS': 0.5,
                                            'APS++': 1})
        
        # filter out the processors
        X = X[X['coreType'].str.contains('2.4') | X['coreType'].str.contains('AMD')]
        if self.model_type == 'intel':
            X = X[X['coreType'].str.contains('2.4')]
            X = X.drop('coreType', axis=1)
        elif self.model_type == 'amd':
            X = X[X['coreType'].str.contains('AMD')]
            X = X.drop('coreType', axis=1)
        else:
            X = pd.get_dummies(X, columns=['coreType'], dtype=float)
            X = X.rename(columns={'coreType_AMD EPYC 7601 32-Core Processor': 'amd',
                                  'coreType_Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz': 'intel'})


        # extract and drop dependent variable
        # if X contains all target columns
        if all(col in X.columns for col in self.targets):
            time = X['transtotaltime(s)'] # minutes
            mem = X['peakresmem(MBytes)'] # GB
            cpu = X['totalused(%)'] # CPUs
            if self.target == 'mem': y = mem # memory
            elif self.target == 'cpu': y = cpu
            elif self.target == 'time_cpu': y = time * cpu
            else: y = time
            X = X.drop(self.targets, axis=1)
            
            # save the target variable
            self.df = X
            self.y = y
            return X, y

        self.df = X
        return X

    def fit_transform_data(self):
        
        # separate X into categorical and linear columns
        X = self.df
        X_cat = None
        if all(col in X.columns for col in self.categorical_cols):
            X_lin = X.drop(self.categorical_cols, axis=1)
            X_cat = X.drop(self.linear_cols, axis=1)
        else:
            X_lin = X

        # scale all the data
        scaler = StandardScaler()
        X_lin = scaler.fit_transform(X_lin)
        X_lin = pd.DataFrame(X_lin, columns=self.linear_cols)
        
        # Create polynomial features
        polynomial_features = PolynomialFeatures(degree=3, include_bias=False)
        X_polynomial = polynomial_features.fit_transform(X_lin)

        # Create logarithmic features
        X_log = np.log(X_lin + 1)  # Adding 1 to avoid taking the log of zero

        # Combine the new features with the original features
        tup = (X_polynomial, X_log) if X_cat is None else (X_polynomial, X_log, X_cat)
        X_new = np.hstack(tup)
        features = polynomial_features.get_feature_names_out().tolist() + \
                    [f'log_{col}' for col in self.linear_cols]

        if X_cat is not None:
            features += self.categorical_cols
    
        # replace numpy nans with 0
        X_new = np.nan_to_num(X_new)
        
        # add the new features to the dataframe
        X_new = pd.DataFrame(X_new, columns=features)
        
        # rename the coreType column to core
        self.scaler = scaler
        self.polynomial_features = polynomial_features
        
        self.df = X_new

    def train_model(self, test_size=0.3, rep=10, verbose=False):
        
        for i in range(rep):
            rand = np.random.randint(0, 100000)
            
            X = self.df
            y = self.y
            
            # Instantiate a linear regression model
            model = LinearRegression()
            
            # split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand)
            if verbose:
                print('\nTRAIN - TEST SPLIT')
                print(f"X_train shape: {X_train.shape}")
                print(f"X_test shape: {X_test.shape}")
                print(f"y_train shape: {y_train.shape}")
                print(f"y_test shape: {y_test.shape}")

            # Fit the model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            # test error
            mse = mean_squared_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            test_accuracy = model.score(X_test, y_test)
            if verbose:
                print('\nTEST SCORE')
                print(f'Test Lin Reg Accuracy: {test_accuracy}')
                print("Test Mean Squared Error:", mse)
                print("Test R-squared:", r2)
            
            # train error
            if verbose:
                mse = mean_squared_error(y_train, y_pred_train)
                r2 = r2_score(y_train, y_pred_train)
                train_accuracy = model.score(X_test, y_pred_test)
                print('\n\n TRAIN SCORE')
                print(f'Train Lin Reg Accuracy: {train_accuracy}')
                print("Train Mean Squared Error:", mse)
                print("Train R-squared:", r2)
            
            if test_accuracy > self.best_accuracy and \
                np.all(y_pred_test > 0) and \
                    np.all(y_pred_train > 0):
                self.best_accuracy = test_accuracy
                self.best_model = model
        # print(self.best_accuracy)
        return self.best_model, self.best_accuracy

    def predict(self, test_df):
        model = Model(pd.DataFrame(test_df), model_type=self.model_type, target=self.target)
        model.process_inputs()
        lin_df = model.df
        for catcol in self.categorical_cols:
            if catcol in lin_df.columns:
                lin_df = lin_df.drop(catcol, axis=1)
        cat_df = model.df.drop(self.linear_cols, axis=1)

        # Normalize the input data using the same feature scalers used during training
        input_scaled = self.scaler.transform(lin_df[self.scaler.get_feature_names_out().tolist()])
        input_polynomial = self.polynomial_features.transform(input_scaled)
        input_log = np.log(input_scaled + 1)  # Adding 1 to avoid taking the log of zero

        # Combine the new features with the original features
        tup = (input_polynomial, input_log) if len(cat_df.columns) == 0 else (input_polynomial, input_log, cat_df)
        input_new = np.hstack(tup)
        input_new = np.nan_to_num(input_new)
        
        # Make predictions using the trained model
        # print(input_new)
        prediction = self.best_model.predict(input_new)
        return prediction
    
    def get_earth_table(self, nodes):
        TDP = {'intel': 150, 'amd': 180} # watts
        COST = {'intel': 0.00001366120219, 'amd': 0.00001802519733} # dollars per cpu-second
        CORE = {'intel': 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz',
                'amd': 'AMD EPYC 7601 32-Core Processor'}
        df_actual = pd.read_csv('main_data.csv')
        actual_time = []
        cores = 1
        
        data = {
            'errpreset': [],
            'perfmode': [],
            'nodes': [],
            'coreType': [],
        }
        
        # https://azure.microsoft.com/en-in/pricing/details/virtual-machines/linux/#pricing
        for errpreset in ['liberal', 'conservative', 'moderate']:
            for perfmode in ['Spectre', 'APS', 'APS++']:
                data['errpreset'].append(errpreset)
                data['perfmode'].append(perfmode)
                data['nodes'].append(nodes)
                data['coreType'].append(CORE[self.model_type])
                
                # look up the actual time (file has the same columns as the training data)
                actual_time.append(np.mean(df_actual[(df_actual['errpreset'].str.contains(errpreset)) & 
                                    (df_actual['perfmode'] == perfmode) &
                                    (df_actual['nodes'] == nodes) &
                                    (df_actual['coreType'] == CORE[self.model_type])]['transtotaltime(s)']))
        test_df = pd.DataFrame(data)
        preds = self.predict(test_df)

        test_df['actual_time (s)'] = actual_time
        test_df['TTM (s)'] = preds
        test_df['cost ($)'] = test_df['TTM (s)'] * COST[self.model_type]
        test_df['energy (kJ)'] = test_df['TTM (s)'] * TDP[self.model_type] / 1000
        
        # multiply all the values by the number of cores
        test_df['actual_time (s)'] = test_df['actual_time (s)'] / cores
        test_df['TTM (s)'] = test_df['TTM (s)'] / cores
        test_df['cost ($)'] = test_df['cost ($)'] * cores
        test_df['energy (kJ)'] = test_df['energy (kJ)'] * cores
        print(test_df)
        # save the df to a csv
        test_df.to_csv('earth_table.csv', index=False)
        
        print("\nREPORT\n")
        print('Lowest TTM:', test_df['TTM (s)'].min())
        print('Lowest Cost:', test_df['cost ($)'].min())
        print('Lowest Energy:', test_df['energy (kJ)'].min())


if __name__ == '__main__':
    # create new dataset
    train_df = pd.read_csv('main_data.csv')
    # dataset = Model(model_type='both')
    dataset = Model(df=train_df, model_type='amd')
    
    # process the inputs
    dataset.process_inputs()
    
    # normalize the data
    dataset.fit_transform_data()
    
    # train the model
    dataset.train_model(verbose=False, rep=1000)
    
    dataset.get_earth_table(103)
    
    # test_datapoint = pd.DataFrame([{
    #     'errpreset': 'moderate',
    #     'perfmode': 'APS',
    #     'nodes': 503,
    #     'coreType': 'AMD EPYC 7601 32-Core Processor',
    #     # 'coreType': 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz',
    # }]) # time = 4467
    # pred = dataset.predict(test_datapoint)
    
    # print(pred)
    
    # don't use one-hot, just make columns yourself at the very start
    # dropna the cols for single core models
    
    # upload all spectre.outs to streamlit
    # click upload button
        # extract nodes, perfmode, errpreset, targets -> df
        # train model

    # user input nodes numeric
    # integer number line for n_cores

    # make 3x3x2=18 predictions for (errpreset, perfmode, core)
    # for each one, calculate expected TTM, cost, and energy
        # tdp = power(core_type)
        # cost = cost(core_type)

        # energy = tdp * n_cores * time
        # TTM = time / n_cores
        # cost = cost * n_cores * time
    # show results as table
    
    # inputs for bounds for energy, TTM, cost
    # filter table for results that are out of bounds
    
    
    
    
    
 