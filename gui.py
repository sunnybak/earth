import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


df = pd.read_csv('main_data.csv')
df = df.dropna(axis=0, how='all')

st.set_page_config(layout="wide")

st.title('EARTH')
st.write('EARTH (Eco-Aware Resource Tailored HPC) is a software that helps you predict the Time-to-Market (TTM), Cost, and Energy consumption of your SPICE simulations on HPCs. It helps you select the optimal SPICE simulation settings for your circuit type based on your priorities (TTM, Cost, Energy).')
st.write('This project was developed by the EARTH team at the University of California, Irvine in collaboration with Microsoft Azure.')


# add a sidebar
st.sidebar.title('Settings')
st.sidebar.write('Adjust the settings below to change the plot.')

core_names = {
    'intel': 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz', 'amd': 'AMD EPYC 7601 32-Core Processor'
}
irrelevant_cols = ['reltol', 'method', 'bsim4', 'gmin(S)',  'abstolV(V)', 'abstolI(A)', 'numcpu',  'maxstep(s)',  'coreType']
linear_cols = ['errpreset', 'nodes', 'perfmode']
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {
        'nodes': 103,
        'topk': 5,
        'weights': {
            'ttm': 0.7,
            'energy': 0.2,
            'cost': 0.1
        },
        'circuit': 'Ring Oscillator',
        'core_filter': 'No Filter',
        'core_cost': {
            'intel': 0.00001366120219,
            'amd': 0.00001802519733
        },
        'core_tdp': {
            'intel': 150,
            'amd': 180
        },
    }

# Sidebar
with st.sidebar:
    # nodes
    st.session_state.user_inputs['nodes'] = st.number_input('SPICE Nodes', min_value=103, max_value=503, step=100, value=st.session_state.user_inputs['nodes'])
    
    # circuit
    st.session_state.user_inputs['circuit'] = st.selectbox('Circuit', ['Ring Oscillator', 'Feedback'])
    

    # weights (must sum to 1)
    st.markdown('### Weights')
    st.session_state.user_inputs['weights']['ttm'] = st.slider('TTM %', min_value=0.1, max_value=1.0, step=0.1, value=st.session_state.user_inputs['weights']['ttm'])
    st.session_state.user_inputs['weights']['energy'] = st.slider('Energy %', min_value=0.1, max_value=1.0, step=0.1, value = st.session_state.user_inputs['weights']['energy'])
    st.session_state.user_inputs['weights']['cost'] = st.slider('Cost %', min_value=0.1, max_value=1.0, step=0.1, value = st.session_state.user_inputs['weights']['cost'])
    
    # a red button
    if st.button('Run', type='primary'):
        if (wsum := sum(st.session_state.user_inputs['weights'].values())) != 1:
            if wsum == 0:
                st.session_state.user_inputs['weights']['ttm'] = 0.7
                st.session_state.user_inputs['weights']['energy'] = 0.2
                st.session_state.user_inputs['weights']['cost'] = 0.1
            else:
                st.session_state.user_inputs['weights']['ttm'] /= wsum
                st.session_state.user_inputs['weights']['energy'] /= wsum
                st.session_state.user_inputs['weights']['cost'] /= wsum


def create_dataset(df):
    print(df.head())
    # make a deep copy of the dataframe
    X = df.copy()
    
    # map errpreset and perfmode to numerical values
    X['errpreset'] = X['errpreset'].map({'conservative': 0,
                                         'conservative_sigglobal': 0,
                                         'moderate': 0.5,
                                         'liberal': 1})

    X['perfmode'] = X['perfmode'].map({'Spectre': 0,
                                        'APS': 0.5,
                                        'APS++': 1})  
    # drop the irrelevant columns
    X = X.drop(irrelevant_cols, axis=1)

    # extract and drop label column from df
    time = X['transtotaltime(s)'] # minutes
    mem = X['peakresmem(MBytes)'] # GB
    cpu = X['totalused(%)'] # CPUs
    target = 'time'
    if target == 'time':
        y = time # workload
    elif target == 'mem':
        y = mem # memory
    elif target == 'cpu':
        y = cpu
    elif target == 'time_cpu':
        y = time * cpu
    X = X.drop(['transtotaltime(s)', 'peakresmem(MBytes)', 'totalused(%)'], axis=1)

    # drop columns with numpy NaN values
    X = X.dropna(axis=1)
    cols = X.columns

    # normalize all the columns using keras
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    
    # Create polynomial features
    polynomial_features = PolynomialFeatures(degree=3, include_bias=False)
    X_polynomial = polynomial_features.fit_transform(X)

    # Create logarithmic features
    X_log = np.log(X + 1)  # Adding 1 to avoid taking the log of zero

    # Combine the new features with the original features
    X_new = np.hstack((X_polynomial, X_log))
    features = polynomial_features.get_feature_names_out().tolist() + [f'log_{col}' for col in cols]
    
    # replace numpy nans with 0
    X_new = np.nan_to_num(X_new)
    
    # add the new features to the dataframe
    X_new = pd.DataFrame(X_new, columns=features)

    return X_new, y, scaler, polynomial_features

def lin_reg(X, y, show=True, rand=42):
    
    X = X.copy()
    y = y.copy()

    # Instantiate a linear regression model
    model = LinearRegression()
    
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand)
    if show:
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
    if show:
        print('\nTEST SCORE')
        print(f'Test Lin Reg Accuracy: {test_accuracy}')
        print("Test Mean Squared Error:", mse)
        print("Test R-squared:", r2)
    
    # train error
    # mse = mean_squared_error(y_train, y_pred_train)
    # r2 = r2_score(y_train, y_pred_train)
    # train_accuracy = model.score(X_test, y_pred_test)
    # print('\n\n TRAIN SCORE')
    # print(f'Train Lin Reg Accuracy: {train_accuracy}')
    # print("Train Mean Squared Error:", mse)
    # print("Train R-squared:", r2)
    return model, test_accuracy

if 'model' not in st.session_state:
    st.session_state.model = {'intel': None, 'amd': None}


    for core in ['intel', 'amd']:
        core_mask = df['coreType'].str.contains('2.4' if core == 'intel' else 'AMD')
        
        X, y, scaler, polynomial_features = create_dataset(df[core_mask])
        model, test_acc = lin_reg(X, y, show=False)
        features = X.columns
        coefficients = model.coef_
        features_and_coefficients = list(zip(features, coefficients))
        features_and_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

        features_plt = [feature for feature, _ in features_and_coefficients]
        coefficients_plt = [abs(coefficient) for _, coefficient in features_and_coefficients]
        
        # store the model
        st.session_state.model[core] = (model, test_acc, scaler, polynomial_features, features_plt, coefficients_plt)



def process_inputs(test_df, model_type):
    X = test_df.copy()
        
    # drop the irrelevant columns
    if all(col in X.columns for col in irrelevant_cols):
        X = X.drop(irrelevant_cols, axis=1)
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
    if model_type == 'intel':
        X = X[X['coreType'].str.contains('2.4')]
        X = X.drop('coreType', axis=1)
    elif model_type == 'amd':
        X = X[X['coreType'].str.contains('AMD')]
        X = X.drop('coreType', axis=1)
    else:
        raise

    # extract and drop dependent variable
    # if X contains all target columns
    if all(col in X.columns for col in ['time']):
        time = X['transtotaltime(s)'] # minutes
        mem = X['peakresmem(MBytes)'] # GB
        cpu = X['totalused(%)'] # CPUs
        y = time
        X = X.drop(['time'], axis=1)
        
        # save the target variable
        test_df = X
        # y = y
        return X, y

    test_df = X
    return X


# PREDICT
def predict(test_df, model_type):
    # model = Model(pd.DataFrame(test_df), model_type=self.model_type, target=self.target)
    model = st.session_state.model[model_type][0]
    scaler = st.session_state.model[model_type][2]
    polynomial_features = st.session_state.model[model_type][3]
    
    lin_df = process_inputs(test_df, model_type)
    # drop coreType
    print("COLUMNS IN LIN_DF", lin_df.columns)

    # Normalize the input data using the same feature scalers used during training
    input_scaled = scaler.transform(lin_df[scaler.get_feature_names_out().tolist()])
    input_polynomial = polynomial_features.transform(input_scaled)
    input_log = np.log(input_scaled + 1)  # Adding 1 to avoid taking the log of zero

    # Combine the new features with the original features
    tup = (input_polynomial, input_log)# if len(cat_df.columns) == 0 else (input_polynomial, input_log, cat_df)
    input_new = np.hstack(tup)
    input_new = np.nan_to_num(input_new)
    
    # Make predictions using the trained model
    prediction = model.predict(input_new)
    return prediction
    

def get_earth_table(nodes, core):
    TDP = st.session_state.user_inputs['core_tdp'] # watts
    COST = st.session_state.user_inputs['core_cost'] # dollars per cpu-second
    CORE = {'intel': 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz',
            'amd': 'AMD EPYC 7601 32-Core Processor'}

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
            data['coreType'].append(CORE[core])

    test_df = pd.DataFrame(data)
    preds = predict(test_df, core)

    test_df['TTM (s)'] = preds
    test_df['cost ($)'] = test_df['TTM (s)'] * COST[core]
    test_df['energy (kJ)'] = test_df['TTM (s)'] * TDP[core] / 1000
    
    # multiply all the values by the number of cores
    test_df['TTM (s)'] = test_df['TTM (s)'] / cores
    test_df['cost ($)'] = test_df['cost ($)'] * cores
    test_df['energy (kJ)'] = test_df['energy (kJ)'] * cores
    print(test_df)
    # save the df to a csv
    # test_df.to_csv('earth_table.csv', index=False)
    
    print("\nREPORT\n")
    print('Lowest TTM:', test_df['TTM (s)'].min())
    print('Lowest Cost:', test_df['cost ($)'].min())
    print('Lowest Energy:', test_df['energy (kJ)'].min())
    return test_df

# get results
result_intel = get_earth_table(st.session_state.user_inputs['nodes'], 'intel')
result_amd = get_earth_table(st.session_state.user_inputs['nodes'], 'amd')
    
# divider
st.markdown('#### Filters')
col1, col2, col3, col4 = st.columns(4)

with col1:
    filter_errpreset = st.selectbox('Accuracy Mode', ['All', 'liberal', 'moderate', 'conservative'])

with col2:
    filter_perfmode = st.selectbox('Performance Mode', ['All', 'Spectre', 'APS', 'APS++'])

with col3:
    filter_core = st.selectbox('Core', ['All', 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz', 'AMD EPYC 7601 32-Core Processor'])

with col4:
    # if topk is None
    topk = st.number_input('No. Results', min_value=1, max_value=10, step=1, value=5)
# topk
# st.session_state.user_inputs['topk'] = st.number_input('No. Results', min_value=1, max_value=10, step=1, value=st.session_state.user_inputs['topk'])

# apply filters
if filter_core == 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz':
    result = result_intel
elif filter_core == 'AMD EPYC 7601 32-Core Processor':
    result = result_amd
else:
    result = pd.concat([result_intel, result_amd])

# apply filters
if filter_errpreset != 'All':
    result = result[result['errpreset'] == filter_errpreset]
if filter_perfmode != 'All':
    result = result[result['perfmode'] == filter_perfmode]
if filter_core != 'All':
    result = result[result['coreType'] == filter_core]
    
result = result.sort_values(by=['TTM (s)'])
# result = result.head(st.session_state.user_inputs['topk'])
result = result.head(topk)
result = result.reset_index(drop=True)
# st.markdown('---')

# make the TTM column 1 decimal place
result['TTM (s)'] = result['TTM (s)'].apply(lambda x: f'{x:.1f}')
# make the cost column 4 decimal places
result['cost ($)'] = result['cost ($)'].apply(lambda x: f'{x:.4f}')
# make the energy column 3 decimal places
result['energy (kJ)'] = result['energy (kJ)'].apply(lambda x: f'{x:.3f}')


st.subheader('Result Histogram')
errpreset_plt = result['errpreset']
TTM_plt = result['TTM (s)']
cost_plt = result['cost ($)']
energy_plt = result['energy (kJ)']
width = 0.25
x = np.arange(len(result))
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, TTM_plt, width, label='TTM (s)')
ax.bar(x, cost_plt, width, label='cost ($)')
ax.bar(x + width, energy_plt, width, label='energy (kJ)')
ax.yaxis.set_tick_params(labelleft=False, left=False)
for i in range(len(result)):
    ax.text(x[i] - width, TTM_plt[i], f'{float(TTM_plt[i]):.1f}', ha='center', va='bottom')
    ax.text(x[i], cost_plt[i], f'{float(cost_plt[i]):.3f}', ha='center', va='bottom')
    ax.text(x[i] + width, energy_plt[i], f'{float(energy_plt[i]):.2f}', ha='center', va='bottom')

def core_pretty_name(core):
    if core == 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz':
        return 'Intel Gold'
    else:
        return 'AMD EPYC'

ax.set_title('Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels([f"{preset}\n{mode}\n{core_pretty_name(core)}" for preset, mode, core in zip(result['errpreset'], result['perfmode'], result['coreType'])])
ax.legend()

plt.tight_layout()
st.pyplot(plt)


# show the results
st.subheader('Result Table')
st.dataframe(result.rename(columns={'errpreset': 'Accuracy Mode', 'perfmode': 'Performance Mode', 'nodes': 'Nodes', 'coreType': 'Core'}), use_container_width=True)


st.subheader('Core Profiles - Relative Importance of Features')
col1, col2 = st.columns(2)
num_coeffs = 7
with col1:
    st.markdown('### Intel')
    plt.figure(figsize=(6, 3))
    plt.bar(st.session_state.model['intel'][4][:num_coeffs], st.session_state.model['intel'][5][:num_coeffs])
    # hide y axis label and ticks
    plt.yticks([])
    plt.xticks(rotation=90)
    st.pyplot(plt)

with col2:
    st.markdown('### AMD')
    plt.figure(figsize=(6, 3))
    print(st.session_state.model['amd'][4])
    plt.bar(st.session_state.model['amd'][4][:num_coeffs], st.session_state.model['amd'][5][:num_coeffs])
    plt.yticks([])
    plt.xticks(rotation=90)
    st.pyplot(plt)
