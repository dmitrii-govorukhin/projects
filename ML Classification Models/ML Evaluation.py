import os
import sys
import pandas as pd
import numpy as np
import itertools
import collections
# from pyhive import hive
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import shap

import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import kaleido

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, brier_score_loss, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as ltb
from lightgbm import LGBMClassifier
import catboost as ctb
from catboost import CatBoostClassifier

import joblib as jb
from joblib import dump, load
seed = 42


def get_bin_analysis(actuals, preds, probs, tracked_units='target subpopulation leads'):
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    tracked_units = tracked_units.title()
    
    bin_df = pd.DataFrame({
        'No. of Leads': preds,
        'No. of %s' % tracked_units: actuals, 
        'Probability': probs[:, 1],
    })

    bin_analysis = bin_df.assign(
        Bin = lambda x: pd.cut(x['Probability'], bins, include_lowest=True)).groupby(['Bin']).agg({
            'No. of %s' % tracked_units: 'sum',
            'No. of Leads': 'count',
         })

    # revenue & generated leads in bin
    bin_analysis['%s Per Bin' % tracked_units] = bin_analysis['No. of %s' % tracked_units] \
        / bin_analysis['No. of Leads']

    # reverse bins
    bin_analysis.index = bins[:-1]
    bin_analysis = bin_analysis.iloc[::-1]

    # cumulative % generated leads in bin
    bin_analysis['Cumulative %% of %s' % tracked_units] = bin_analysis['No. of %s' % tracked_units].cumsum() \
        / bin_analysis['No. of %s' % tracked_units].sum()

    # cumulative % of returns in bin out of all returns 
    bin_analysis['Cumulative % of Population in the Bin'] = bin_analysis['No. of Leads'].cumsum() \
        / bin_analysis['No. of Leads'].sum()

    # potential revenue generating leads
    bin_analysis['No. of Potential %s' % tracked_units] = bin_analysis['No. of Leads'] \
        - bin_analysis['No. of %s' % tracked_units]
    
    bin_analysis = bin_analysis[[
        'No. of Leads',
        'No. of %s' % tracked_units,
        '%s Per Bin' % tracked_units,
        'Cumulative %% of %s' % tracked_units,
        'Cumulative % of Population in the Bin',
        'No. of Potential %s' % tracked_units]]
    
    return bin_analysis    


def calculate_bin_score(bin_analysis):
    """Provide a `bin_analysis` as input.
    This function will take the sum of products of each bin percentile with the corresponding
    number of positive cases scored into that bin."""
    products = [x*y for x,y in zip(list(bin_analysis.index),bin_analysis.iloc[:,1])]
    score = sum(products)
    return score


def find_best_n_features(starting_num_features, ending_num_features,training_pkl_path, testing_pkl_path):
    """This function runs sklearn Recursive Feature Elimination from `starting_features` to `ending_features` and logs the 
    performance results to a dataframe for the user's comparison.
    `training_pkl` and `testing_pkl` should be the paths to tuple-packed .pkl files that contain the x and y for training/testing respectively."""
    from sklearn.feature_selection import RFE
    if ending_num_features < starting_num_features and ending_num_features > 0:
        result = pd.DataFrame(columns = ['num_coefficients','score', 'top_bin_pcnt','avg_pcnt_first_five_bins'])
        for iteration in tqdm(range(starting_num_features, ending_num_features-1,-1)):
            # reset the training and testing data to all features:
            try:
                x_train = x_train[selected_columns]
                x_test = x_test[selected_columns]      
            except:
                x_train, y_train = jb.load(training_pkl_path)
                x_test, y_test = jb.load(testing_pkl_path)

            # instanciate logistic regression and fit the object on the training data
            logistic = LogisticRegression(class_weight='balanced', random_state=np.random.seed, C=.1,max_iter=100000) # reward the model more when it finds the minority class (solves class imbalance problem)
            logistic.fit(x_train, y_train)

            # instanciate feature selector with current num_features
            selector = RFE(estimator = logistic, n_features_to_select=iteration, step=1)

            # columns selected with boolean
            selector.fit(x_train, y_train)
            feature_index = selector.support_

            # specifying selected column names
            selected_columns = x_train.columns[feature_index]
            
            # creating new x_train and x_test with specified columns
            x_train = x_train[selected_columns]
            x_test = x_test[selected_columns]        
            
            # instanciate logistic regression and fit the object on the new specified columns
            logistic = LogisticRegression(class_weight='balanced', random_state=np.random.seed, max_iter=100000) # reward the model more when it finds the minority class (solves class imbalance problem)
            logistic.fit(x_train, y_train)

            # predict the outcomes and the probabilities for the testing data
            predictions = logistic.predict(x_test)
            probabilities = logistic.predict_proba(x_test)
            
            #get the score for the bin analysis these predictions would result in
            this_bin_analysis = get_bin_analysis(y_test,predictions,probabilities)
            this_score = calculate_bin_score(this_bin_analysis)
            
            #append results to the output dataframe
            result = result.append({'num_coefficients':iteration,
                                    'score':this_score, 
                                    'top_bin_pcnt':next(value for value in this_bin_analysis.iloc[:,2] if (not np.isnan(value))),
                                    'avg_pcnt_first_five_bins': this_bin_analysis.iloc[0:5,2].mean()}, ignore_index=True)
        
        #return a matrix containing the results:
        return result

    else:
        print('Error. Please ensure ending features is lower than starting features and is also greater than zero.')


def custom_plot_confusion_matrix(actual, predictions, class_names=['No Notice', 'Notice']):
    '''
    Creates a figure that shows the model's confusion matrix
    inputs:
        actual: array containing the actual target.
        predictions: array containing the predicted target.
        class_names: array containing the different target labels.
                     'No Notice' (0) and 'Notice' (1) by default.
    '''
    
    assert len(predictions) == len(actual), 'Length mismatch between the actual outputs and the predicted outputs'
    assert len(class_names) == len(set(actual)), 'Number of classes must match the number of unique outputs'
    
    cm = confusion_matrix(actual, predictions)
    fig, ax = plt.subplots(figsize=(8,8))
    plot = ax.matshow(cm, cmap='OrRd', alpha=.9)
    
    # add a color bar legend
    fig.colorbar(plot, shrink=.7)
    
    plt.ylim([-.5, 1.5])
    
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    
    tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # annotate the graph
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontalalignment='center', color='black')
        
    plt.tight_layout()


def plot_coefficients(coefficients, figsize=(12, 7)):
    '''
    Creates a figure containing a vertical barplot and returns dataframe
    sorted by the coefficient values
    inputs: 
        coefficients: dict containing pairs of {feature name: coefficient value}.
        figsize: optional tuple representing the figure size (height, width).
        figsize=(12, 7) by default.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    
    ## sort coefs
    coefficients = pd.DataFrame.from_dict(coefficients)
    coefficients = coefficients.rename({0: 'Coefficient Name', 1: 'Coefficient Value'}, axis=1)
    coefficients = coefficients.sort_values(by=['Coefficient Value'], ascending=False)

    # plot coefs
    sns.barplot(y='Coefficient Name', x='Coefficient Value', data=coefficients, ax=ax)
    ax.set_yticklabels(coefficients['Coefficient Name'], ha='right', fontsize=12)
    ax.set_title('Regression Cofficients', fontsize=20)
    ax.set_ylabel('')
    ax.set_xlabel('Coefficient Values', fontsize=14)
    
    plt.tight_layout()
    return coefficients


def custom_plot_roc_curve(actual, probabilities, major_class=0, model_name='Logistic Regression'):
    '''
    Creates a figure containing the model's ROC AUC curve
    inputs:
        actual: array containing the actual target.
        probabilities: array containing the predicted probabilities.
        major_class: int, the majority class in the dataset. 
            0 by default.
        model_name: string, model name for figure title
            'Logistic Regression' by default.
    '''
    plt.figure(figsize=(12, 7))
    probabilities = np.array(probabilities)
    
    assert type(probabilities) == np.ndarray, 'The probabilities must be saved as a numpy array'
    assert len(probabilities) == len(actual), 'Length mismatch between the actual outputs and the predicted outputs'
    
    major_probs = [major_class for _ in range(len(actual))]
    
    try:
        preds_probs = probabilities[:, 1]
    except IndexError:
        preds_probs = probabilities
    
    # calculate scores
    major_auc = roc_auc_score(actual, major_probs)
    preds_auc = roc_auc_score(actual, preds_probs)
    
    # calculate roc curves
    major_fpr, major_tpr, _ = roc_curve(actual, major_probs)
    preds_fpr, preds_tpr, _ = roc_curve(actual, preds_probs)
    
    # plot the roc curve for the model
    plt.plot(major_fpr, major_tpr, linestyle='--', label='No Skill: ROC AUC = %.3f' % (major_auc))
    plt.plot(preds_fpr, preds_tpr, marker='.', label='Logistic: ROC AUC = %.3f' % (preds_auc))
    
    # style figure
    plt.title('%s - ROC AUC Curve' % model_name, fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    
    # show the legend
    plt.legend(fontsize=18)
    plt.tight_layout()
    pass


def connect_database(username, password, database='xxx'):
    '''
    Establish connection to the data lake through PyHive
    inputs:
        username: string, login username.
        password: string, login password.
        database: string, database to access. WS_AUDIT_EXTRACT by default.
    '''
    connection = hive.connect(host='xxx',
                          port=xxx,
                          database=xxx,
                          username=xxx,
                          password=xxx,
                          auth='xxx')
    return connection


def plot_feature_importance(model, feature_names, n=20):
    '''
    Receives an sklearn type model and creates feature importances graph
    Note: Not all sklearn model object have this feature
    inputs:
        model: sklearn model object, or any other model with the feature_importances_ attribute
        feature_names: array, containing the names of all features the model was trained on
        n: integer, the number of features to include in the graph, 12 by default
   
    '''
    plt.figure(figsize=(14,10))
   
   
    assert hasattr(model, 'feature_importances_')
    'Model object does not have the feature_importences_ attribute'
    # extract the feature importances
    importances = model.feature_importances_
   
    feature_matrix = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_matrix = feature_matrix.sort_values(by='Importance', ascending=False).head(n)
    ax = sns.barplot(y="Feature", x="Importance", data=feature_matrix, ci='sd')
    plt.tight_layout()
   
    return ax


def score_to_invariant(model, query, connection, chunksize, num_of_iterations,
                                identifier_columns, table_name, input_columns=None):
    
    # create iterator object
    all_records = pd.read_sql(query, connection, chunksize=chunksize)
    
    # iterate through the generator object
    for chunk in tqdm(all_records, ascii=True, total=num_of_iterations):
        
        # remove table name from column names (if necessary)
        try:
            chunk.columns = [str(col).split('.')[1] for col in chunk.columns.values]
    
        except IndexError:
            pass
        
        assert isinstance(identifier_columns, list) or isinstance(identifier_columns, str) 
        # must include 1 or more columns to append the score
        
        scores = pd.DataFrame(chunk[identifier_columns].copy(deep=True))
      
        if isinstance(input_columns, list) or isinstance(input_columns, str):
            chunk = chunk[input_columns]
        # selecting only the columns used during model training
        
        chunk = chunk.fillna(0.0)    # fill missing values if exist
        
        scores['score'] = model.predict_proba(chunk)[:,1]
        
        # create cursor object to excute commands in Hive database
        cursor = connection.cursor()
        
        # without this, to_numpy() will truncate output fed to array2string()
        np.set_printoptions(threshold=sys.maxsize)    
        
        # convert df to np array to string, adjust formating for sql
        scores = np.array2string(scores.to_numpy())[1:-1].replace('[', '(').replace(']', ')').replace('\n', ',').replace(' ', ', ').replace(',,', ',').replace('u', '').replace('None', 'NULL')  
              
        # this might need to be updated for new scoring    
        insert = 'INSERT INTO %s (dln, tpid, tax_yr, score) VALUES %s' % (table_name, scores)   
        
        cursor.execute(insert)
        

def plot_target_class_balance_seaborn(df, normalize=True, target='', figsize=(14, 9), image_path='', label='', classes=None):
    """
    Create a bar plot to visualize the class balance using Plotly and save the figure as an image and HTML file.
    
    :param df: DataFrame containing the data to plot
    :param target: str, column name of the target variable (class labels)
    :param image_path: str, path to save the image file
    :param label: str, descriptive label for the plot
    :param figsize: tuple (int, int), the width and height of the figure in pixels
    :param classes: list, class labels corresponding to the tick values
    """

    # Get value counts
    target_counts = df[target].value_counts(normalize=normalize)

    # Create a bar plot with Seaborn
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=target_counts.index, y=target_counts.values)

    # Set x-axis labels
    ax.set_xticklabels(classes)
    ax.set_xlabel('')

    # Set y-axis label
    ax.set_ylabel('Proportion')
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Set plot title
    ax.set_title(label)

    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_target_class_balance_plotly(df, target='', image_path='', label='', figsize=(2000, 1100), classes=None):
    """
    Create a bar plot to visualize the class balance using Plotly and save the figure as an image and HTML file.
    
    :param df: DataFrame containing the data to plot
    :param target: str, column name of the target variable (class labels)
    :param image_path: str, path to save the image file
    :param label: str, descriptive label for the plot
    :param figsize: tuple (int, int), the width and height of the figure in pixels
    :param classes: list, class labels corresponding to the tick values
    """

    # Get normalized value counts for the target variable
    target_counts = df[target].value_counts(normalize=True)

    # Create a bar trace with default colors
    trace = go.Bar(x=target_counts.index, y=target_counts.values, width=0.2)

    # Create layout with custom title, axis labels, and figure size
    layout = go.Layout(title=label,
                       xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=classes),
                       width=figsize[0], height=figsize[1])

    # Create a Plotly figure with the specified trace and layout
    fig = go.Figure(data=[trace], layout=layout)

    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}_Class_Balance.png"
    pio.write_image(fig, filename)

    # Save the figure as an HTML file with an interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Class_Balance.html", auto_open=True)

    # Display the figure in the notebook
    fig.show()


def multicol_columns_df(df, excluded_columns, threshold=None):
    """
    Plots a correlation matrix for the given DataFrame.

    :param df: DataFrame containing the data
    :param excluded_columns: List of columns to exclude from the correlation matrix
    :param threshold: Threshold value to filter the correlation matrix
    :param output_path: Path to save the correlation matrix image
    :param figsize: Tuple specifying the figure size (width, height) in inches
    :param font_scale: Float for scaling the font size
    :param cmap: Color map for the heatmap
    :param annot: If True, write the data value in each cell
    :param center: Value at which to center the colormap when plotting divergant data
    :param vmin: Minimum data value that corresponds to the bottom of the colormap
    :param vmax: Maximum data value that corresponds to the top of the colormap
    """
    # Calculate the correlation matrix
    corr_matrix = df.drop(columns=excluded_columns).corr()

    # Apply threshold if provided
    correlations = {}
    if threshold is not None:
        corr_matrix = corr_matrix.loc[((corr_matrix > threshold) & (corr_matrix <= 1)).any(),((corr_matrix > threshold) & (corr_matrix <= 1)).any()]
        corr_pairs = corr_matrix.unstack()
        filtered_corr = corr_pairs[(np.abs(corr_pairs) >= threshold) & ~((corr_pairs == 1) & (corr_pairs.index.get_level_values(0) == corr_pairs.index.get_level_values(1)))].sort_index().drop_duplicates()
        for (col1, col2), corr_value in filtered_corr.items():
            correlations[f"{col1}, {col2}"] = corr_value
            
    correlations_df = pd.DataFrame({'Correlations': correlations.values()}, index=correlations.keys())
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    correlations_df = print(correlations_df)
    
    return correlations_df, correlations_df.shape
    
    
def plot_correlation_matrix_seaborn(df, excluded_columns, threshold=None, image_path='', figsize=(200, 200), font_scale=2, cmap='coolwarm', annot=True, center=0, vmin=-1, vmax=1):
    """
    Plots a correlation matrix for the given DataFrame.

    :param df: DataFrame containing the data
    :param excluded_columns: List of columns to exclude from the correlation matrix
    :param threshold: Threshold value to filter the correlation matrix
    :param output_path: Path to save the correlation matrix image
    :param figsize: Tuple specifying the figure size (width, height) in inches
    :param font_scale: Float for scaling the font size
    :param cmap: Color map for the heatmap
    :param annot: If True, write the data value in each cell
    :param center: Value at which to center the colormap when plotting divergant data
    :param vmin: Minimum data value that corresponds to the bottom of the colormap
    :param vmax: Maximum data value that corresponds to the top of the colormap
    """
    # Calculate the correlation matrix
    corr_matrix = df.drop(columns=excluded_columns).corr()

    # Apply threshold if provided
    correlations = {}
    if threshold is not None:
        corr_matrix = corr_matrix.loc[((corr_matrix > threshold) & (corr_matrix <= 1)).any(),((corr_matrix > threshold) & (corr_matrix <= 1)).any()]
        corr_pairs = corr_matrix.unstack()
        filtered_corr = corr_pairs[(np.abs(corr_pairs) >= threshold) & ~((corr_pairs == 1) & (corr_pairs.index.get_level_values(0) == corr_pairs.index.get_level_values(1)))].sort_index().drop_duplicates()
        for (col1, col2), corr_value in filtered_corr.items():
            correlations[f"{col1}, {col2}"] = corr_value
            
    correlations_df = pd.DataFrame({'Correlations': correlations.values()}, index=correlations.keys())
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    print(correlations_df)
    
        
    # Set figure size and font scale
    sns.set(rc={'figure.figsize': figsize})
    sns.set(font_scale=font_scale)

    # Plot heatmap
    heatmap = corr_matrix.loc[((corr_matrix > threshold) & (corr_matrix < 1)).any(),((corr_matrix > threshold) & (corr_matrix < 1)).any()]
    ax = sns.heatmap(heatmap, cmap=cmap, annot=annot, center=center, vmin=vmin, vmax=vmax)
    ax.xaxis.tick_top()
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(ax.get_xticklabels(), rotation=90)
    title='Correlation Matrix'
    plt.title(title)
    plt.savefig(f"{image_path}{title.replace(' ', '_')}.png")
    plt.show()

    # Reset the figure size and font scale to their default values
    sns.set(rc={"figure.figsize": [6.4, 4.8]})
    sns.set(rc={"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12})


def plot_correlation_matrix_plotly(df, excluded_columns, threshold=None, image_path='', figsize=(800, 600), font_scale=2, colorscale=None, center=0, vmin=-1, vmax=1):
    """
    Plots a correlation matrix for the given DataFrame using Plotly.

    :param df: DataFrame containing the data
    :param excluded_columns: List of columns to exclude from the correlation matrix
    :param image_path: Path to save the correlation matrix image
    :param figsize: Tuple specifying the figure size (width, height) in pixels
    :param font_scale: Float for scaling the font size
    :param colorscale: Color scale for the heatmap
    :param center: Value at which to center the colormap when plotting divergant data
    :param vmin: Minimum data value that corresponds to the bottom of the colormap
    :param vmax: Maximum data value that corresponds to the top of the colormap
    """
    # Set default colorscale if not provided
    if colorscale is None:
        colorscale = [
            [0, "steelblue"],
            [0.5, "white"],
            [1, "firebrick"]
        ]
    
    # Calculate the correlation matrix
    corr_matrix = df.drop(columns=excluded_columns).corr()

    # Apply threshold if provided
    if threshold is not None:
        corr_matrix = corr_matrix.where((np.abs(corr_matrix) >= threshold))

    # Create a mask to hide NaN values in the annotation text
    mask = np.where(np.isnan(corr_matrix), '', corr_matrix.round(2))

    # Create the Plotly heatmap figure
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=mask,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        zmid=center
    )

    # Customize the appearance of the plot
    fig.update_layout(
        title="Correlation Matrix",
        xaxis=dict(title="Features"),
        yaxis=dict(title="Features", autorange="reversed"),
        font=dict(size=font_scale * 10),
        width=figsize[0],
        height=figsize[1]
    )

    # Save the figure to a file with a descriptive name
    filename = f"{image_path}Correlation_Matrix.png"
    pio.write_image(fig, filename)

    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}Correlation_Matrix.html", auto_open=True)

    # Display the plot
    fig.show()


# Define the function to standard scale specified continuous columns in X_train and X_test
def standard_scaler(X_train, X_test, continuous_columns, scaler_path):
    """
    Standard scale continuous columns and return X_train and X_test ready for modeling.
    
    :param X_train: DataFrame, training feature set
    :param X_test: DataFrame, testing feature set
    :param continuous_columns: str or list, continuous columns to scale
    """
    # Create a ColumnTransformer instance with the StandardScaler transformer
    # Apply the transformer to the specified continuous_columns and pass through the remaining columns
    preprocessor = ColumnTransformer(transformers=[
    ('Scale', StandardScaler(), continuous_columns)])
    
    # Fit the ColumnTransformer on the training data (learn the encoding from X_train)
    preprocessor.fit(X_train[continuous_columns])
    ss = preprocessor.named_transformers_['Scale']

    # save the standard scaler to inverse transform later
    #scaler = scaler_path
    #jb.dump(ss, scaler)
    
    #get the transformed columns, save them to _ohe variables
    X_train_ss = pd.DataFrame(preprocessor.transform(X_train[continuous_columns]), index=X_train.index,
                             columns=continuous_columns)
    X_test_ss = pd.DataFrame(preprocessor.transform(X_test[continuous_columns]), index=X_test.index,
                            columns=continuous_columns)

    #combine the transformed columns with the remaining columns:
    X_train = pd.concat([X_train_ss, X_train.drop(columns=continuous_columns).copy(deep=True)], axis=1)
    X_test = pd.concat([X_test_ss, X_test.drop(columns=continuous_columns).copy(deep=True)], axis=1)
    
    # Return the transformed X_train and X_test datasets, ready for modeling
    return X_train, X_test


# Define the function to one-hot encode specified categorical columns in X_train and X_test
def one_hot_encode(X_train, X_test, categorical_columns):
    """
    One hot encode categorical columns and return X_train and X_test ready for scaling or modeling.
    
    :param X_train: DataFrame, training feature set
    :param X_test: DataFrame, testing feature set
    :param categorical_columns: str or list, categorical columns to encode
    """
    # Create a ColumnTransformer instance with the OneHotEncoder transformer
    # Apply the transformer to the specified categorical_columns and pass through the remaining columns
    preprocessor = ColumnTransformer(transformers=[
    ('OneHotEncode', OneHotEncoder(sparse=False), categorical_columns)])
    
    # Fit the ColumnTransformer on the training data (learn the encoding from X_train)
    preprocessor.fit(X_train[categorical_columns])
    ohe = preprocessor.named_transformers_['OneHotEncode']
    ohe_column_names = ohe.get_feature_names_out(categorical_columns)
    
    #get the transformed columns, save them to _ohe variables
    X_train_ohe = pd.DataFrame(preprocessor.transform(X_train[categorical_columns]), index=X_train.index,
                             columns=ohe_column_names)
    X_test_ohe = pd.DataFrame(preprocessor.transform(X_test[categorical_columns]), index=X_test.index,
                            columns=ohe_column_names)

    #combine the transformed columns with the remaining columns:
    X_train = pd.concat([X_train_ohe, X_train.drop(columns=categorical_columns).copy(deep=True)], axis=1)
    X_test = pd.concat([X_test_ohe, X_test.drop(columns=categorical_columns).copy(deep=True)], axis=1)

    # Return the transformed X_train and X_test datasets, ready for modeling
    return X_train, X_test


# defining an evaluation classification function for automation and evaluating subsequent models
def evaluate_classification(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        classes=None,
        model_path='xxx',
        image_path='xxx',
        label=''
):
    """
    Evaluate a classification model and display the results as figures and classification reports.

    :param model: Trained classifier model
    :param X_train: DataFrame, training feature set
    :param X_test: DataFrame, testing feature set
    :param y_train: Series, training target set
    :param y_test: Series, testing target set
    :param classes: list, class labels for the target variable
    :param model_path: str, path to save the trained model
    :param image_path: str, path to save the evaluation figures
    :param label: str, descriptive label for the model
    """
    # retrieve predictions for train and validation data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # retrieve probabilites for train and validation data
    y_hat_train = model.predict_proba(X_train)
    y_hat_test = model.predict_proba(X_test)

    # retrieve probabilities for the positive class
    pos_probs_train = y_hat_train[:, 1]
    pos_probs_test = y_hat_test[:, 1]

    # save the trained model
    model_filename = f"{model_path}{label.replace(' ', '_')}.pkl"
    dump(model, model_filename)


    # print training classification report
    header = label + " Classification Report - Train"
    dashes = "---" * 20
    print(dashes, header, dashes, sep='\n')
    print(classification_report(y_train, y_pred_train, target_names=classes, digits=4))

    # display training figures
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
    # adjust spacing between subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    # Plot a confusion matrix on the train data
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_train, y=y_train, display_labels=classes, values_format=',', ax=axes[0])
    axes[0].grid(False)
    axes[0].set(title='Confusion Matrix - Train')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # plot ROC curve
    RocCurveDisplay.from_estimator(model, X_train, y_train, name=label, ax=axes[1])
    roc = axes[1]
    roc.plot([0,1], [0,1], ls=':', label='No Skill')
    roc.grid()
    roc.set_title('Receivier Operator Characteristic - Train')

    # plot Precision-Recall curve
    PrecisionRecallDisplay.from_estimator(model, X_train, y_train, ax=axes[2], name=label)
    # y axis is Precision
    axes[2].set_ylabel('Precision')
    # x axis is Recall
    axes[2].set_xlabel('Recall')
    axes[2].set_title('Precision-Recall AUC - Train')

    fig.tight_layout()
    plt.savefig(f"{image_path}{label.replace(' ', '_')}_Model_Evaluation_Train.png")
    plt.show();


    # print test classification report
    header_ = label + " Classification Report - Test"
    print(dashes, header_, dashes, sep='\n')
    print(classification_report(y_test, y_pred_test, target_names=classes, digits=4))

    # display test figures
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
    # adjust spacing between subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    # Plot a confusion matrix on the test data
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test, display_labels=classes, values_format=',', ax=axes[0])
    axes[0].grid(False)
    axes[0].set(title='Confusion Matrix - Test')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # plot ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=label, ax=axes[1])
    axes[1].plot([0,1], [0,1], ls=':', label='No Skill')
    axes[1].grid()
    axes[1].set_title('Receivier Operator Characteristic - Test')

    # plot Precision-Recall curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[2], name=label)
    # y axis is Precision
    axes[2].set_ylabel('Precision')
    # x axis is Recall
    axes[2].set_xlabel('Recall')
    axes[2].set_title('Precision-Recall AUC - Test')

    plt.legend()
    fig.tight_layout()
    plt.savefig(f"{image_path}{label.replace(' ', '_')}_Model_Evaluation_Test.png")
    plt.show();
    
    
def evaluate_algorithms(X_train, X_test, y_train, y_test, random_state, model_path='', image_path='', classes=None):
    """
    Evaluate different sampling methods with a given classifier on the given data.
    
    Parameters:
        - algorithms: a list of tuples, each containing a string name for the algorithm 
          and the sampler instance.
        - X_train: the training set features.
        - X_test: the validation set features.
        - y_train: the training set target.
        - y_test: the validation set target.
        - random_state: the random state for the classifier.
    """
    # defining the algorithms
    algorithms = [
    # ('Dummy Classifier', DummyClassifier(strategy='most_frequent')),
    ('Logistic Regression', LogisticRegression(random_state=seed)),
#     ('Bagging Classifier', BaggingClassifier(random_state=seed)),
#     ('AdaBoost Classifier', AdaBoostClassifier(random_state=seed)),
#     ('Decision Tree Classifier', DecisionTreeClassifier(random_state=seed)),
#     ('Extra Trees Classifier', ExtraTreesClassifier(random_state=seed)),
#    ('Random Forest Classifier', RandomForestClassifier(random_state=seed)),
    # ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=seed)),
    # ('Hist Gradient Boosting Classifier', HistGradientBoostingClassifier(random_state=seed)),
    ('XGB Classifier', XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='aucpr')),
    ('LGBM Classifier', LGBMClassifier(random_state=seed, tree_learner='feature', verbose=-1)),
    ('CatBoost Classifier', CatBoostClassifier(random_state=seed, logging_level='Silent')),
]

    for name, algorithm in algorithms:
        algorithm.fit(X_train, y_train)
        evaluate_classification(algorithm, 
                                X_train=X_train, 
                                X_test=X_test, 
                                y_train=y_train, 
                                y_test=y_test, 
                                classes=classes,
                                label=f"{name}", 
                                model_path=model_path,
                                image_path=image_path
                               )
     

def random_under_sampler(X_train, y_train):
    data = np.column_stack((X_train, y_train))
    unique_classes, class_counts = np.unique(y_train, return_counts=True)

    # Find the minimum number of samples across classes
    min_samples = np.min(class_counts)

    # Initialize an empty array for the new resampled data
    resampled_data = np.empty((0, data.shape[1]))

    for cls in unique_classes:
        # Randomly select 'min_samples' number of samples for each class
        class_samples = data[data[:, -1] == cls]
        selected_samples = class_samples[np.random.choice(class_samples.shape[0], min_samples, replace=False)]
        resampled_data = np.vstack((resampled_data, selected_samples))

    # Separate features and labels and return them
    X_resampled = pd.DataFrame(resampled_data[:, :-1], columns=X_train.columns)
    y_resampled = resampled_data[:, -1]

    return X_resampled, y_resampled
    return X_resampled, y_resampled
        
        
def random_over_sampler(X_train, y_train):
    # Determine the majority class
    unique, counts = np.unique(y_train, return_counts=True)
    majority_class = unique[np.argmax(counts)]

    # Separate the majority and minority classes
    X_majority = X_train[y_train == majority_class]
    y_majority = y_train[y_train == majority_class]
    X_minority = X_train[y_train != majority_class]
    y_minority = y_train[y_train != majority_class]

    # Duplicate the minority class samples with replacement to match the majority class count
    minority_size = len(y_majority)
    indices = np.random.randint(0, len(y_minority), size=minority_size)
    
    X_resampled_minority = X_minority.iloc[indices]
    y_resampled_minority = y_minority.iloc[indices]
        
    X_resampled = pd.concat([X_majority, X_resampled_minority], axis=0)
    y_resampled = pd.concat([y_majority, y_resampled_minority], axis=0)

    # Shuffle the combined dataset
    X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=seed)

    return X_resampled, y_resampled


def recursive_feature_elimination(X_train, X_test, y_train, y_test, start_n, end_n=5, model=LogisticRegression(random_state=seed), scoring='f1_macro', stepdown=1, model_path=''):
    """
    This function finds the top 5 performing models using Recursive Feature Elimination (RFE) 
    and saves them as .pkl files. The performance is evaluated using various metrics, such as 
    accuracy, f1-score, recall, precision, brier_score, and recall_macro. The function returns
    a sorted DataFrame containing these evaluation metrics for different values of n, where n 
    is the number of selected features in RFE.

    Parameters:
    start_n (int): Initial number of features to start with.
    end_n (int): Final number of features to end with.
    model (estimator): The estimator to use for RFE.
    X_train (DataFrame): The training dataset.
    X_test (DataFrame): The testing dataset.
    y_train (Series): The target variable for the training dataset.
    y_test (Series): The target variable for the testing dataset.
    scoring (string): What metric to sort by
    stepdown (int): how many n_features_to_select to proceed down to after running RFE for n_features. Default is 1.

    Returns:
    DataFrame: Sorted dataframe containing evaluation metrics for different values of n.
    """
    # Create an empty dataframe to store the results
    results = pd.DataFrame(columns=['n', 'f1', 'f1_macro', 'recall', 'recall_macro', 'precision', 'precision_macro', 'brier_score'])

    # Initialize an empty list to store the top 5 models
    top_models = []

    # Loop through the range of n values
    for this_n in tqdm(range(start_n, end_n - 1, (-1 * stepdown))):
        # Instantiate RFE with the specified number of features to select
        selector = RFE(estimator=model, n_features_to_select=this_n, step=1)

        # Fit the selector on the training data
        selector.fit(X_train, y_train)

        # Get the selected features
        feature_index = selector.get_support()

        # Specify the selected field names
        selected_columns = X_train.columns[feature_index]

        # Select the columns from the training and testing data
        X_train_selected = X_train[selected_columns]
        X_test_selected = X_test[selected_columns]
        
        # Instantiate the model and fit it to the scaled training data
        model = model.fit(X_train_selected, y_train)

        # Predict the outcomes for test data using the trained model
        predictions = model.predict(X_test_selected)
        
        # Set brier to None if the model does not have predict_proba
        brier = None
        # Check if the model has predict_proba method and if it does calculate the probabilities for class 1
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test_selected)[:, 1]
            brier = brier_score_loss(y_test, probabilities)

        # Gather evaluation metrics for the predictions
        f1 = f1_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions)
        recall_macro = recall_score(y_test, predictions, average='macro')
        precision = precision_score(y_test, predictions)
        precision_macro = precision_score(y_test, predictions, average='macro')

        # Store the metrics in a dictionary
        metrics = {
            'n': this_n, 
            'f1': f1, 
            'f1_macro': f1_macro,
            'recall': recall, 
            'recall_macro': recall_macro,
            'precision': precision,
            'precision_macro': precision_macro,
            'brier_score': brier
            }

        # Append the metrics to the results dataframe
        results = results.append(metrics, ignore_index=True)

        # Update the top 25 models list
        top_models.append((metrics, selector))
        top_models.sort(key=lambda x: x[0][scoring], reverse=True)

        # Rename old models when a new model is ranked among the top 20
        for rank in range(len(top_models)-1, 0, -1):
            old_model_path = f"{model_path}RFE_{type(model).__name__}_{rank}.pkl"
            new_model_path = f"{model_path}RFE_{type(model).__name__}_{rank+1}.pkl"
            if os.path.exists(old_model_path):
                os.rename(old_model_path, new_model_path)

        # Save the new top 20 performing models using joblib
        for rank, (model_info, selector) in enumerate(top_models, start=1):
            n_features = model_info['n']
            jb.dump(selector, f"{model_path}RFE_{type(model).__name__}_{rank}.pkl")

    # Sort the results by accuracy in descending order
    results = results.sort_values(by=scoring, ascending=False)

    # Add an index to the sorted results
    results = results.reset_index(drop=True)

    return results


def select_k_best(X_train, X_test, y_train, y_test, start_k=44, end_k=5, score_func=mutual_info_classif, sampling=None, model=LogisticRegression(random_state=seed), scoring='f1', model_path=''):
    """
    This function finds the top 5 performing models using SelectKBest and saves them as .pkl files.
    The performance is evaluated using various metrics, such as accuracy, f1-score, recall, precision,
    brier_score, and recall_macro. The function returns a sorted DataFrame containing these evaluation
    metrics for different values of k, where k is the number of selected features in SelectKBest.

    Parameters:
    start_n (int): Initial number of features to start with.
    end_n (int): Final number of features to end with.
    model (estimator): The estimator to use for SelectKBest.
    X_train (DataFrame): The training dataset.
    X_test (DataFrame): The testing dataset.
    y_train (Series): The target variable for the training dataset.
    y_test (Series): The target variable for the testing dataset.

    Returns:
    DataFrame: Sorted dataframe containing evaluation metrics for different values of k.
    """
    # Create an empty dataframe to store the results
    results = pd.DataFrame(columns=['k', 'accuracy', 'f1', 'f1_macro', 'recall', 'recall_macro', 'precision', 'precision_macro', 'brier_score'])

    # Initialize an empty list to store the top 5 models
    top_models = []
    
    # Resample X_train and y_train
    if sampling == 'Rand Over Samp':
        X_train_resampled, y_train_resampled = random_over_sampler(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Loop through the range of k values
    for this_k in range(start_k, end_k - 1, -1):
        # Instantiate SelectKBest with the specified number of features to select
        selector = SelectKBest(score_func=score_func, k=this_k)
        

        # Create a pipeline with the SelectKBest selector and the model
        pipeline = Pipeline([
            ('selector', selector),
            ('estimator', model)
        ])
        
        # Fit the selector on the training data
        pipeline.fit(X_train_resampled, y_train_resampled)

        # Predict the outcomes for test data using the trained model
        predictions = pipeline.predict(X_test)
        
        # Set brier to None if the model does not have predict_proba
        brier = None
        # Check if the model has predict_proba method and if it does calculate the probabilities for class 1
        if hasattr(model, 'predict_proba'):
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            brier = brier_score_loss(y_test, probabilities)

        # Gather evaluation metrics for the predictions
        f1 = f1_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions)
        recall_macro = recall_score(y_test, predictions, average='macro')
        precision = precision_score(y_test, predictions)
        precision_macro = precision_score(y_test, predictions, average='macro')

        # Store the metrics in a dictionary
        metrics = {
            'n': this_n, 
            'f1': f1, 
            'f1_macro': f1_macro,
            'recall': recall, 
            'recall_macro': recall_macro,
            'precision': precision,
            'precision_macro': precision_macro,
            'brier_score': brier
            }
        # Append the metrics to the results dataframe
        results = results.append(metrics, ignore_index=True)

        # Update the top 5 models list
        top_models.append((metrics, selector))
        top_models.sort(key=lambda x: x[0][scoring], reverse=True)
        top_models = top_models[:5]

        # Save and rename the top 5 performing models
        for rank, (model_info, selector) in enumerate(top_models, start=1):
            # Remove the model at the current rank, if it exists
            old_model_path = f"{model_path}K_{type(model).__name__}_{rank}.pkl"
            if os.path.exists(old_model_path):
                os.remove(old_model_path)

          # Save the new model at the current rank
            n_features = model_info['k']
            pipeline = Pipeline([
                ('selector', SelectKBest(score_func=score_func, k=n_features)),
                ('estimator', model)
            ])
            pipeline.fit(X_train, y_train)
            jb.dump(pipeline, old_model_path)

    # Sort the results by accuracy in descending order
    results = results.sort_values(by=scoring, ascending=False)

    # Add an index to the sorted results
    results = results.reset_index(drop=True)

    return results


from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


def pipeline(param_grid, 
             best_n,
             X_train, 
             X_test, 
             y_train, 
             y_test, 
             model, 
             scoring='f1', 
             cv=5, 
             n_repeats=3, 
             classes=None, 
             model_path='xxx', 
             image_path='xxx', 
             label='Grid Search'):
    """
    Train a classification model using a pipeline with RFE feature selection and GridSearchCV for hyperparameter tuning.
    Evaluate the trained model and display the optimal hyperparameters.

    :param best_n: int, number of top features to select using RFE
    :param model: estimator, classification model to be used (e.g., LogisticRegression)
    :param param_grid: dict, grid of hyperparameters to search during grid search
    :param scoring: str, performance metric to optimize during grid search (e.g., 'f1')
    :param cv: int, number of cross-validation folds for RepeatedStratifiedKFold
    :param n_repeats: int, number of repetitions for RepeatedStratifiedKFold
    :param classes: list, class labels for the target variable
    :param model_path: str, path to save the trained model
    :param image_path: str, path to save the evaluation figures
    """
    
    # Create a pipeline object with a single step - RFE (Recursive Feature Elimination) for feature selection.
    # The RFE uses the model as the estimator for identifying the best_n most important features.
    pipe = Pipeline(steps=[
        ('selector', RFE(estimator=model, n_features_to_select=best_n, step=1))
    ])

    # Set up the GridSearchCV object with the following settings:
    # 1. The pipeline object (RFE with Extra Trees Classifier) as the estimator.
    # 2. The param_grid containing hyperparameters to be tuned.
    # 3. Cross-validation using RepeatedStratifiedKFold with cv number of splits and n_repeats repetitions.
    # 4. Return train scores for better insight into model performance.
    # 5. Optimize the model for the F1 score (scoring='f1').
    # 6. Use all available CPU cores for parallel processing (n_jobs=-1).
    # 7. Set verbosity to display messages during the grid search process (verbose=3).
    grid_search = GridSearchCV(estimator=pipe,
                               param_grid=param_grid,
                               cv=RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=seed),
                               return_train_score=True,
                               scoring=scoring,
                               n_jobs=-1,
                               verbose=3)

    # Fit the GridSearchCV object to the training data (X_train, y_train), running the grid search
    # to find the best combination of hyperparameters for the pipeline.
    grid_search.fit(X_train, y_train)

    # Retrieve the best estimator (model with optimal hyperparameters) from the grid search results.
    best_grid = grid_search.best_estimator_

    # Fit the best estimator to the training data, using the optimal hyperparameters found in the grid search.
    best_grid.fit(X_train, y_train)

    # Print the optimal parameters found by the grid search, providing insight into which hyperparameters
    # led to the best F1 score.
    print(f"\nOptimal Parameters: {grid_search.best_params_}\n")

    # Evaluate the best model's performance using a custom evaluation function (evaluate_classification).
    # This function takes the following inputs:
    # 1. The best model (best_grid).
    # 2. Training and testing data (X_train, X_test, y_train, y_test).
    # 3. Model and image paths for saving the trained model and performance plots.
    # 4. Class labels for the target variable (classes).
    # 5. A label to be used when saving the model file (e.g., "Grid Search LogisticRegression.pkl").
    evaluate_classification(best_grid, X_train, X_test, y_train, y_test, model_path=model_path, image_path=image_path, classes=classes, label=f"{label} {type(model).__name__}")
   

from sklearn.pipeline import Pipeline


def load_and_identify_model(loaded_model):
    """
    Identify the type of pickled model. Is it simply an algorithm, is it an RFE model, or an RFE Pipeline.
    
    :param loaded_model: pickled model accessed using joblib
    """
    # Check if the model is neither RFE nor Pipeline, i.e., a simple model
    if not isinstance(loaded_model, (RFE, Pipeline)):
        # Return the model type as 1 (simple model) and the model itself
        return 1, loaded_model
    
    # Check if the model is an RFE model with an estimator
    if isinstance(loaded_model, RFE):
        # Return the model type as 2 (RFE model) and the model itself
        return 2, loaded_model
    
   # Check if the model is a Pipeline
    if isinstance(loaded_model, Pipeline):
        # Check if the Pipeline contains an RFE model
        is_rfe_in_pipeline = any(isinstance(step[1], RFE) for step in loaded_model.steps)
        if is_rfe_in_pipeline:
            # Return the model type as 3 (Pipeline with RFE model) and the model itself
            return 3, loaded_model
        # Check if the Pipeline contains a SelectKBest model
        is_selectkbest_in_pipeline = any(isinstance(step[1], SelectKBest) for step in loaded_model.steps)
        if is_selectkbest_in_pipeline:
            # Return the model type as 4 (Pipeline with SelectKBest model) and the model itself
            return 4, loaded_model
        else:
            # Raise an error if the Pipeline does not contain an RFE or SelectKBest model
            raise ValueError("The pipeline does not contain an RFE or SelectKBest model.")
    
    # Raise an error if the model is not one of the expected types (simple, RFE, Pipeline with RFE or SelectKBest)
    raise ValueError("The loaded model is not one of the expected types.")    
    
    
def plot_coefficients_seaborn(X_train, y_train, figure_size=(16, 16), model_path='xxx', image_path='xxx', label=None):
    """
    Create a bar plot of model coefficients using Seaborn.
    
    :param X_train: DataFrame, training feature set
    :param y_train: Series, training target set (unused in the function)
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the coefficients figure
    :param label: str, descriptive label for the model
    """

    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
    elif model_type == 2:
        model = loaded_model.estimator_
        selected_features = loaded_model.get_support(indices=True)
        X_train = X_train.iloc[:, selected_features]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                selected_features = step_model.get_support(indices=True)
                X_train = X_train.iloc[:, selected_features]
    elif model_type == 4:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, SelectKBest):
                model = loaded_model.named_steps['estimator']
                selected_features = step_model.get_support(indices=True)
                X_train = X_train.iloc[:, selected_features]

    # Set the plot and figure size
    fig, ax = plt.subplots(figsize=(figure_size[0], figure_size[1]))

    # Create a DataFrame containing the names and values of the coefficients
    coefficients = pd.DataFrame({'Coefficient Name': X_train.columns, 'Coefficient Value': model.coef_.ravel().flatten()})
    
    # Sort the coefficients by value in descending order
    coefficients = coefficients.sort_values(by='Coefficient Value', ascending=False)

    # Create a bar plot of the coefficients using Seaborn
    sns.set(style="dark")
    sns.barplot(y='Coefficient Name', x='Coefficient Value', data=coefficients, ax=ax, palette="mako")

    # Set the plot title and axis labels
    ax.set_title(f"{label} Coefficients")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("")
    ax.set_yticklabels(coefficients['Coefficient Name'])

    # Save the figure to a file with a descriptive name
    plt.tight_layout()
    plt.savefig(f"{image_path}{label.replace(' ', '_')}_Coefficients.png")

    # Display the plot on the screen
    plt.show()


def plot_coefficients_plotly(X_train, y_train, figure_size=(2000, 1000), model_path='xxx', image_path='xxx', label=None):
    """
    Create a bar plot of model coefficients using Plotly.
    
    :param X_train: DataFrame, training feature set
    :param y_train: Series, training target set (unused in the function)
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the coefficients figure
    :param label: str, descriptive label for the model
    """

    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
    elif model_type == 2:
        model = loaded_model.estimator_
        selected_features = loaded_model.get_support(indices=True)
        X_train = X_train.iloc[:, selected_features]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                selected_features = step_model.get_support(indices=True)
                X_train = X_train.iloc[:, selected_features]
    elif model_type == 4:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, SelectKBest):
                model = loaded_model.named_steps['estimator']
                selected_features = step_model.get_support(indices=True)
                X_train = X_train.iloc[:, selected_features]

    # Get the coefficients from the model and sort them by value
    # Create a dataframe with the names and values of the coefficients
    coefficients = pd.DataFrame({'Coefficient Name': X_train.columns, 'Coefficient Value': model.coef_.ravel().flatten()})
    # Sort the coefficients by value in descending order
    coefficients = coefficients.sort_values(by='Coefficient Value', ascending=True)

    # # Create a blue color scale using Plotly's colorscale functions
    # colorscale = [[0, '#073f6e'], [0.2, '#0a619c'], [0.4, '#136ebd'], [0.6, '#2a93d5'], [0.8, '#4cb3d9'], [1, '#b7e3e4']]

    # Create a horizontal bar chart using Plotly with a bigger figure size
    fig = go.Figure(go.Bar(
                x=coefficients['Coefficient Value'],
                y=coefficients['Coefficient Name'],
                orientation='h'
                # marker=dict(color=coefficients['Coefficient Value'],
                #             colorbar=dict(title='Coefficient Value'),
                #             colorscale=colorscale,
                #             cmin=coefficients['Coefficient Value'].min(),
                #             cmax=coefficients['Coefficient Value'].max())
                ))
    fig.update_layout(title=f"{label} Coefficients",
                      xaxis_title='Coefficient Value',
                      yaxis_title='',
                      width=figure_size[0],
                      height=figure_size[1])
    
    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}_Coefficients.png"
    pio.write_image(fig, filename)

    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Coefficients.html", auto_open=True)

    # Show the plot
    fig.show()


def coefficients(X_train, y_train, model_path='xxx', data_path='xxx', label=None):
    """
    Create a bar plot of model coefficients using Seaborn.
    
    :param X_train: DataFrame, training feature set
    :param y_train: Series, training target set (unused in the function)
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the coefficients figure
    :param label: str, descriptive label for the model
    """

    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
    elif model_type == 2:
        model = loaded_model.estimator_
        selected_features = loaded_model.get_support(indices=True)
        X_train = X_train.iloc[:, selected_features]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                selected_features = step_model.get_support(indices=True)
                X_train = X_train.iloc[:, selected_features]
    elif model_type == 4:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, SelectKBest):
                model = loaded_model.named_steps['estimator']
                selected_features = step_model.get_support(indices=True)
                X_train = X_train.iloc[:, selected_features]

    # Create a DataFrame containing the names and values of the coefficients
    coefficients = pd.DataFrame({'Coefficient Name': X_train.columns, 'Coefficient Value': model.coef_.ravel().flatten()})
    
    # Add intercept to the DataFrame
    intercept_df = pd.DataFrame({'Coefficient Name': ['Intercept'], 'Coefficient Value': [model.intercept_]})
    coefficients = pd.concat([intercept_df, coefficients], ignore_index=True)
    
    # Sort the coefficients by value in descending order
    coefficients = coefficients.sort_values(by='Coefficient Value', ascending=False)
    
    coefficients.to_csv(f"{data_path}{label}_Coefficients.csv", index=False)
    
    return coefficients


def plot_feature_importances_seaborn(X_train, y_train, figure_size=(16, 16), model_path='xxx', image_path='xxx', label=None):
    """
    Create a bar plot of feature importances using Seaborn.
    
    :param X_train: DataFrame, training feature set
    :param y_train: Series, training target set (unused in the function)
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the feature importances figure
    :param label: str, descriptive label for the model
    """

    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
    elif model_type == 2:
        model = loaded_model.estimator_
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
    elif model_type == 4:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, SelectKBest):
                model = loaded_model.named_steps['estimator']
                break

    # Extract feature importances from the trained model
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Convert importances to percentages
    importances_percentages = importances[indices] * 100

    # Set the style of the plot background to dark
    sns.set(style="dark")

    # Set the figure size
    plt.figure(figsize=(figure_size[0], figure_size[1]))

    # Create a bar plot using Seaborn with sorted feature importances in percentages
    sns.barplot(y=[X_train.columns[i] for i in indices], x=importances_percentages, palette="viridis")

    # Set the plot title and axis labels
    plt.title(f"Feature Importances - {label}")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")

    # Save the figure to a file with a descriptive name
    plt.savefig(f"{image_path}{label.replace(' ', '_')}_Feature_Importances.png")

    # Display the plot on the screen
    plt.show()
    
    return len(indices)

    
from sklearn.inspection import permutation_importance


def plot_permutation_importances_seaborn(X_train, y_train, figure_size=(16, 16), model_path='xxx', image_path='xxx', label=None):
    """
    Create a bar plot of permutation importances using Seaborn.
    
    :param X_train: DataFrame, training feature set
    :param y_train: Series, training target set
    :param figure_size: tuple, size of the output figure
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the permutation importances figure
    :param label: str, descriptive label for the model
    """

    # Load the pickled model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")
    selected_features = jb.load(f"{model_path}{label.replace(' ', '_')}_Selected_Features.pkl")
    
    X_train_selected = X_train[selected_features]

    # Calculate permutation importances
    result = permutation_importance(model, X_train_selected, y_train, n_repeats=10, random_state=0)

    # Sort permutation importances in descending order
    indices = np.argsort(result.importances_mean)[::-1]

    # Convert importances to percentages
    importances_percentages = result.importances_mean[indices] * 100

    # Set the style of the plot background to dark
    sns.set(style="dark")

    # Set the figure size
    plt.figure(figsize=(figure_size[0], figure_size[1]))

    # Create a bar plot using Seaborn with sorted permutation importances in percentages
    sns.barplot(y=[X_train.columns[i] for i in indices], x=importances_percentages, palette="mako")

    # Set the plot title and axis labels
    plt.title(f"Permutation Importances - {label}")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")

    # Save the figure to a file with a descriptive name
    plt.savefig(f"{image_path}{label.replace(' ', '_')}_Permutation_Importances.png")

    # Display the plot on the screen
    plt.show()
    
    
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
    
    
def grid_search(
    X_train,
    y_train, 
    X_test, 
    y_test,
    best_n, 
    param_distributions,
    model=LogisticRegression(random_state=seed), 
    scoring='f1', 
    cv=5, 
    n_repeats=3, 
    classes=None, 
    model_path='xxx', 
    image_path='xxx', 
    label='Grid Search'
):
    """
    Train a classification model using a pipeline with RFE feature selection and GridSearchCV for hyperparameter tuning. Evaluate the trained model and display the optimal hyperparameters.
        
    :param best_n: int, number of top features to select using RFE
    :param model: estimator, classification model to be used (e.g., LogisticRegression)
    :param param_grid: dict, grid of hyperparameters to search during grid search
    :param scoring: str, performance metric to optimize during grid search (e.g., 'f1')
    :param cv: int, number of cross-validation folds for RepeatedStratifiedKFold
    :param n_repeats: int, number of repetitions for RepeatedStratifiedKFold
    :param classes: list, class labels for the target variable
    :param model_path: str, path to save the trained model
    :param image_path: str, path to save the evaluation figures
    """
    
    # Create a pipeline object with a single step - RFE (Recursive Feature Elimination) for feature selection.
    # The RFE uses the model as the estimator for identifying the best_n most important features.
    pipe = Pipeline(steps=[
        ('selector', RFE(estimator=model, n_features_to_select=best_n, step=1))
    ])

    # Set up the GridSearchCV object with the following settings:
    # 1. The pipeline object (RFE with Classifier) as the estimator.
    # 2. The param_grid containing hyperparameters to be tuned.
    # 3. Cross-validation using RepeatedStratifiedKFold with cv number of splits and n_repeats repetitions.
    # 4. Return train scores for better insight into model performance.
    # 5. Optimize the model for the selected score (scoring='f1').
    # 6. Use all available CPU cores for parallel processing (n_jobs=-1).
    # 7. Set verbosity to display messages during the grid search process (verbose=3).
    grid_search = RandomizedSearchCV(estimator=pipe,
                               param_distributions=param_distributions,
                               cv=RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=seed),
                               return_train_score=True,
                               scoring=scoring,
                               n_jobs=-1,
                               verbose=1)

    # Fit the GridSearchCV object to the training data (X_train, y_train), running the grid search
    # to find the best combination of hyperparameters for the pipeline.
    grid_search.fit(X_train, y_train)

    # Retrieve the best estimator (model with optimal hyperparameters) from the grid search results.
    best_grid = grid_search.best_estimator_

    # Fit the best estimator to the training data, using the optimal hyperparameters found in the grid search.
    best_grid.fit(X_train, y_train)

    # Print the optimal parameters found by the grid search, providing insight into which hyperparameters
    # led to the best F1 score.
    print(f"\nOptimal Parameters: {grid_search.best_params_}\n")

    # Evaluate the best model's performance using a custom evaluation function (evaluate_classification).
    # This function takes the following inputs:
    # 1. The best model (best_grid).
    # 2. Training and testing data (X_train, X_test, y_train, y_test).
    # 3. Model and image paths for saving the trained model and performance plots.
    # 4. Class labels for the target variable (classes).
    # 5. A label to be used when saving the model file (e.g., "Grid Search LogisticRegression.pkl").
    evaluate_classification(best_grid, X_train, X_test, y_train, y_test, model_path=model_path, image_path=image_path, classes=classes, label=f"{label} {type(model).__name__}")

    
def decile_bin_analysis(y_true, y_probs, tracked_units = 'target subpopulation leads'.title(), positive_case_label = 'positive cases'.title(), include_potential=False):
    '''
    Creates a bin analysis broken out by population deciles where observations are rank-ordered by score (y-prob).
    
    Returns a dataframe -- run the function without assignment to display the dataframe, or else assign the function
    to a variable to continue working with the dataframe.
    
    Inputs:
        `y_true`: the actual labels for the population of cases
        `y_probs`: the probability score that the observation is in the positive clase for the target variable
          --should be the result of .predict_proba()[:,1]
    '''
    
    decile_df = pd.DataFrame({
      'score': y_probs, 
      'label': y_true
    })
    
    decile_df['decile'] = pd.qcut(decile_df['score'], q=10, labels=False)
    
    aggregated_df = decile_df.groupby('decile').agg({'score': ['min', 'max'], 'label': ['sum'], 'decile': ['size']})
    aggregated_df.columns = aggregated_df.columns.map('_'.join)
    aggregated_df = aggregated_df.rename(columns={'decile_size': tracked_units, 'label_sum': positive_case_label})
    deciles = aggregated_df.index
    
    #retrieve cumulative counts of positive cases by decile
    cumulative_count = 0
    for decile in deciles:
        cumulative_count += aggregated_df.loc[decile, positive_case_label]
        aggregated_df.loc[decile, 'Cumulative ' + positive_case_label] = cumulative_count
        
    aggregated_df['Cumulative Percent of ' + positive_case_label] = aggregated_df['Cumulative ' + positive_case_label]/aggregated_df['Cumulative ' + positive_case_label].max()
    
    return aggregated_df.sort_values(by='decile', ascending=False)


def plot_decile_bin_analysis(decile_bin_analysis_df, save_name, positive_case_label = 'positive cases'.title()):
    decile_bin_analysis_df.sort_index(ascending=False)
    deciles = sorted(decile_bin_analysis_df.index)
    decile_labs = [str(x) + '0th percentile' if x > 0 else '0th percentile' for x in deciles]
    decile_labs[-1] =  decile_labs[-1] + " and higher"
    min_scores = decile_bin_analysis_df['score_min'][::-1]
    max_scores = decile_bin_analysis_df['score_max'][::-1]
    cumulative_successes = decile_bin_analysis_df['Cumulative Percent of ' + positive_case_label][::-1]
    sns.set(style='darkgrid')
    plt.figure(figsize=(10,6), tight_layout=True)
    plt.plot(min_scores,deciles,label='Minimum Score')
    plt.plot(max_scores,deciles,label='Maximum Score')
    plt.plot(cumulative_successes, deciles, label='Cumulative Percent of ' + positive_case_label, linestyle=':', alpha=0.75)

    plt.xlabel('Model Score / Cum. % of ' + positive_case_label)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Decile Bin')


    plt.yticks(deciles, decile_labs)
    plt.yticks
    plt.title('Decile-based Bin Analysis: ' + positive_case_label)

    plt.legend()
    plt.savefig(save_name)
    plt.show()
    
    return None


def pull_samples(scores_file, target_column, label, number_samples, probability_column, threshold, samples_file, random_state):
    # load the CSV data into a pandas DataFrame
    samples = pd.read_csv(scores_file)
    
    # Filder the DataFrame based on the label and threshold
    samples_filtered = samples[(samples[target_column] == label) & \
                               (samples[probability_column] >= threshold) & \
                               (samples[probability_column] < threshold + 1) & \
                               (samples['start_date_time'] > '2019-01-01') & \
                               (samples['max_balance_in_period'] > 5_000) & \
                               (samples['max_balance_in_period'] < 25_000) & \
                               (samples['latest_rtn_ca_agi_amt_in_period'] > 0) & \
                               ((samples['latest_rtn_sch_c_inc_amt_in_period'] > 0) | (samples['latest_rtn_sch_e_inc_amt_in_period'] > 0))]
    
    
    # If the filtered DataFrame has less samples than required, raise an error
    if samples_filtered.shape[0] < number_samples:
        raise ValueError(f" Not enough samples witin the specified range. Only found {samples_filtered.shape[0]} samples.")
    
    # Randomly sample the specified number of rows from the filtered DataFrame
    samples = samples_filtered.sample(n=number_samples, random_state=random_state)
    
    # Save the samples DataFrame to a new CSV file
    samples.to_csv(samples_file, index=False)
    
    return samples.transpose()


def pull_samples_2(scores_file, target_column, label, number_samples, probability_column, threshold, samples_file, random_state):
    # load the CSV data into a pandas DataFrame
    samples = pd.read_csv(scores_file)
    
    # Filder the DataFrame based on the label and threshold
    samples_filtered = samples[(samples[target_column] == label) & \
                               (samples[probability_column] >= threshold) & \
                               (samples[probability_column] < threshold + 1) & \
                               (samples['start_date_time'] > '2021-01-01') & \
                               (samples['case_max_open_balance'] > 3_000)]
    
    
    # If the filtered DataFrame has less samples than required, raise an error
    if samples_filtered.shape[0] < number_samples:
        raise ValueError(f" Not enough samples witin the specified range. Only found {samples_filtered.shape[0]} samples.")
    
    # Randomly sample the specified number of rows from the filtered DataFrame
    samples = samples_filtered.sample(n=number_samples, random_state=random_state)
    
    # Save the samples DataFrame to a new CSV file
    samples.to_csv(samples_file, index=False)
    
    return samples.transpose()


def shapley_values_plotly_waterfall_2(
      identifiers,
      identifier_cols,
      selected_features,
      df,
      X_test_clean,
      X_test, 
      y_test, 
      sample_size=1, 
      figsize=(2000, 1100), 
      threshold=0.9, 
      prob_class=1, 
      actual_class=1, 
      model_path='', 
      image_path='', 
      label=None, 
      seed=42
):
    """
    Create a waterfall plot of the Shapley values using Plotly for one random sample with a specified probability threshold of being classified as 1.
    
    :param X_test: DataFrame, testing feature set
    :param y_test: Series, testing target set
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the Shapley values figure
    :param label: str, descriptive label for the model
    :param seed: int, random seed for reproducibility
    :param threshold: float, probability threshold for classification as 1
    """
    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
        # Get the dictionary of column names and indices
        column_dict = {col_name: i for i, col_name in enumerate(X_test.columns)}
        # Compare the feature names in the model with the dictionary keys
        model_indices = [column_dict[feature_name] for feature_name in model.feature_names_in_ if feature_name in column_dict]
        X_test_transformed = X_test.iloc[:, model_indices]
    elif model_type == 2:
        model = loaded_model.estimator_
        # Apply RFE transformation to the testing data
        X_test_transformed = X_test.iloc[:, loaded_model.get_support()]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                # Apply RFE transformation to the testing data
                X_test_transformed = X_test.iloc[:, step_model.get_support()]
    elif model_type == 4:
      for step_name, step_model in loaded_model.named_steps.items():
        if isinstance(step_model, SelectKBest):
          model = loaded_model.named_steps['estimator']
          # Apply SelectKBest transformation to the testing data
          feature_index = step_model.get_support()
          X_test_transformed = X_test.iloc[:, feature_index]
          break

    # Predict the probabilities for each sample in X_test_transformed
    probabilities = model.predict_proba(X_test_transformed)[:, prob_class]

    # Create a boolean mask for samples with probabilities within the range [threshold, threshold + 0.1)
    mask = pd.Series(index=X_test.index, data=((probabilities >= threshold) & (probabilities < threshold + 0.1)))

    # Filter the samples in X_test that meet the condition
    filtered_samples = X_test[mask]

    # Filter the samples in y_test that are classified as 1
    filtered_labels = y_test[mask]

    # If no samples meet the condition, raise an error
    if filtered_samples.empty:
        raise ValueError("No samples found with the specified probability threshold")

    # Filter the samples with a label of 1
    positive_samples = filtered_samples[filtered_labels == actual_class]

    # If no samples with a label of 1 are found, raise an error
    if positive_samples.empty:
        raise ValueError("No samples found with the specified probability threshold and classified as 1")

    # If model_type is an RFE Pipeline then call the get_suport() method on step_model to access the features the model used
    if model_type == 3 or model_type == 4:
        # Apply RFE transformation to the positive_samples
        positive_samples_transformed = positive_samples.iloc[:, step_model.get_support()]
    elif model_type == 2:
      positive_samples_transformed = positive_samples.iloc[:, loaded_model.get_support()]
    else:
        # Apply RFE transformation to the positive_samples
        positive_samples_transformed = positive_samples.iloc[:, model_indices]

    # Randomly select one instance from the positive_samples_transformed
    chosen_sample = positive_samples_transformed.sample(n=sample_size, random_state=seed)

    # Convert the selected sample into a DataFrame and reset the index
    chosen_sample_df = pd.DataFrame(chosen_sample)

    # Create a Shapley explainer object using the appropriate explainer class based on the base estimator
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression)):
        explainer = shap.LinearExplainer(model, X_test_transformed)
    # Create a Shapley explainer object using the appropriate explainer class based on the base estimator
    else:
        # Use KernelExplainer as a fallback option for other model types
        explainer = shap.KernelExplainer(model.predict_proba, X_test_transformed.sample(n=sample_size, random_state=seed))

    # Compute the Shapley values for the chosen sample
    shap_values = explainer.shap_values(chosen_sample)

    # Create a DataFrame containing the feature names and their corresponding Shapley values
    if len(shap_values) > 1:
      if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        shap_values_to_use = shap_values[1].flatten()
      elif isinstance(model, (LogisticRegression)):
        shap_values_to_use = shap_values[0].flatten()
    # This will handle the case when shap_values has only one array   
    else: 
      shap_values_to_use = shap_values[0].flatten()

    
    # Create a DataFrame containing the feature names and their corresponding Shapley values
    shap_df = pd.DataFrame({"Feature": chosen_sample.columns, "Shapley Value": shap_values_to_use})

    # Sort the DataFrame by ascending Shapley value
    shap_df.sort_values("Shapley Value", ascending=False, inplace=True)

    # Calculate the total absolute sum of Shapley values
    total_shap_sum = np.abs(shap_df["Shapley Value"]).sum()

    # Normalize the Shapley values and convert them to percentages
    shap_df["Shapley Value (%)"] = (shap_df["Shapley Value"] / total_shap_sum) * 100

    # Create a waterfall plot of the Shapley values using Plotly
    fig = go.Figure(go.Waterfall(
        x=shap_df["Feature"],
        y=shap_df["Shapley Value (%)"],
        textposition="outside",
        text=shap_df["Shapley Value (%)"].apply(lambda x: f'{x:+,.2f}%'),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        measure=["relative"] * len(shap_df),
        base=0,
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}}
    ))

    # Update the layout of the plot
    fig.update_layout(
        title="Shapley Values Waterfall Plot for a Single Sample (Percentages)",
        xaxis_title="Feature",
        yaxis_title="Shapley Value (%)",
        width=figsize[0],
        height=figsize[1],
        xaxis=dict(tickmode='array', tickvals=list(shap_df.index), ticktext=list(shap_df["Feature"])),
        xaxis_tickangle=-90,
        margin=dict(l=100, r=100, t=100, b=100)
    )

    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}_Waterfall.png"
    pio.write_image(fig, filename)
    
    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Waterfall.html", auto_open=True)

    fig.show()

    X_test_clean_sample = X_test_clean.loc[chosen_sample_df.index]

    X_test_clean_sample = X_test_clean_sample[chosen_sample_df.columns]
    
    identifiers_sample = identifiers.loc[chosen_sample_df.index]

    chosen_sample_df = pd.concat([identifiers_sample, X_test_clean_sample], axis=1)

    # Load the dataframe
    df = pd.read_csv(df)
    
    # Filted DataFrame to selected features of interest or target features that couldn't be used in the model
    X_selected = df[selected_features]
    
    # Merge df and X_test_w_ids using the intendifiers
    chosen_sample_df = chosen_sample_df.merge(X_selected, on=identifier_cols)

    chosen_sample_df
    
    return chosen_sample_df.transpose()


def shapley_values_plotly_waterfall(
      identifiers,
      ss,
      X_test, 
      y_test, 
      sample_size=1, 
      figsize=(2000, 1100), 
      threshold=0.9, 
      prob_class=1, 
      actual_class=1, 
      model_path='', 
      image_path='', 
      label=None, 
      seed=42
):
    """
    Create a waterfall plot of the Shapley values using Plotly for one random sample with a specified probability threshold of being classified as 1.
    
    :param X_test: DataFrame, testing feature set
    :param y_test: Series, testing target set
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the Shapley values figure
    :param label: str, descriptive label for the model
    :param seed: int, random seed for reproducibility
    :param threshold: float, probability threshold for classification as 1
    """
    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
        # Get the dictionary of column names and indices
        column_dict = {col_name: i for i, col_name in enumerate(X_test.columns)}
        # Compare the feature names in the model with the dictionary keys
        model_indices = [column_dict[feature_name] for feature_name in model.feature_names_in_ if feature_name in column_dict]
        X_test_transformed = X_test.iloc[:, model_indices]
    elif model_type == 2:
        model = loaded_model.estimator_
        # Apply RFE transformation to the testing data
        X_test_transformed = X_test.iloc[:, loaded_model.get_support()]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                # Apply RFE transformation to the testing data
                X_test_transformed = X_test.iloc[:, step_model.get_support()]
    elif model_type == 4:
      for step_name, step_model in loaded_model.named_steps.items():
        if isinstance(step_model, SelectKBest):
          model = loaded_model.named_steps['estimator']
          # Apply SelectKBest transformation to the testing data
          feature_index = step_model.get_support()
          X_test_transformed = X_test.iloc[:, feature_index]
          break

    # Predict the probabilities for each sample in X_test_transformed
    probabilities = model.predict_proba(X_test_transformed)[:, prob_class]

    # Create a boolean mask for samples with probabilities within the range [threshold, threshold + 0.1)
    mask = pd.Series(index=X_test.index, data=((probabilities >= threshold) & (probabilities < threshold + 0.1)))

    # Filter the samples in X_test that meet the condition
    filtered_samples = X_test[mask]

    # Filter the samples in y_test that are classified as 1
    filtered_labels = y_test[mask]

    # If no samples meet the condition, raise an error
    if filtered_samples.empty:
        raise ValueError("No samples found with the specified probability threshold")

    # Filter the samples with a label of 1
    positive_samples = filtered_samples[filtered_labels == actual_class]

    # If no samples with a label of 1 are found, raise an error
    if positive_samples.empty:
        raise ValueError("No samples found with the specified probability threshold and classified as 1")

    # If model_type is an RFE Pipeline then call the get_suport() method on step_model to access the features the model used
    if model_type == 3 or model_type == 4:
        # Apply RFE transformation to the positive_samples
        positive_samples_transformed = positive_samples.iloc[:, step_model.get_support()]
    elif model_type == 2:
      positive_samples_transformed = positive_samples.iloc[:, loaded_model.get_support()]
    else:
        # Apply RFE transformation to the positive_samples
        positive_samples_transformed = positive_samples.iloc[:, model_indices]

    # Randomly select one instance from the positive_samples_transformed
    chosen_sample = positive_samples_transformed.sample(n=sample_size, random_state=seed)

    # Convert the selected sample into a DataFrame and reset the index
    chosen_sample_df = pd.DataFrame(chosen_sample)

    # Create a Shapley explainer object using the appropriate explainer class based on the base estimator
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression)):
        explainer = shap.LinearExplainer(model, X_test_transformed)
    # Create a Shapley explainer object using the appropriate explainer class based on the base estimator
    else:
        # Use KernelExplainer as a fallback option for other model types
        explainer = shap.KernelExplainer(model.predict_proba, X_test_transformed.sample(n=sample_size, random_state=seed))

    # Compute the Shapley values for the chosen sample
    shap_values = explainer.shap_values(chosen_sample)

    # Create a DataFrame containing the feature names and their corresponding Shapley values
    if len(shap_values) > 1:
      if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        shap_values_to_use = shap_values[1].flatten()
      elif isinstance(model, (LogisticRegression)):
        shap_values_to_use = shap_values[0].flatten()
    # This will handle the case when shap_values has only one array   
    else: 
      shap_values_to_use = shap_values[0].flatten()

    
    # Create a DataFrame containing the feature names and their corresponding Shapley values
    shap_df = pd.DataFrame({"Feature": chosen_sample.columns, "Shapley Value": shap_values_to_use})

    # Sort the DataFrame by ascending Shapley value
    shap_df.sort_values("Shapley Value", ascending=False, inplace=True)

    # Calculate the total absolute sum of Shapley values
    total_shap_sum = np.abs(shap_df["Shapley Value"]).sum()

    # Normalize the Shapley values and convert them to percentages
    shap_df["Shapley Value (%)"] = (shap_df["Shapley Value"] / total_shap_sum) * 100

    # Create a waterfall plot of the Shapley values using Plotly
    fig = go.Figure(go.Waterfall(
        x=shap_df["Feature"],
        y=shap_df["Shapley Value (%)"],
        textposition="outside",
        text=shap_df["Shapley Value (%)"].apply(lambda x: f'{x:+,.2f}%'),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        measure=["relative"] * len(shap_df),
        base=0,
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}}
    ))

    # Update the layout of the plot
    fig.update_layout(
        title="Shapley Values Waterfall Plot for a Single Sample (Percentages)",
        xaxis_title="Feature",
        yaxis_title="Shapley Value (%)",
        width=figsize[0],
        height=figsize[1],
        xaxis=dict(tickmode='array', tickvals=list(shap_df.index), ticktext=list(shap_df["Feature"])),
        xaxis_tickangle=-90,
        margin=dict(l=100, r=100, t=100, b=100)
    )

    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}_Waterfall.png"
    pio.write_image(fig, filename)
    
    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Waterfall.html", auto_open=True)

    fig.show()

    discrete_cols = [col for col in X_test.columns.to_list() if 'ind' in col or 'initial_asmt_type' in col or 'latest_rtn_filing_status' in col]

    continuous_cols = X_test.drop(columns=discrete_cols).columns
    
    X_test_reverse = pd.DataFrame(ss.inverse_transform(X_test[continuous_cols]), columns=X_test[continuous_cols].columns, index=X_test.index)

    X_test_reverse_sample = X_test_reverse.loc[chosen_sample_df.index]

    discrete_cols = [col for col in chosen_sample_df.columns.to_list() if 'ind' in col or 'initial_asmt_type' in col or 'latest_rtn_filing_status' in col]

    continuous_cols = chosen_sample_df.drop(columns=discrete_cols).columns

    chosen_sample_df = pd.concat([X_test_reverse_sample[continuous_cols], chosen_sample_df.drop(columns=continuous_cols).copy(deep=True)], axis=1)

    identifiers_sample = identifiers.loc[chosen_sample_df.index]

    chosen_sample_df = pd.concat([identifiers_sample, chosen_sample_df], axis=1)
    
    return chosen_sample_df.transpose()


def shapley_values_plotly_bar(
        X_test, 
        sample_size, 
        figsize=(1600, 1100),
        model_path='xxx', 
        image_path='xxx', 
        label=None, 
        seed=42
):
    """
    Create a bar plot of the mean absolute Shapley values using Plotly.
    
    :param X_test: DataFrame, testing feature set
    :param sample_size: float, proportion of X_test to use for computing SHAP values
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the Shapley values figure
    :param label: str, descriptive label for the model
    :param seed: int, random seed for reproducibility
    """

    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transforme d test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
        # Get the dictionary of column names and indices
        column_dict = {col_name: i for i, col_name in enumerate(X_test.columns)}
        # Compare the feature names in the model with the dictionary keys
        model_indices = [column_dict[feature_name] for feature_name in model.feature_names_in_ if feature_name in column_dict]
        X_test_transformed = X_test.iloc[:, model_indices]
    elif model_type == 2:
        model = loaded_model.estimator_
        # Apply RFE transformation to the testing data
        X_test_transformed = X_test.iloc[:, loaded_model.get_support()]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                # Apply RFE transformation to the testing data
                X_test_transformed = X_test.iloc[:, step_model.get_support()]
    elif model_type == 4:
      for step_name, step_model in loaded_model.named_steps.items():
        if isinstance(step_model, SelectKBest):
          model = loaded_model.named_steps['estimator']
          # Apply SelectKBest transformation to the testing data
          feature_index = step_model.get_support()
          X_test_transformed = X_test.iloc[:, feature_index]
          break

    # Calculate the sample size equal to 10% of X_test
    sample_size = sample_size

    # Randomly sample 10% of X_test_transformed using the specified seed for reproducibility
    sample = X_test_transformed.sample(n=sample_size, random_state=seed)

    # Create a Shapley explainer object using the appropriate explainer class based on the base estimator
    if isinstance(model, (GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression, )):
        explainer = shap.LinearExplainer(model, X_test_transformed)
    else:
        # Use KernelExplainer as a fallback option for other model types
        explainer = shap.KernelExplainer(model.predict_proba, X_test_transformed.sample(n=sample_size, random_state=seed))

    # Compute the Shapley values for the testing data using the shap_values() function
    shap_values = explainer.shap_values(sample)

    # Get the mean absolute Shapley values for each feature
    if isinstance(model, (LGBMClassifier)):
        mean_shap = np.abs(shap_values[1]).mean(axis=0)
    elif isinstance(model, (LogisticRegression, GradientBoostingClassifier, XGBClassifier, CatBoostClassifier)):
        mean_shap = np.abs(shap_values).mean(axis=0)
    else:
        raise ValueError("Unsupported model type")

    # Calculate the total sum of mean absolute Shapley values
    total_mean_shap = np.sum(mean_shap)

    # Calculate the percentage contribution of each feature
    mean_shap_percentages = (mean_shap / total_mean_shap) * 100

    # Filter out features with zero mean absolute Shapley values
    nonzero_mean_shap_percentages = mean_shap_percentages[mean_shap_percentages > 0]

    # Sort the features by ascending mean Shapley value
    feature_order = np.argsort(nonzero_mean_shap_percentages)

    # Create a bar plot of the non-zero mean absolute Shapley values using Plotly
    fig = go.Figure()
    
    # Add a bar plot trace for the sorted non-zero mean absolute Shapley values
    fig.add_trace(go.Bar(x=nonzero_mean_shap_percentages[feature_order],
                         y=sample.columns[feature_order],
                         orientation='h', 
                         marker=dict(color=nonzero_mean_shap_percentages[feature_order], colorscale='RdBu_r')))
    
    # Customize the appearance of the plot
    fig.update_layout(title="SHAP Summary Plot", xaxis_title="Mean Absolute Shapley Value (Percentage)", yaxis_title="Feature",
                      yaxis=dict(tickmode='array', tickvals=list(sample.columns[feature_order]),
                                 ticktext=list(sample.columns[feature_order]),
                                 tickfont=dict(size=12), tickangle=0),
                      height=figsize[1], width=figsize[0], margin=dict(l=100, r=100, t=100, b=100))
    
    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}_Shapley_Values_Bar.png"
    pio.write_image(fig, filename)

    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Shapley_Values_Bar.html", auto_open=True)

    # Display the plot
    fig.show()


def shapley_values_plotly_scatter(
        X_test, 
        sample_size, 
        figsize=(1600, 1100), 
        model_path='xxx', 
        image_path='xxx', 
        label=None, 
        seed=42
):
    """
    Create a scatter plot of the Shapley values using Plotly.
    
    :param X_test: DataFrame, testing feature set
    :param sample_size: float, proportion of X_test to use for computing SHAP values
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the Shapley values figure
    :param label: str, descriptive label for the model
    :param seed: int, random seed for reproducibility
    """

    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
        # Get the dictionary of column names and indices
        column_dict = {col_name: i for i, col_name in enumerate(X_test.columns)}
        # Compare the feature names in the model with the dictionary keys
        model_indices = [column_dict[feature_name] for feature_name in model.feature_names_in_ if feature_name in column_dict]
        X_test_transformed = X_test.iloc[:, model_indices]
    elif model_type == 2:
        model = loaded_model.estimator_
        # Apply RFE transformation to the testing data
        X_test_transformed = X_test.iloc[:, loaded_model.get_support()]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                # Apply RFE transformation to the testing data
                X_test_transformed = X_test.iloc[:, step_model.get_support()]
    elif model_type == 4:
      for step_name, step_model in loaded_model.named_steps.items():
        if isinstance(step_model, SelectKBest):
          model = loaded_model.named_steps['estimator']
          # Apply SelectKBest transformation to the testing data
          feature_index = step_model.get_support()
          X_test_transformed = X_test.iloc[:, feature_index]
          break

    # Calculate the sample size equal to 10% of X_test
    sample_size = sample_size

    # Randomly sample 10% of X_test_transformed using the specified seed for reproducibility
    sample = X_test_transformed.sample(n=sample_size, random_state=seed)

    # Create a Shapley explainer object using the appropriate explainer class based on the base estimator
    if isinstance(model, (GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression)):
        explainer = shap.LinearExplainer(model, X_test_transformed)
    else:
        # Use KernelExplainer as a fallback option for other model types
        explainer = shap.KernelExplainer(model.predict_proba, X_test_transformed.sample(n=sample_size, random_state=seed))

    # Compute the Shapley values for the testing data using the shap_values() function
    shap_values = explainer.shap_values(sample)

    # Handle CatBoost separately
    if isinstance(model, (LogisticRegression, GradientBoostingClassifier, XGBClassifier, CatBoostClassifier)):
      # Here, shap_values is already a single array, not a list of two arrays
      total_shap_sums = np.abs(shap_values).sum(axis=1)
      shap_values_percent = (shap_values / total_shap_sums[:, None]) * 100
    else:
      # For other models, we use shap_values[1]
      total_shap_sums = np.abs(shap_values[1]).sum(axis=1)
      shap_values_percent = (shap_values[1] / total_shap_sums[:, None]) * 100

    # Sort the features by ascending mean Shapley value
    feature_order = np.argsort(np.abs(shap_values_percent).mean(axis=0))

    # Create a violin plot of the Shapley values using Plotly
    fig = go.Figure()

    # Iterate through each feature in the sorted order
    for i in feature_order:
      # Add a scatter plot trace for the current feature
      fig.add_trace(go.Scatter(y=[sample.columns[i]] * len(shap_values_percent[:, i]),
                             x=shap_values_percent[:, i],  # NEW: Use percentage values
                             mode='markers',
                             name=sample.columns[i],
                             marker=dict(color=shap_values_percent[:, i], colorscale='RdBu_r')))

    # Customize the appearance of the plot
    fig.update_layout(title="SHAP Summary Plot", xaxis_title="Shapley Value (%)", yaxis_title="Feature",
                      yaxis=dict(tickmode='array', tickvals=list(sample.columns[feature_order]),
                                 ticktext=list(sample.columns[feature_order]),
                                 tickfont=dict(size=12), tickangle=0),
                      height=figsize[1], width=figsize[0], margin=dict(l=100, r=100, t=100, b=100))

    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}_Shapley_Values_Scatter.png"
    pio.write_image(fig, filename)

    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Shapley_Values_Scatter.html", auto_open=True)


    # Display the plot
    fig.show()


def shapley_values_plotly_violin(
        X_test, 
        sample_size, 
        figsize=(1600, 1100), 
        model_path='xxx', 
        image_path='xxx', 
        label=None, 
        seed=42
):
    """
    Create a violin plot of the Shapley values using Plotly.
    
    :param X_test: DataFrame, testing feature set
    :param sample_size: float, proportion of X_test to use for computing SHAP values
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the Shapley values figure
    :param label: str, descriptive label for the model
    :param seed: int, random seed for reproducibility
    """

    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
        # Get the dictionary of column names and indices
        column_dict = {col_name: i for i, col_name in enumerate(X_test.columns)}
        # Compare the feature names in the model with the dictionary keys
        model_indices = [column_dict[feature_name] for feature_name in model.feature_names_in_ if feature_name in column_dict]
        X_test_transformed = X_test.iloc[:, model_indices]
    elif model_type == 2:
        model = loaded_model.estimator_
        # Apply RFE transformation to the testing data
        X_test_transformed = X_test.iloc[:, loaded_model.get_support()]
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
                # Apply RFE transformation to the testing data
                X_test_transformed = X_test.iloc[:, step_model.get_support()]
    elif model_type == 4:
      for step_name, step_model in loaded_model.named_steps.items():
        if isinstance(step_model, SelectKBest):
          model = loaded_model.named_steps['estimator']
          # Apply SelectKBest transformation to the testing data
          feature_index = step_model.get_support()
          X_test_transformed = X_test.iloc[:, feature_index]
          break

    # Calculate the sample size equal to 10% of X_test
    sample_size = sample_size

    # Randomly sample 10% of X_test_transformed using the specified seed for reproducibility
    sample = X_test_transformed.sample(n=sample_size, random_state=seed)

    # Create a Shapley explainer object using the appropriate explainer class based on the base estimator
    if isinstance(model, (GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression)):
        explainer = shap.LinearExplainer(model, X_test_transformed)  
    else:
        # Use KernelExplainer as a fallback option for other model types
        explainer = shap.KernelExplainer(model.predict_proba, X_test_transformed.sample(n=sample_size, random_state=seed))

    # Compute the Shapley values for the testing data using the shap_values() function
    shap_values = explainer.shap_values(sample)

    # Handle CatBoost separately
    if isinstance(model, (GradientBoostingClassifier, XGBClassifier, CatBoostClassifier)):
      # Here, shap_values is already a single array, not a list of two arrays
      total_shap_sums = np.abs(shap_values).sum(axis=1)
      shap_values_percent = (shap_values / total_shap_sums[:, None]) * 100
    else:
      # For other models, we use shap_values[1]
      total_shap_sums = np.abs(shap_values[1]).sum(axis=1)
      shap_values_percent = (shap_values[1] / total_shap_sums[:, None]) * 100

    # Sort the features by ascending mean Shapley value
    feature_order = np.argsort(np.abs(shap_values_percent).mean(axis=0))

    # Create a violin plot of the Shapley values using Plotly
    fig = go.Figure()

    # Iterate through each feature in the sorted order
    for i in feature_order:
        # Add a violin plot trace for the current feature
        fig.add_trace(go.Violin(y=[sample.columns[i]] * len(shap_values_percent[:, i]),
                                x=shap_values_percent[:, i],  # NEW: Use percentage values
                                box_visible=True,
                                line_color='blue',
                                meanline_visible=True,
                                fillcolor='lightseagreen',
                                opacity=0.6,
                                x0=i,
                                y0=sample.columns[i],
                                name=sample.columns[i],
                                orientation='h'))

    # Customize the appearance of the plot
    fig.update_layout(title="SHAP Summary Plot", xaxis_title="Shapley Value (%)", yaxis_title="Feature",
                      yaxis=dict(tickmode='array', tickvals=list(sample.columns[feature_order]),
                                 ticktext=list(sample.columns[feature_order]),
                                 tickfont=dict(size=12), tickangle=0),
                      height=figsize[1], width=figsize[0], margin=dict(l=100, r=100, t=100, b=100))

    # Save the figure to a file with a descriptive name
    filename = f"{image_path}{label.replace(' ', '_')}_Shapley_Values_Violin.png"
    pio.write_image(fig, filename)

    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Shapley_Values_Violin.html", auto_open=True)

    # Display the plot
    fig.show()

    
def plot_feature_importances_plotly(X_train, y_train, figure_size=(2000, 1000), model_path='xxx', image_path='xxx', label=None):
    """
    Create a bar plot of feature importances using Plotly, with the most important features at the top.
    
    :param X_train: DataFrame, training feature set
    :param y_train: Series, training target set (unused in the function)
    :param model_path: str, path to the saved model
    :param image_path: str, path to save the feature importances figure
    :param label: str, descriptive label for the model
    """  
    # Load the pickled RFE model
    loaded_model = jb.load(f"{model_path}{label.replace(' ', '_')}.pkl")

    # Identify the type of model and obtain the processed model and transformed test dataset
    model_type, loaded_model = load_and_identify_model(loaded_model)

    # Assign the appropriate model and transformed test dataset based on the model type
    if model_type == 1:
        model = loaded_model
    elif model_type == 2:
        model = loaded_model.estimator_
    elif model_type == 3:
        for step_name, step_model in loaded_model.named_steps.items():
            if isinstance(step_model, RFE):
                model = step_model.estimator_
    elif model_type == 4:
      for step_name, step_model in loaded_model.named_steps.items():
        if isinstance(step_model, SelectKBest):
          model = loaded_model.named_steps['estimator']
          break

    # Pull feature importances from the trained model
    importances = model.feature_importances_

    # Convert importances to percentages
    importances_percentage = importances * 100

    # Sort feature importances in descending order
    indices = np.argsort(importances_percentage)[::-1]

    # Create a horizontal bar chart using Plotly with the most important features listed at the top
    fig = go.Figure(go.Bar(
                x=importances_percentage[indices][::-1],
                y=[X_train.columns[i] for i in indices][::-1],
                orientation='h',
                marker=dict(color=importances_percentage[indices][::-1],
                            colorbar=dict(title='Importance'))
                ))
    fig.update_layout(title=f"Feature Importances - {label}",
                      xaxis_title='Importance (%)',
                      yaxis_title='',
                      width=figure_size[0],
                      height=figure_size[1])

    # Save the figure as png
    filename = f"{image_path}{label.replace(' ', '_')}_Feature_Importances.png"
    pio.write_image(fig, filename)

    # Save the figure as html with interactive figure
    pio.write_html(fig, file=f"{image_path}{label.replace(' ', '_')}_Feature_Importances.html", auto_open=True)

    # Show the plot
    fig.show()

