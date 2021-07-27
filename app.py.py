import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix,  plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
from sklearn.metrics import (precision_recall_curve, plot_precision_recall_curve,
                             average_precision_score)
from sklearn.metrics import f1_score, precision_score, recall_score


import pickle

import streamlit as st
import feature_engine

import shap


st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("Customer Churn Prediction")

st.sidebar.title("Customer Churn Prediction")
st.sidebar.markdown("Explanation with SHAP")

select_display = st.sidebar.radio("Select Analysis",
    options=["Data","Model Results", "Feature Importance","Individual Results", 
             "What IF Analysis"])


# load data
@st.cache(persist=True)
def load_data():
    data = pd.read_csv("Churn.csv", na_values=" ")
    return data

data = load_data()

# drop na
data  = data.dropna()


# select only the features used in modeling from the data
selected_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'tenure', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'TechSupport', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'MonthlyCharges', 'TotalCharges']


# DISPLAY DATA
if select_display == "Data":
    n_rows = st.slider("Select No. of Rows to Display", min_value=10,
                       max_value=len(data), value = 10, step = 10)
    st.text("Data")
    st.dataframe(data[selected_features+['Churn']].head(n_rows))
    

    
# MODEL RESULTS

# load the encoder
@st.cache(persist=True)
def load_label_encoder():
   with open("encoder.pkl", 'rb') as file:
      encoder = pickle.load(file)
        
      return encoder

# load the model
@st.cache(persist=True,allow_output_mutation=True)
def load_model():
   with open('model.pkl','rb') as file:
      RF_model = pickle.load(file)
      return RF_model

label_encoder = load_label_encoder()
model = load_model()

# seprate features and target
features = data[selected_features]
target = data['Churn']
    
# convert target to binary
target = target.map({'No':0, 'Yes':1})
    
# encode the features
features = label_encoder.transform(features)

if select_display =="Model Results":
    
   # select the evaluation metrics
    metric = st.selectbox("Select the Metric", 
                          options=["Confusion Matrix","ROC-AUC", "Precision-Recall"])
    
    if metric == "Confusion Matrix":
        threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0,
                              value=0.5,step=0.1)
        
        # make predictions for churn
        predictions = model.predict_proba(features)[:,1]
        pred_labels = np.where(predictions > threshold, 1, 0)
        
        col1, col2 = st.beta_columns((2,1))
        
        # confusion matrix
        with col1:
            fig, ax = plt.subplots(figsize = (3,3))
            conf_mat = confusion_matrix(target, pred_labels)
            sns.heatmap(conf_mat, cmap='Blues', cbar=False, annot=True, fmt='.4g',
                        xticklabels=['No', 'Yes'],yticklabels=['No', 'Yes'], ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("True Labels")
            
            st.pyplot(fig)
            
        
        with col2:
            
            st.markdown(f'F1 Score: {round(f1_score(target, pred_labels),3)}')
            st.markdown(f'Precision: {round(precision_score(target, pred_labels),3)}')
            st.markdown(f'Recall: {round(recall_score(target, pred_labels),3)}')
        
    if metric == "ROC-AUC":
        fig, ax = plt.subplots(figsize = (2,2))
        plot_roc_curve(estimator=model, X=features, y=target)
        plt.plot([0,1], [0,1], linestyle='--', color='orange')
        st.pyplot()
        
    if metric == "Precision-Recall":
        fig, ax = plt.subplots(figsize = (2,2))
        plot_precision_recall_curve(estimator=model, X=features, y=target)
        plt.axhline(0.266,linestyle='--', color='orange' )
        plt.legend(loc='upper right')
        st.pyplot()   



#FEATURE IMPORTANCE

#load shap values
@st.cache(persist=True,allow_output_mutation=True)
def load_SHAP_VALUES():
    
    with open('shap_values.pkl','rb') as file:
        shap_values = pickle.load(file)
        return shap_values


if select_display =="Feature Importance":
    
    
    shap_values = load_SHAP_VALUES()
    st.markdown('**Feature Importance with SHAP**')
    shap.summary_plot(shap_values.values[:,:,1], plot_type='bar', 
                      feature_names=selected_features)
    st.pyplot()
    


# INDIVIDUAL CUSTOMER RESULT

if select_display =="Individual Results":
    
    customer_id = st.selectbox(label = "Select a Customer ID", 
                               options =data.customerID.values)
    data_index = data[data.customerID == customer_id].index.values.item()
    
    # churn probability
    churn_prob = model.predict_proba(features.iloc[data_index].values.reshape(1,-1))[:,1].item()
    
    if st.button("Submit"):
        
        # show the customer info
    
        st.write("Customer Info")
        customer_info = data.iloc[data_index][selected_features]
        customer_info.name = 'Values'
        st.write(pd.DataFrame(customer_info).T.reset_index(drop = True))
        st.write("")
        
        col1,col2 = st.beta_columns((3,1))
        
        with col1:
            
            st.write("WaterFall Chart")
            shap_values = load_SHAP_VALUES()
            shap.plots.waterfall(shap_values=shap_values[data_index][:,1], max_display=15)
            st.pyplot()
        
        with col2:
           st.markdown(f"Churned(Actual): {data.iloc[data_index]['Churn']}")
                       
           st.markdown(f'Predicted Churn Probability: {round(churn_prob,3)}') 


#WHAT IF ANALYSIS

if select_display=="What IF Analysis":
    
    #st.markdown("**What-IF-Analysis**")

    st.subheader("Select the Feature Values")
    
    input_data = pd.DataFrame(index = [0], columns=selected_features)
    for feat in selected_features:
        if (data[feat].dtype == 'O') or (len(data[feat].unique()) < 10):
            input_data[feat] = st.selectbox(label=feat, options=data[feat].unique())
        else:
            input_data[feat] = st.number_input(label=feat, min_value=0)

    submit = st.button(label="Submit")
    if submit:
        st.markdown("**Selected Customer Info**")
        st.write(input_data)
        
        # encoding the labels
        encoded_data = label_encoder.transform(input_data)
        
        # predictions
        pred = model.predict_proba(encoded_data)[:,1].item()
        st.write(f"Predicted Probability of Churn: {round(pred, 3)}")
        
        col1, col2 = st.beta_columns((1,1.2))
        explainer = shap.TreeExplainer(model = model)
        input_data_shap_values = explainer(encoded_data)
        
       # variable importance
        with col1:
            st.markdown("**variable Importance**")
            shap.summary_plot(input_data_shap_values.values[:,:,1], plot_type='bar', 
                              feature_names=selected_features)
            st.pyplot()
        
            
        # waterfall chart
        with col2:
            st.markdown("**Explaning the Prediction with waterfall chart**")
            shap.plots.waterfall(shap_values = input_data_shap_values[0][:,1], max_display=15)
            st.pyplot()
            