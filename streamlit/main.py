#basics
import pandas as pd
import numpy as np
import joblib
import pickle5 as pickle
from PIL import Image
import time

#eli5
from eli5 import show_prediction

#streamlit
import streamlit as st
import SessionState
from load_css import local_css
local_css("style.css")

DEFAULT = '< PICK A VALUE >'
def selectbox_with_default(text, values, default=DEFAULT, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

#helper functions
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import src.clean_dataset as clean

sys.path.pop(0)

#define empty list to save data
@st.cache(allow_output_mutation=True)
def get_data():
    return []

#%%
#1. load in complete transformed and processed dataset for pre-selection and exploration purpose
df = pd.read_csv('../data/taxonomy_final.csv')
df_columns = df.drop(columns=['PIMS_ID', 'all_text_clean', 'all_text_clean_spacy',  'hyperlink',
 'title',
 'leading_country',
 'grant_amount',
 'country_code',
 'lon',
 'lat'])
    
to_match = df_columns.columns.tolist()

#2. load parent dict
with open("../data/parent_dict.pkl", 'rb') as handle:
    parents = pickle.load(handle)

#3. load sub category dict
with open("../data/category_dict.pkl", 'rb') as handle:
    sub = pickle.load(handle)    
    
#4. Load Training Scores:
with open("../data/tfidf_only_f1.pkl", 'rb') as handle:
    scores_dict = pickle.load(handle)     

#5. Load all categories as list:
with open("../data/all_categories_list.pkl", 'rb') as handle:
    all_categories = pickle.load(handle)


#sort list
all_categories = sorted(all_categories)    

#%%
session = SessionState.get(run_id=0)
categories = ""

#%%
#title start page
#%%
#title start page
st.title('Machine Learning for Nature Climate Energy Portfolio')

sdg = Image.open('../streamlit/logo.png')
st.sidebar.image(sdg, width=200)
st.sidebar.title('Navigation')

st.write('## Frontend Application that takes text as input and outputs classification decision.')

st.write("Map your project, document or text to NCE taxonomy and help improve the models by giving your feedback!")

    
items = [k  for  k in  parents.keys()]
items.insert(0,'')

option = st.sidebar.selectbox('Select a category:', items, format_func=lambda x: 'Select a category' if x == '' else x)
if option:
    st.sidebar.write('You selected:', option)
    st.sidebar.markdown("**Categories**")

    for i in parents[option]:
        st.sidebar.write(i)
    st.sidebar.markdown("**Further Choices**")
    
    agree = st.sidebar.checkbox(label='Would you like to display and predict sub-categories of your choice?', value = False)            
    if agree:
        sub_option = st.sidebar.selectbox('Select a category:', parents[option], format_func=lambda x: 'Select a category' if x == '' else x)
        if sub_option:
            st.sidebar.markdown("**Sub Categories:**")
            for i in sub[sub_option]:
                st.sidebar.write(i)
        categories = sub[sub_option]                   
    else:
        categories = parents[option]


#choose one category from all:
agree = st.sidebar.checkbox(label='Would you like to predict specific categories?', value = False)            
if agree:
    all_options = st.sidebar.multiselect('Select a category:', all_categories, format_func=lambda x: 'Select a category' if x == '' else x)
    if all_options:
        st.sidebar.markdown("**You've chosen:**")
        for i in all_options:
            st.sidebar.write(i)                
        categories = all_options

# predict each category:             
agree = st.sidebar.checkbox(label='Would you like to predict the whole taxonomy?', value = False, key= "key1")            
if agree:
    categories = all_categories                              

    

text_input = st.text_input('Please Input your Text:')

#define lists
name = []
hat = []
number = []        
top_5 = []
last_5 = []
top_5_10 = []
last_5_10 = []

if text_input != '':
    placeholder = st.empty()
    
    with placeholder.beta_container():
        with st.spinner('Load Models and Predict...'):
            
            if categories != "":
                for category in categories:
                    
                    # take only models with over 20 training datapoints
                    if df[category].sum(axis=0) > 20:
                        
                        #prune the long names to ensure proper loading
                        if len(category) > 50:
                            category = category[0:20] 
                            st.write(category)
                        else:
                            pass 
                        
                        # Pre-process text:
                        input_list = [text_input]
                        input_df = pd.DataFrame(input_list, columns =['input_text'])
                        
                        # clean text
                        input_df['input_text'] = input_df['input_text'].apply(clean.spacy_clean)
                        clean_df = pd.Series(input_df['input_text'])   
                        
                        tfidf_vectorizer = joblib.load('../models/tf_idf/tf_idf_only/'+category+'_'+'vectorizer.sav')        
                        fnames = tfidf_vectorizer.get_feature_names()
                        
                        vector_df = tfidf_vectorizer.transform(clean_df)

                        clf = joblib.load('../models/tf_idf/tf_idf_only/'+category+'_'+'model.sav')
                        y_hat = clf.predict(vector_df)
                        y_prob = clf.predict_proba(vector_df)
                        
                        if y_hat == 1:        
                            element = st.write(category)
                            number.append(df[category].sum(axis=0))
                            name.append(category)
                            element = st.write("Yes with Confidence:", y_prob[0][1].round(2)*100, "%")                  
                            hat.append(y_prob[0][1].round(2)*100)
                            
                            results= dict(zip(name, hat))
                            
                            #return top features:
                            w = show_prediction(clf, tfidf_vectorizer.transform(clean_df), 
                                            highlight_spaces = True, 
                                            top=5000, 
                                            feature_names=fnames, 
                                            show_feature_values  = True)                                    
                            result = pd.read_html(w.data)[0]
                            top_5_list = result.Feature.iloc[0:5].tolist()
                            top_5.append(top_5_list)
                            
                            top_5_10_list = result.Feature.iloc[5:10].tolist()
                            top_5_10.append(top_5_10_list)
                            
                            last_5_list = result.Feature.iloc[-5:].tolist()
                            last_5.append(last_5_list)
                            
                            last_5_10_list = result.Feature.iloc[-10:].tolist()
                            last_5_10_list = list(set(last_5_10_list) - set(last_5_list))
                            last_5_10.append(last_5_10_list)

                        if y_hat == 0:
                            element= st.write(category)
                            element = st.write("No with Confidence:", y_prob[0][0].round(2)*100, "%")
            else:
                st.warning('No category is selected')

    time.sleep(3)
    placeholder.empty()
    if name != []:    
        st.header("Suggested Categories:")
        st.write('           ')
        
        
        a = 0
        for key, value in results.items():
            new_df = clean_df
            st.write(key, 'with', value, '% confidence.')
            if st.button("Correct?"):
                get_data().append({"text": text_input, key: 1})
            # st.write('Model was trained on', number[0], 'examples with accuracy (F1 Score) of:', scores_dict[key].round(2)*100, "%")
            
                
            # st.write('Detailed Explanation of Prediction:')
            # for item in top_5[a]:
            #     green = "<span class='highlight green'>"+item+"</span>"
            #     item = item
            #     new_df = new_df.str.replace(item,green)
                
            # for item in last_5[a]:    
            #     red = "<span class='highlight red'>"+item+"</span>"
            #     item = " "+item+" "
            #     new_df = new_df.str.replace(item,red)
                
            # for item in top_5_10[a]:
            #     lightgreen = "<span class='highlight lightgreen'>"+item+"</span>"
            #     item = " "+item+" "
            #     new_df = new_df.str.replace(item,lightgreen)
                
            # for item in last_5_10[a]:
            #     lightred = "<span class='highlight IndianRed'>"+item+"</span>"
            #     item = " "+item+" "
            #     new_df = new_df.str.replace(item,lightred)

            # text = new_df[0]
            # text = "<div>"+text+"</div>"
            # st.markdown(text, unsafe_allow_html=True)
            # st.write('           ')
            # st.write('           ')
            # st.write('           ')
            
            a = a+1
    else:
        t = "<div> <span class='highlight red'>Not enough confidence in any category.</div>"
        st.markdown(t, unsafe_allow_html=True)        



new_annotations = pd.DataFrame(get_data())

st.write(pd.DataFrame(get_data()))

#%%
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
if st.button("Run again!"):
  session.run_id += 1

#%%
from pathlib import Path
p = Path('.')