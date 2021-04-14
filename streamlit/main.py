#basics
import pandas as pd
import numpy as np
import joblib
import pickle5 as pickle
from PIL import Image
import time
import datetime

#firestore 
from google.cloud import firestore
db = firestore.Client.from_service_account_json("firestore_key.json")

#eli5
from eli5 import show_prediction

#streamlit
import streamlit as st
st.set_page_config(layout="wide")
import SessionState
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
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
#%%

@st.cache(allow_output_mutation=True)
def load_data():
    #1. load in complete transformed and processed dataset for pre-selection and exploration purpose
    df_taxonomy = pd.read_csv('../data/taxonomy_final.csv')
    df_columns = df_taxonomy .drop(columns=['PIMS_ID', 'all_text_clean', 'all_text_clean_spacy',  'hyperlink',
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

    # sort list
    all_categories = sorted(all_categories)    

    return df_taxonomy, df_columns, to_match, parents, sub, scores_dict, all_categories

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

df_taxonomy, df_columns, to_match, parents, sub, scores_dict, all_categories = load_data()
items = [k  for  k in  parents.keys()]
items.insert(0,'')

# load data and choose category
with st.spinner('Choose categories to predict...'):

    agree = st.sidebar.checkbox(label='Would you like to browse and choose the different classes?', value = False, key= "key0")       
    if agree:
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
    agree = st.sidebar.checkbox(label='Would you like to predict specific categories?', value = False, key = "key1")     
    if agree:
        all_options = st.sidebar.multiselect('Select a category:', all_categories, format_func=lambda x: 'Select a category' if x == '' else x)
        if all_options:
            st.sidebar.markdown("**You've chosen:**")
            for i in all_options:
                st.sidebar.write(i)                
            categories = all_options

    # predict each category:             
    agree = st.sidebar.checkbox(label='Would you like to predict the whole taxonomy?', value = False, key= "key2")            
    if agree:
        categories = all_categories             

           
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def predict(text_input):

    #define lists
    all_results_name = []
    all_results_score = []
    name = []
    hat = []
    number = []        
    top_5 = []
    last_5 = []
    top_5_10 = []
    last_5_10 = []

    if text_input != '':
        #placeholder = st.empty()
        
        #with placeholder.beta_container():
        with st.spinner('Load Models and Predict...'):
            
            if categories != "":
                for category in categories:
                    
                    # take only models with over 20 training datapoints
                    if df_taxonomy[category].sum(axis=0) > 20:
                        
                        #prune the long names to ensure proper loading
                        if len(category) > 50:
                            #st.write("pruning the long name of category:", category)
                            category = category[0:20] 
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

                            all_results_name.append(category)
                            all_results_score.append(y_prob[0][1].round(1)*100)

                            #element = st.write(category)
                            number.append(df_taxonomy[category].sum(axis=0))
                            name.append(category)
                            #element = st.write("Yes with Confidence:", y_prob[0][1].round(2)*100, "%")                  
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

                            all_results_name.append(category)
                            all_results_score.append(y_prob[0][1].round(1)*100)

                            #element= st.write(category)
                            #element = st.write("No with Confidence:", y_prob[0][0].round(2)*100, "%")

                # make dataframe from all prediction results
                df = pd.DataFrame(
                    {'category': all_results_name,
                    'confidence_score': all_results_score 
                    })

                # add decision column:
                df['prediction'] = np.where(df['confidence_score']>= 50, "True", "False")
                df['confidence_score'] = df['confidence_score'].astype(str) + "%"
                df = df[['category', 'prediction', 'confidence_score']]

            else:
                st.warning('No category is selected')

    return all_results_name, all_results_score, name, hat, number, top_5, last_5, top_5_10, last_5_10, clean_df, results, df

text_input = st.text_area('Please Input your Text:')

if text_input != "":
    all_results_name, all_results_score, name, hat, number, top_5, last_5, top_5_10, last_5_10, clean_df, results, df = predict(text_input)

    #time.sleep(3)
    #placeholder.empty()
    if all_results_name != []:    
        # st.write("Prediction Results:")
        # st.table(df)

        #grid table
        #Example controlers
        st.sidebar.subheader("AgGrid layout options")

        #sample_size = st.sidebar.number_input("rows", min_value=0, value=len(df))
        grid_height = st.sidebar.slider("Grid height", min_value=100, max_value=1000, value=400)

        return_mode_value = DataReturnMode.FILTERED
        update_mode_value = GridUpdateMode.MODEL_CHANGED

        selection_mode = 'multiple'
        
        # use_checkbox = True
        # if use_checkbox:
        groupSelectsChildren = True
        groupSelectsFiltered = True

        # if selection_mode == 'multiple':
        rowMultiSelectWithClick = False
        if not rowMultiSelectWithClick:
            suppressRowDeselection = False

        #Infer basic colDefs from dataframe types
        gb = GridOptionsBuilder.from_dataframe(df)

        #row height
        gb.configure_grid_options(rowHeight=45)
        #customize gridOptions
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)

        #configures last row to use custom styles based on cell's value, injecting JsCode on components front end
        cellsytle_jscode = JsCode("""
        function(params) {
            if (params.value == 'True') {
                return {
                    'color': 'white',
                    'backgroundColor': 'green'
                }
            }   
            else {
                return {
                    'color': 'black',
                    'backgroundColor': 'white'
                }
            }
        };
        """)

        gb.configure_column("prediction", cellStyle=cellsytle_jscode)
        gb.configure_column("category", cellStyle={'color': 'blue'})

        gb.configure_side_bar()

        gb.configure_selection(selection_mode)
        gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
        gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)

        gb.configure_grid_options(domLayout='normal')
        gridOptions = gb.build()

        #Display the grid
        st.write("## Prediction Result")
        st.write("Please tick all categories you think are NOT correct and then submit your choices. You may take guidance from the model's confidence scores. Your feedback will be stored together with the predicted text and will help the models to make better decisions in the future.")
        grid_response = AgGrid(
            df, 
            gridOptions=gridOptions,
            height=grid_height, 
            width='100%',
            data_return_mode=return_mode_value, 
            update_mode=update_mode_value,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
            )

        if st.button('Submit Corrections.'):
            df = grid_response['data']
            selected = grid_response['selected_rows']
            selected_df = pd.DataFrame(selected)
            selected_df['corrected_prediction'] = np.where(selected_df['prediction']== "False", "True", "False")

            selected_df = selected_df[['category', 'corrected_prediction']]
            selected_df = selected_df.T
            selected_df.columns = selected_df.iloc[0]
            selected_df.drop(selected_df.index[0], inplace=True)
            selected_df.reset_index(drop=True, inplace=True)
            st.table(selected_df)

            selected_df['text'] = text_input
            #shift columns
            cols = list(selected_df.columns)
            cols = [cols[-1]] + cols[:-1]
            selected_df = selected_df[cols]

            

            #store in firestore:
            selected_df = selected_df.astype(str)
            selected_df.index = selected_df.index.map(str)
            postdata = selected_df.to_dict()

            date = str(datetime.datetime.now())
            db.collection(u'feedback').document(date).set(postdata)

            st.success("Feedback successfully stored in the cloud!")

    else:
        t = "<div> <span class='highlight red'>Not enough confidence in any category.</div>"
        st.markdown(t, unsafe_allow_html=True)        
#%%
# st.write('           ')
# st.write('           ')
# st.write('           ')
# st.write('           ')
# st.write('           ')
# st.write('           ')
# if st.button("Run again!"):
#   session.run_id += 1

#%%
from pathlib import Path
p = Path('.')



# https://share.streamlit.io/pablocfonseca/streamlit-aggrid/main/examples/example.py