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

sdg = Image.open('../streamlit/logo_dark.png')
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
    placeholder = st.empty()
    
    with placeholder.beta_container():
        with st.spinner('Load Models and Predict...'):
            
            if categories != "":
                for category in categories:
                    
                    # take only models with over 20 training datapoints
                    if df[category].sum(axis=0) > 20:
                        
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
                            all_results_score.append(y_prob[0][1].round(2)*100)

                            #element = st.write(category)
                            number.append(df[category].sum(axis=0))
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
                            all_results_score.append(y_prob[0][1].round(2)*100)

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

    #time.sleep(3)
    #placeholder.empty()
    if name != []:    
        st.write("Prediction Results:")
        #st.table(df)


        #grid table
        #Example controlers
        st.sidebar.subheader("St-AgGrid example options")

        sample_size = st.sidebar.number_input("rows", min_value=10, value=10)
        grid_height = st.sidebar.number_input("Grid height", min_value=200, max_value=800, value=200)

        return_mode = st.sidebar.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
        return_mode_value = DataReturnMode.__members__[return_mode]

        update_mode = st.sidebar.selectbox("Update Mode", list(GridUpdateMode.__members__), index=6)
        update_mode_value = GridUpdateMode.__members__[update_mode]

        #features
        fit_columns_on_grid_load = st.sidebar.checkbox("Fit Grid Columns on Load")

        enable_selection=st.sidebar.checkbox("Enable row selection", value=True)
        if enable_selection:
            st.sidebar.subheader("Selection options")
            selection_mode = st.sidebar.radio("Selection Mode", ['single','multiple'])
            
            use_checkbox = st.sidebar.checkbox("Use check box for selection")
            if use_checkbox:
                groupSelectsChildren = st.sidebar.checkbox("Group checkbox select children", value=True)
                groupSelectsFiltered = st.sidebar.checkbox("Group checkbox includes filtered", value=True)

            if ((selection_mode == 'multiple') & (not use_checkbox)):
                rowMultiSelectWithClick = st.sidebar.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
                if not rowMultiSelectWithClick:
                    suppressRowDeselection = st.sidebar.checkbox("Suppress deselection (while holding CTRL)", value=False)
                else:
                    suppressRowDeselection=False
            st.sidebar.text("___")


        enable_pagination = st.sidebar.checkbox("Enable pagination", value=False)

        if enable_pagination:
            st.sidebar.subheader("Pagination options")
            paginationAutoSize = st.sidebar.checkbox("Auto pagination size", value=True)
            if not paginationAutoSize:
                paginationPageSize = st.sidebar.number_input("Page size", value=5, min_value=0, max_value=sample_size)
            st.sidebar.text("___")

        #Infer basic colDefs from dataframe types
        gb = GridOptionsBuilder.from_dataframe(df)

        #customize gridOptions
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)

        #configures last row to use custom styles based on cell's value, injecting JsCode on components front end
        cellsytle_jscode = JsCode("""
        function(params) {
            if (params.value == 'A') {
                return {
                    'color': 'white',
                    'backgroundColor': 'darkred'
                }
            } else {
                return {
                    'color': 'black',
                    'backgroundColor': 'white'
                }
            }
        };
        """)

        enable_sidebar = False

        if enable_sidebar:
            gb.configure_side_bar()

        if enable_selection:
            gb.configure_selection(selection_mode)
            if use_checkbox:
                gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
            if ((selection_mode == 'multiple') & (not use_checkbox)):
                gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)

        if enable_pagination:
            if paginationAutoSize:
                gb.configure_pagination(paginationAutoPageSize=True)
            else:
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

        gb.configure_grid_options(domLayout='normal')
        gridOptions = gb.build()

        #Display the grid
        st.header("Ag-Grid")

        grid_response = AgGrid(
            df, 
            gridOptions=gridOptions,
            height=grid_height, 
            width='100%',
            data_return_mode=return_mode_value, 
            update_mode=update_mode_value,
            fit_columns_on_grid_load=fit_columns_on_grid_load,
            allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
            )

        df = grid_response['data']
        selected = grid_response['selected_rows']
        selected_df = pd.DataFrame(selected)

        st.table(selected_df)

        a = 0
        st.write("Suggested Categories:")
        for key, value in results.items():
            new_df = clean_df
            st.write(key, 'with', value, '% confidence.')
            # if st.button("Correct?"):
            #     get_data().append({"text": text_input, key: 1})
           
            a = a+1
    else:
        t = "<div> <span class='highlight red'>Not enough confidence in any category.</div>"
        st.markdown(t, unsafe_allow_html=True)        



new_annotations = pd.DataFrame(get_data())

#st.write(pd.DataFrame(get_data()))

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



# https://share.streamlit.io/pablocfonseca/streamlit-aggrid/main/examples/example.py