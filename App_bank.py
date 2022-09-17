# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
import pickle
import time
from PIL import Image        

filename = './modelisation/classifier_lgbm_model.sav'
with open(filename, 'rb') as lgbm_model:
    best_model = pickle.load(lgbm_model)
logo =  Image.open("./Logo.png")

st.sidebar.image(logo, width=240, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')

html_header="""
    <head>
        <title>Application Dashboard Crédit Score</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Loetitia Rabier">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:300%; color:Crimson; font-family:Arial"> Prêt à dépenser <br>
        <h2 style="color:Gray; font-family:Georgia"> DASHBOARD</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Select Menu:',
    ['Overview' ,'Data Analysis', 'Model & Prediction','Prédire solvabilité client','Intéprétabilité'],
)


if sidebar_selection == 'Overview':
    selected_item =""
    with st.spinner('Data load in progress...'):
        time.sleep(2)
    st.success('Data loaded')
    show_data () 
    show_overview () 

if sidebar_selection == 'Data Analysis':
    selected_item = st.sidebar.selectbox('Select Menu:', 
                                ('Graphs', 'Distributions'))

if sidebar_selection == 'Model & Prediction':
    selected_item = st.sidebar.selectbox('Select Menu:', 
                                    ( 'Prediction','Model'))

if sidebar_selection == 'Prédire solvabilité client':
    selected_item="predire_client"

if sidebar_selection == 'Intéprétabilité':
    selected_item =""
    affiche_facteurs_influence()
    

seuil_risque = st.sidebar.slider("Seuil de Solvabilité", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if selected_item == 'Data':
    show_data ()  

if selected_item == 'Solvency':
    show_overview ()  

if selected_item == 'Graphs':
    #hist_graph()
    is_educ_selected,is_statut_selected,is_income_selected = filter_graphs()
    if(is_educ_selected=="oui"):
        education_type()
    if(is_statut_selected=="oui"):
        statut_plot()
    if(is_income_selected=="oui"):  
        income_type()

if selected_item == 'Distributions':
    is_age_selected,is_incomdis_selected = filter_distribution()
    if(is_age_selected=="oui"):
        age_distribution()
    if(is_incomdis_selected=="oui"):
        revenu_distribution()

    
    

if selected_item == 'Prediction':
    show_client_predection()

if selected_item == 'Model':
    show_model_analysis()

if selected_item == 'predire_client':
    show_client_prediction()

if selected_item == 'Fluance':
    affiche_facteurs_influence()
 








html_line="""
<br>
<br>
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;">
<p style="color:Gray; text-align: right; font-size:12px;">Auteur : elfenaouyreda@gmail.com - 24/08/2022</p>
"""
st.markdown(html_line, unsafe_allow_html=True)

                          
  
