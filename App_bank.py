# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
                 
logo =  Image.open("./Logo.png")

st.sidebar.image(logo, width=240, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')

number = st.slider("Pick a number",0,100)

st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Select Menu:',
    ['Overview' ,'Data Analysis', 'Model & Prediction','Prédire solvabilité client','Intéprétabilité'],
)

                          
  
