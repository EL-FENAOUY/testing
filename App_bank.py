# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st

number = st.slider("Pick a number",0,100)




# ====================================================================
# MENUS
# ====================================================================


st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Select Menu:',
    ['Overview' ,'Data Analysis', 'Model & Prediction','Prédire solvabilité client','Intéprétabilité'],
)
                          
  
