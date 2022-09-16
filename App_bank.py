# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
from PIL import Image               
logo =  Image.open("./Logo.png")

st.sidebar.image(logo, width=240, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')

number = st.slider("Pick a number",0,100)

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

                          
  
