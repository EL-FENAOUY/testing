# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import time
import json
import lightgbm
import plotly.graph_objects as go
from urllib.request import urlopen
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.figure import Figure
import streamlit.components.v1 as components
from PIL import Image  
import pandas as pd
from data_api import *
import shap

filename = './modelisation/classifier_lgbm_model.sav'
with open(filename, 'rb') as lgbm_model:
    best_model = pickle.load(lgbm_model)

# Chargement de agg_pay_num
filename = './shap/shap_values.pickle'
with open(filename, 'rb') as shap_file:
    shap_values = pickle.load(shap_file)


sample_size = 20000
data ,train_set,y_pred_test_export = load_all_data(sample_size)
test_set = pd.read_csv('data/test_set_echantillon.csv')



def show_data ():
    st.write(data.head(10)) 
    

def pie_chart(thres):
    #st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
    percent_sup_seuil =100* (data['TARGET']>thres).sum()/data.shape[0]
    percent_inf_seuil = 100-percent_sup_seuil
    d = {'col1': [percent_sup_seuil,percent_inf_seuil], 'col2': ['% Non Solvable','% Solvable',]}
    df = pd.DataFrame(data=d)
    fig = px.pie(df,values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
    st.plotly_chart(fig)
def show_overview():
    st.title("Risque")
    risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                    max_value = 1.0 ,
                     value = 0.5,
                     step = 0.1)
    
    pie_chart(risque_threshold) 
 
def filter_graphs():
    st.subheader("Filtre des Graphes")
    col1, col2,col3 = st.columns(3)
    is_educ_selected = col1.radio("Graph Education",('non','oui'))
    is_statut_selected = col2.radio('Graph Statut',('non','oui'))
    is_income_selected = col3.radio('Graph Revenu',('non','oui'))

    return is_educ_selected,is_statut_selected,is_income_selected

def hist_graph ():
    st.bar_chart(data['DAYS_BIRTH'])
    df = pd.DataFrame(data[:200],columns = ['DAYS_BIRTH','AMT_CREDIT'])
    df.hist()
    st.pyplot()

def education_type():
    ed = train_set.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = train_set.NAME_EDUCATION_TYPE.unique() 
    

    fig = go.Figure(data=[go.Bar(
            x=u_ed,
            y=ed
        )])
    fig.update_layout(title_text='Data education')

    st.plotly_chart(fig)

    ed_solvable = train_set[train_set['TARGET']==0].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    ed_non_solvable = train_set[train_set['TARGET']==1].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = train_set.NAME_EDUCATION_TYPE.unique() 
    

    fig = go.Figure(data=[
        go.Bar(name='Solvable',x=u_ed,y=ed_solvable),
        go.Bar(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
        ])
    fig.update_layout(title_text='Solvabilité Vs education')

    st.plotly_chart(fig)

def statut_plot ():
    ed = train_set.groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = train_set.NAME_FAMILY_STATUS.unique() 
    

    fig = go.Figure(data=[go.Bar(
            x=u_ed,
            y=ed
        )])
    fig.update_layout(title_text='Data situation familiale')

    st.plotly_chart(fig)

    ed_solvable = train_set[train_set['TARGET']==0].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    ed_non_solvable = train_set[train_set['TARGET']==1].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = train_set.NAME_FAMILY_STATUS.unique() 
    

    fig = go.Figure(data=[
        go.Bar(name='Solvable',x=u_ed,y=ed_solvable),
        go.Bar(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
        ])
    fig.update_layout(title_text='Solvabilité Vs situation familiale')

    st.plotly_chart(fig)

def income_type ():
    ed = train_set.groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = train_set.NAME_INCOME_TYPE.unique() 
    

    fig = go.Figure(data=[go.Bar(
            x=u_ed,
            y=ed
        )])
    fig.update_layout(title_text='Data Type de Revenu')

    st.plotly_chart(fig)

    ed_solvable = train_set[train_set['TARGET']==0].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    ed_non_solvable = train_set[train_set['TARGET']==1].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = train_set.NAME_INCOME_TYPE.unique() 
    

    fig = go.Figure(data=[
        go.Bar(name='Solvable',x=u_ed,y=ed_solvable),
        go.Bar(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
        ])
    fig.update_layout(title_text='Solvabilité Vs Type de Revenu')

    st.plotly_chart(fig)


def filter_distribution():
    st.subheader("Filtre des Distribution")
    col1, col2 = st.beta_columns(2)
    is_age_selected = col1.radio("Distribution Age ",('non','oui'))
    is_incomdis_selected = col2.radio('Distribution Revenus ',('non','oui'))

    return is_age_selected,is_incomdis_selected 

def age_distribution():
    df = pd.DataFrame({'Age':data['DAYS_BIRTH'],
                'Solvabilite':data['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}        
    df=df.replace({"Solvabilite": dic})    
      
    
    fig = px.histogram(df,x="Age", color="Solvabilite", nbins=40)
    st.subheader("Distribution des ages selon la sovabilité")
    st.plotly_chart(fig)  
    
def revenu_distribution():
    df = pd.DataFrame({'Revenus':data['AMT_INCOME_TOTAL'],
                'Solvabilite':data['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}        
    df=df.replace({"Solvabilite": dic})    
      
    
    fig = px.histogram(df,x="Revenus", color="Solvabilite", nbins=40)
    st.subheader("Distribution des revenus selon la sovabilité")
    st.plotly_chart(fig)
    
    
#--------------------------- Client Predection --------------------------

def show_client_predection():
    client_id = st.number_input("Donnez Id du Client",100002)
    if st.button('Voir Client'):
        client=data[data['SK_ID_CURR']==client_id]
        
        display_client_info(str(client['SK_ID_CURR'].values[0]),str(client['AMT_INCOME_TOTAL'].values[0]),str(round(client['DAYS_BIRTH'].values[0])),str(round(client['DAYS_EMPLOYED']/-365).values[0]))
        
        
        #st.header('ID :'+str(client['SK_ID_CURR'][0]))
        #st.write(data['age_bins'].value_counts())
        API_url = "https://banking2023.herokuapp.com/credit/" + str(client_id)
        with st.spinner('Chargement du score du client...'):
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())
            y_pred = API_data['prediction']
            y_proba = API_data['proba']
            
        
        
        
        
        st.info('Prediction du client : '+str(int(100*y_proba))+' %')
        client_prediction= st.progress(0)
        for percent_complete in range(int(100*y_proba)):
            time.sleep(0.01)

        client_prediction.progress(percent_complete + 1)
        if(y_proba < seuil_risque):
            st.success('Client solvable')
        if(y_proba >=seuil_risque):
            st.error('Client non solvable')

        st.subheader("Tous les détails du client :")
        st.write(client)
        

    
        
        #Bar Chart
        age_bins = data['age_bins'].value_counts(sort=False)

        d = {'Ages par Decennie': age_bins.index, 'Nombre de clients par Decennie':age_bins.values}
        ages_decinnie = pd.DataFrame(data=d)

        ages_decinnie['Ages par Decennie'] = ages_decinnie['Ages par Decennie'].astype(str)
        idx_decinnie = ages_decinnie[ages_decinnie['Ages par Decennie'] == client['age_bins'].values[0]].index

        colors = ['lightslategray',] * len(ages_decinnie['Nombre de clients par Decennie'])
        colors[idx_decinnie.values[0]] = 'crimson'

        fig = go.Figure(data=[go.Bar(
            x=ages_decinnie['Ages par Decennie'],
            y=ages_decinnie['Nombre de clients par Decennie'],
            marker_color=colors # marker color can be a single color value or an iterable
        )])
        fig.update_layout(title_text='Nombre de Clients par Décinnie')

        st.plotly_chart(fig)

        












#--------------------------- model analysis -------------------------
### Confusion matrixe
def matrix_confusion (X,y):
    cm = confusion_matrix(X, y)
    print('\nTrue Positives(TP) = ', cm[0,0])
    print('\nTrue Negatives(TN) = ', cm[1,1])
    print('\nFalse Positives(FP) = ', cm[0,1])
    print('\nFalse Negatives(FN) = ', cm[1,0])
    return  cm

def show_model_analysis():
    conf_mtx = matrix_confusion(y_pred_test_export['y_test'],y_pred_test_export['y_predicted'])
    #st.write(conf_mtx)
    fig = go.Figure(data=go.Heatmap(
                   z=conf_mtx,
                    x=[ 'Actual Negative:0','Actual Positive:1'],
                   y=['Predict Negative:0','Predict Positive:1'],
                   hoverongaps = False))
    st.plotly_chart(fig)

    fpr, tpr, thresholds = roc_curve(y_pred_test_export['y_test'],y_pred_test_export['y_probability'])

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    st.plotly_chart(fig)
















def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Selectionner un fichier client', filenames)
    return os.path.join(folder_path, selected_filename)

def show_client_prediction():
    st.subheader("Selectionner source des données du client")
    selected_choice = st.radio("",('Client existant dans le dataset','Nouveau client'))

    if selected_choice == 'Client existant dans le dataset':
        client_id = st.number_input("Donnez Id du Client",100002)
        if st.button('Prédire Client'):
            API_url = "https://banking2023.herokuapp.com/credit/" + str(client_id)
            with st.spinner('Chargement du score du client...'):
                json_url = urlopen(API_url)
                API_data = json.loads(json_url.read())
                y_pred   = API_data['prediction']
                y_proba  = API_data['proba']
            st.info('Probabilité de solvabilité du client : '+str(100*y_proba)+' %')
            st.info("Notez que 100% => Client non slovable ")

            if(y_proba<seuil_risque):
                st.success('Client prédit comme solvable')
            if(y_proba>=seuil_risque):
                st.error('Client prédit comme non solvable !')

    if selected_choice == 'Nouveau client':   
        filename = file_selector()
        st.write('Fichier du nouveau client selectionné `%s`' % filename)
        
        if st.button('Prédire Client'):
            nouveau_client = pd.read_csv(filename)
            with st.spinner('Chargement du score du client...'):
                json_url = urlopen(API_url)
                API_data = json.loads(json_url.read())
                y_pred   = API_data['prediction']
                y_proba  = API_data['proba']
            st.info('Probabilité de solvabilité du client : '+str(100*y_proba)+' %')
            st.info("Notez que 100% => Client non slovable ")
            
            if(y_proba<seuil_risque):
                st.success('Client prédit comme solvable')
            if(y_proba>=seuil_risque):
                st.error('Client prédit comme non solvable !')
                
                
# --------------------------------------------------------------------
# FACTEURS D'INFLUENCE : SHAP VALUE
# --------------------------------------------------------------------
    
def affiche_facteurs_influence():
    ''' Affiche les facteurs d'influence du client courant
    '''
    html_facteurs_influence="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Variables importantes
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Voir facteurs d\'influence"):     
        
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)

        with st.spinner('**Affiche les facteurs d\'influence du client courant...**'):                 
                       
            
                
                explainer = shap.TreeExplainer(best_model)
                id_input = st.number_input("Donnez Id du Client",100001)
                client_index = test_set[test_set['SK_ID_CURR'] == id_input].index.item()
                X_shap = test_set.set_index('SK_ID_CURR')
                X_test_courant = X_shap.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                
                col1, col2 = st.columns([1, 1])
                # BarPlot du client courant
                with col1:

                    plt.clf()
                    

                    # BarPlot du client courant
                    shap.plots.bar(shap_values[client_index], max_display=40)
                    
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Décision plot du client courant
                with col2:
                    plt.clf()

                    # Décision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1],
                                    X_test_courant)
                
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)
                
             
                        





# ====================================================================
# IMAGES
# ====================================================================    
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
        
    

# ====================================================================
# FOOTER
# ====================================================================
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
