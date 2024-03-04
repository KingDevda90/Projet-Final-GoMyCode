import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------- Importation des données -------#
Mydata = pd.read_csv('/Users/mac/Downloads/ProjetMLNet.csv', sep=',')

st.title('Diagnostic Prediction')
st.subheader('Training data')
st.write(Mydata)

X = Mydata.drop('Diagnostic', axis=1)
y = Mydata['Diagnostic']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def user_report():
    # Définir les variables
    codetection = st.checkbox("Codetection")
    influenza = st.checkbox("Influenza")
    adenovirus = st.checkbox("Adenovirus")
    bocavirus = st.checkbox("Bocavirus")
    coronavirus = st.checkbox("Coronavirus")
    enterovirus = st.checkbox("Enterovirus")
    metapneumonia = st.checkbox("Metapneumonia")
    parainfluenza = st.checkbox("ParaInfluenza")
    rhinovirus = st.checkbox("Rhinovirus")
    covid = st.checkbox("Sarscov_2")
    vrs = st.checkbox("VRS")

    fievre = st.checkbox("FIEVRE")
    toux = st.checkbox("TOUX")
    rhinite = st.checkbox("RHINITE")
    pharyngite = st.checkbox("PHARINGITE")
    cephalees = st.checkbox("CEPHALEES")
    conjonctivite = st.checkbox("CONJONCTIVITE")
    myalgia = st.checkbox("Myalgia")
    diff_res = st.checkbox("Diff_Res")
    algies = st.checkbox("Algies")
    arthralgies = st.checkbox("Arthralgies")
    vomissement_diarhee = st.checkbox("Vomissement-Diarhee")

    # Afficher les symptômes dans une multibox

    # Afficher les maladies dans un box
    symptomes = st.multiselect("Symptômes",
                                         [fievre, toux, rhinite, pharyngite, cephalees, conjonctivite, myalgia, diff_res,
                                         algies, arthralgies, vomissement_diarhee])
    virus = st.checkbox("Virus", [codetection, influenza, adenovirus, bocavirus, coronavirus, enterovirus, metapneumonia,
                                          parainfluenza, rhinovirus, covid, vrs])
    age = st.sidebar.slider("Age", 0, 100, 5)
    temperature = st.sidebar.slider("TEMPERATURE", 20.0, 45.0, 36.5, step=0.1)

    user_report = {
        'symptomes': symptomes,
        'virus': virus,
        'age': age,
        'temperature': temperature

    }

    report_data = pd.DataFrame(user_report)
    return report_data

user_data = user_report()

model = RandomForestClassifier()
model.fit(X_train, y_train)
st.subheader("Accuracy : ")
st.write(str(accuracy_score(y_test, model.predict(X_test))*100)+'%')
user_result = model.predict(user_data)
