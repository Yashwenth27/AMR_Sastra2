import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_navigation_bar import st_navbar
from mlxtend.frequent_patterns import apriori, association_rules
import os

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)

navbar_html = '''
<style>
    .st-emotion-cache-h4xjwg{
        z-index: 100;
    }
    .css-hi6a2p {padding-top: 0rem;}
    .navbar {
        background-color: #007BFF;
        padding: 0.3rem;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .navbar .logo {
        display: flex;
        align-items: center;
    }
    .navbar .logo img {
        height: 40px;
        margin-right: 10px;
    }
    .navbar .menu {
        display: flex;
        gap: 1.5rem;
    }
    .navbar .menu a {
        color: white;
        font-size: 1.2rem;
        text-decoration: none;
    }
    .content {
        padding-top: 5rem;
    }
</style>

<nav class="navbar">
    <div class="logo">
        <h3>Antimicrobial Resistant Dashboard</h3>
    </div>
    <div class="menu">
        <a href="">Dashboard</a>
        <a href="mvp">Multi Variable Plots</a>
        <a href="chatbot">ChatBot</a>
        <a href="mlp">ML Predictions</a>
        <a href="ml-predictions">About Us</a>
    </div>
</nav>

<div class="content">
'''

st.markdown(navbar_html, unsafe_allow_html=True)

orgcsv = {
    "Ecoli": "1weNBHFde03wQMs-kstnml9_61gK0ASDa"
}

st.title("Associative Mining On Customised Parameters")
st.markdown(
    """
    <style>
    hr.blue-neon {
        border: none;
        height: 2px;
        background-color: #007BFF; /* Blue color */
        margin: 20px 0;
    }
    </style>
    <hr class="blue-neon">
    """, unsafe_allow_html=True
)

a, b = st.columns([0.3, 0.7])

with a:
    with st.container():
        orgcsvopt = st.selectbox("Choose Organism:", ["Ecoli"])
        param_chosen = st.selectbox("Choose Basis:", ["Year", "Country"])
        output = 'Ecoli_corrected_data_1ver.csv'

        # Check if the file already exists locally
        if not os.path.exists(output):
            import gdown
            with b:
                with st.spinner("Downloading Datasets"):
                    url = f'https://drive.google.com/uc?id={orgcsv[orgcsvopt]}' 
                    gdown.download(url, output, quiet=False)
        else:
            st.write("Using the local copy of the dataset.")

        if param_chosen == "Year":
            year = st.slider("Choose Year", 2004, 2023)
            min_sup = st.slider("Choose Minimum support", 0.1, 1.0, step=0.01)
        elif param_chosen == "Country":
            df = pd.read_csv(output, low_memory=False)
            country_list = df['Country'].unique().tolist()
            country = st.selectbox("Choose Country:", country_list)
            min_sup = st.slider("Choose Minimum support", 0.1, 1.0, step=0.01)

with b:
    with st.spinner("Loading Datasets"):
        import time
        time.sleep(3)
    
    df = pd.read_csv(output, low_memory=False)
    df3 = df.copy()

    param = {
        "Year": ['Isolate Id', 'Study', 'Species', 'Family', 'Country', 'State', 'Gender', 'Age Group', 'Speciality', 'Source', 'In / Out Patient'],
        "Country": ['Isolate Id', 'Study', 'Species', 'Family', 'Year', 'State', 'Gender', 'Age Group', 'Speciality', 'Source', 'In / Out Patient']
    }

    s = list(df3.columns[df3.columns.str.contains("_I")])
    s.append(param_chosen)
    df2 = df[s]
    org_association_data = df2.copy()

    if param_chosen == "Year":
        org_year = org_association_data[org_association_data.Year.isin([year])]
        for column in org_year.columns:
            if org_year[column].notnull().sum() == 0:
                org_year.drop(column, axis=1, inplace=True)
        org_year.drop(['Year'], axis=1, inplace=True)

        org_converted = pd.get_dummies(org_year, dtype="bool")
        freq_org = apriori(org_converted, min_support=0.3, use_colnames=True, low_memory=True, verbose=2, max_len=5)
        rules = association_rules(freq_org)
        st.subheader(str(rules.shape[0]) + " Rules found!")
        st.write(rules)

        csv = rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Organism Association Rules📥",
            data=csv,
            file_name="Org_Association_Rules_Year.csv",
            mime='text/csv'
        )
    
    elif param_chosen == "Country":
        org_country = org_association_data[org_association_data.Country == country]
        for column in org_country.columns:
            if org_country[column].notnull().sum() == 0:
                org_country.drop(column, axis=1, inplace=True)
        org_country.drop(['Country'], axis=1, inplace=True)
        org_converted = pd.get_dummies(org_country, dtype="bool")
        freq_org = apriori(org_converted, min_support=0.3, max_len=5,use_colnames=True, low_memory=True, verbose=2)
        rules = association_rules(freq_org)
        st.subheader(str(rules.shape[0]) + " Rules found!")
        st.write(rules)

        csv = rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Organism Association Rules📥",
            data=csv,
            file_name="Org_Association_Rules_Country.csv",
            mime='text/csv'
        )
