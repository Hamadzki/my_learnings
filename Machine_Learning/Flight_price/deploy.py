import pandas as pd
import numpy as np
import streamlit as st
import pickle
from streamlit_lottie import st_lottie 
import requests

st.set_page_config(page_title = 'Flight Prediction' , page_icon = ':relaxed:' ,layout='wide')

pipe = pickle.load(open("model.pkl" , "rb"))

df = pd.read_csv("Hammad/Hammad/Projects/Flight_price/data_cleaned.csv")
x = df.drop('Price' , axis = 1)
y = df['Price']
                 
# Animation
def load_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
#anm = load_url("https://assets7.lottiefiles.com/packages/lf20_xo7s9v.json")

col1, col2 = st.columns(2)

with col1:
    st.title("Flight Price Prediction")
    
#with col2:
#        st_lottie(anm , height= 100 , width = 100 )    

if st.button('Data Frame'):
    st.write(df.head() )
    
Airline = st.selectbox('Airline' , df['Airline'].unique() )

Source = st.selectbox('Source' , df['Source'].unique() )
 
Destination = st.selectbox('Destination' , df['Destination'].unique() )
 
Total_Stops = st.number_input('Total_Stops' , df['Total_Stops'].min() ,df['Total_Stops'].max() ) 

journey_day = st.number_input('journey_day' , df['journey_day'].min() ,df['journey_day'].max() )
 
journey_month = st.number_input('journey_month', df['journey_month'].min() ,df['journey_month'].max() )

dep_time_hour = st.number_input('dep_time_hour' , df['dep_time_hour'].min() ,df['dep_time_hour'].max() )

dep_time_min= st.number_input('dep_time_min' , df['dep_time_min'].min() ,df['dep_time_min'].max() )

arrival_time_hour = st.number_input('arrival_time_hour' , df['arrival_time_hour'].min() ,df['arrival_time_hour'].max() )

arrival_time_min = st.number_input('arrival_time_min' , df['arrival_time_min'].min() ,df['arrival_time_min'].max() )

Route_1 = st.selectbox('Route 1' , df['Route 1'].unique() )
 
Route_2 = st.selectbox('Route 2' , df['Route 2'].unique() )
 
Route_3 = st.selectbox('Route 3' , df['Route 3'].unique() )
 
Route_4 = st.selectbox('Route 4' , df['Route 4'].unique() )
 
Route_5 = st.selectbox('Route 5' , df['Route 5'].unique() )

Duration_hours = st.number_input('Duration_hours', df['Duration_hours'].min() ,df['Duration_hours'].max() )

Duration_mins = st.number_input('Duration_mins' , df['Duration_mins'].min() ,df['Duration_mins'].max() )

input_feature = [Airline,Source,Destination,Total_Stops,journey_day,
                 journey_month,dep_time_hour,dep_time_min,arrival_time_hour,
                 arrival_time_min,Route_1,Route_2,Route_3,Route_4,Route_5,
                 Duration_hours,Duration_mins]

input_array = np.array([input_feature])

input_data = pd.DataFrame(data = input_array , columns = x.columns)

st.write(input_data)

if st.button('Predict'):
    st.write(pipe.predict(input_data))

