import streamlit as st
import pickle
import numpy as np
import  math

pipe=pickle.load(open('pipe.pkl','rb'))
data=pickle.load(open('data.pkl','rb'))
st.title("Laptop Price Predictor")
company=st.selectbox('Brands',data["Company"].unique())
type=st.selectbox("Type",data["TypeName"].unique())
ram=st.selectbox("RAM(in GB)",[2,4,6,8,12,16,24,32,64])
weight =st.number_input("Weight of Laptop")
touchscreen =st.selectbox("TouchScreen",['NO','YES'])
IPS =st.selectbox("IPS",['NO','YES'])
screen_size =st.number_input("Screen_size")
resolution=st.selectbox("Screen  Resolutions",['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu=st.selectbox("Brand",data["CPU Brand"].unique())
hdd=st.selectbox("HDD(in GB)",[0,128,256,512,1024,2048])
ssd=st.selectbox("SSD(in GB)",[0,8,128,256,512,1024])
gpu=st.selectbox("GPU",data["Gpu"].unique())
os=st.selectbox("OS",data['os'].unique())
if st.button("Predict Price"):
    if touchscreen=='YES':
        touchscreen=1
    else:
        touchscreen=0
    if IPS=='YES':
        IPS=1
    else:
        IPS=0

    x_res=int(resolution.split('x')[0])
    y_res=int(resolution.split('x')[1])

    ppi =(math.sqrt(math.pow(x_res,2)+math.pow(y_res,2)))/screen_size


    query=np.array([company,type,ram,gpu,weight,touchscreen,IPS,ppi,cpu,hdd,ssd,os])
    query=query.reshape(1,12)
    st.title(int(np.exp(pipe.predict(query)[0])))

