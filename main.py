import streamlit as st
import pandas as pd
import joblib

model=joblib.load('knn_heart_model.pkl')
scaler=joblib.load('heart_scaler.pkl')
expected_columns=joblib.load('heart_columns.pkl')

st.title('heart stroke predicion by PRASHANTH A')
st.markdown('provide the folloeing details')

age=st.slider('AGE',18,100,40)
sex=st.selectbox('SEX',['M','F'])
chest_pain=st.selectbox('CHEST PAIN TYPE',['ATA','NAP','TA','ASY'])
resting_bp=st.number_input('RESTING BP(mg/dl)',80,200,120)
cholestrol=st.number_input('CHOLESTROL(mg/dl)',100,600,200)
fasting_bs=st.selectbox("FASTING BLOOD SUGAR>120 mg/dl",[0,1])
resting_ecg=st.selectbox('RESTING ECG',['Normal','ST','LVH'])
max_hr=st.slider('MAX HEART RATE',60,20,150)
exercise_angina=st.selectbox("EXERCISE-INDUCED ANGINA",['Y','N'])
oldpeak=st.slider('OLDPEAK(ST DEPRSSION)',0.0,6.0,1.0)
st_slope=st.selectbox('ST SLOPE',['UP','FLAT','DOWN'])


if st.button('Predict'):
    raw_input={
        'Age':age,
        'RestingBp':resting_bp,
        'Cholesterol':cholestrol,
        'FastingBS':fasting_bs,
        'MaxHR':max_hr,
        'Oldpeak':oldpeak,
        'Sex' +sex:1,
        'ChestPainType_' + chest_pain:1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope' + st_slope: 1

    }

    input_df=pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df=input_df[expected_columns]

    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]

    if prediction ==1:
        st.error('⚠️ HIGH RISK OF HEAR DISEASE')
    else:
        st.success('✅ LOW RISK OF HEART DISEASE')