import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import os

# Load the model
model = pickle.load(open('./Model/ML_Model.pkl', 'rb'))

# File to store loan applications
loan_data_file = 'loan_applications.csv'

# Function to save loan application data
def save_application(data):
    df = pd.DataFrame([data])
    if os.path.exists(loan_data_file):
        df.to_csv(loan_data_file, mode='a', header=False, index=False)
    else:
        df.to_csv(loan_data_file, index=False)

# Function to display admin panel
def display_admin_panel():
    st.subheader("Admin Panel")
    if os.path.exists(loan_data_file) and not pd.read_csv(loan_data_file).empty:
        df = pd.read_csv(loan_data_file)
        st.dataframe(df)
    else:
        st.write("No loan applications yet.")

def loan_form(loan_type):
    st.subheader(f"{loan_type} Application Form")

    account_no = st.text_input('Account Number')
    if len(account_no) != 13 or not account_no.isdigit():
        st.warning("Account number must be a 13-digit numeric value.")
        st.stop()

    fn = st.text_input('Full Name')

    gen_display = ('Female', 'Male')
    gen = st.selectbox("Gender", gen_display)

    mar_display = ('No', 'Yes')
    mar = st.selectbox("Marital Status", mar_display)

    dep_display = ('No', 'One', 'Two', 'More than Two')
    dep = st.selectbox("Dependents", dep_display)

    edu_display = ('Not Graduate', 'Graduate')
    edu = st.selectbox("Education", edu_display)

    emp_display = ('Job', 'Business')
    emp = st.selectbox("Employment Status", emp_display)

    prop_display = ('Rural', 'Semi-Urban', 'Urban')
    prop = st.selectbox("Property Area", prop_display)

    cred_display = ('Between 300 to 500', 'Above 500')
    cred = st.selectbox("Credit Score", cred_display)

    mon_income = st.number_input("Applicant's Monthly Income($)", min_value=0)
    co_mon_income = st.number_input("Co-Applicant's Monthly Income($)", min_value=0)
    loan_amt = st.number_input("Loan Amount", min_value=0)

    dur_display = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
    dur = st.selectbox("Loan Duration", dur_display)

    # Map loan duration to numeric months
    duration_map = {'2 Month': 60, '6 Month': 180, '8 Month': 240, '1 Year': 360, '16 Month': 480}
    duration = duration_map[dur]

    # Prediction Logic
    if st.button("Submit"):
        features = [[gen_display.index(gen), mar_display.index(mar), dep_display.index(dep),
                     edu_display.index(edu), emp_display.index(emp), mon_income,
                     co_mon_income, loan_amt, duration, cred_display.index(cred), prop_display.index(prop)]]
        prediction = model.predict(features)

        result = "Approved" if prediction == 1 else "Rejected"

        # Save application data
        application_data = {
            'Account Number': account_no,
            'Full Name': fn,
            'Gender': gen,
            'Marital Status': mar,
            'Dependents': dep,
            'Education': edu,
            'Employment': emp,
            'Income': mon_income,
            'Co-Applicant Income': co_mon_income,
            'Loan Amount': loan_amt,
            'Duration (Months)': duration,
            'Credit Score': cred,
            'Property Area': prop,
            'Loan Type': loan_type,
            'Status': result
        }
        save_application(application_data)

        if result == "Rejected":
            st.error(f"Hello {fn}, your {loan_type} application was **not approved**.")
        else:
            st.success(f"Hello {fn}, congratulations! Your {loan_type} application was **approved**.")

def run():
    # Layout for top-right Admin button
    col1, col2 = st.columns([8, 1])  # Adjust the column ratio to push the button to the right
    with col2:
        if st.button("Admin Panel"):
            display_admin_panel()

    img1 = Image.open('bank.png')
    img1 = img1.resize((156, 145))
    st.image(img1, use_column_width=False)
    st.title("Bank Loan Prediction using Machine Learning")

    # Loan Type Selection
    loan_types = ['Home Loan', 'Car Loan', 'Personal Loan', 'Education Loan']
    selected_loan = st.selectbox("Select Loan Type", loan_types)

    # Display the form based on loan type selection
    if selected_loan:
        loan_form(selected_loan)

run()
