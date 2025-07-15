import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('ML_MODEL/knn_model.pkl')

# Try to extract feature names used during training
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("❌ Model doesn't contain feature names. Re-train with feature_names_in_.")
    st.stop()

def main():
    # Configure the Streamlit page
    st.set_page_config(page_title="Student Pass/Fail Predictor", layout="wide")

    # Title
    st.title("🎓 Student Pass/Fail Predictor")

    # Create two tabs: one for input, one for prediction
    tab1, tab2 = st.tabs(["🧑‍🎓 Enter Student Data", "📈 Prediction & Report"])

    # -------------------- TAB 1: Data Input --------------------
    with tab1:
        st.subheader("🧾 Fill Student Details Below")

        # Use columns for neat layout
        col1, col2 = st.columns(2)

        with col1:
            student_name = st.text_input('🧑 Student Name')
            sex = st.selectbox("Sex", ['Female', 'Male'])
            age = st.slider("Age", 15, 22, 17)
            address = st.radio("Address", ['Urban', 'Rural'], horizontal=True)
            famsize = st.radio("Family Size", ['3 or Less', 'More than 3'], horizontal=True)
            Pstatus = st.selectbox("Parents' Status", ['Living Together', 'Apart'])
            Medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
            Fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
            Mjob = st.selectbox("Mother's Job", ['Teacher', 'Healthcare', 'Services', 'At Home', 'Other'])
            Fjob = st.selectbox("Father's Job", ['Teacher', 'Healthcare', 'Services', 'At Home', 'Other'])

        with col2:
            reason = st.selectbox("School Choice Reason", ['Close to Home', 'School Reputation', 'Course Preference', 'Other'])
            guardian = st.selectbox("Guardian", ['Mother', 'Father', 'Other'])
            traveltime = st.slider("Travel Time (1-4)", 1, 4, 2)
            studytime = st.slider("Study Time (1-4)", 1, 4, 2)
            failures = st.slider("Past Failures", 0, 3, 0)
            schoolsup = st.selectbox("School Support", ['No', 'Yes'])
            famsup = st.selectbox("Family Support", ['No', 'Yes'])
            paid = st.selectbox("Paid Classes", ['No', 'Yes'])
            activities = st.selectbox("Extra Activities", ['No', 'Yes'])
            nursery = st.selectbox("Attended Nursery", ['No', 'Yes'])
            higher_edu = st.selectbox("Wants Higher Ed", ['No', 'Yes'])
            internet = st.selectbox("Internet Access", ['No', 'Yes'])
            romantic = st.selectbox("Romantic Relationship", ['No', 'Yes'])

        col3, col4 = st.columns(2)
        with col3:
            famrel = st.slider("Family Relationship", 1, 5, 3)
            freetime = st.slider("Free Time", 1, 5, 3)
            goout = st.slider("Going Out", 1, 5, 3)
        with col4:
            Dalc = st.slider("Workday Alcohol", 1, 5, 3)
            Walc = st.slider("Weekend Alcohol", 1, 5, 3)
            health = st.slider("Health Status", 1, 5, 3)
            G3 = st.slider("Final Grade (G3)", 1, 20, 10)
            GPA = st.slider("GPA (0.0 - 4.0)", 0.0, 4.0, 2.0)
            absences = st.slider("Absences", 0, 50, 10)

    # -------------------- TAB 2: Prediction --------------------
    with tab2:
        st.subheader("📊 Prediction and Report")

        # Encode categorical variables
        sex = 1 if sex == 'Female' else 0
        address = 1 if address == 'Urban' else 0
        famsize = 1 if famsize == '3 or Less' else 0
        Pstatus = 1 if Pstatus == 'Living Together' else 0
        Mjob = {'At Home': 0, 'Healthcare': 1, 'Other': 2, 'Services': 3, 'Teacher': 4}[Mjob]
        Fjob = {'At Home': 0, 'Healthcare': 1, 'Other': 2, 'Services': 3, 'Teacher': 4}[Fjob]
        reason = {'Close to Home': 0, 'Course Preference': 1, 'Other': 2, 'School Reputation': 3}[reason]
        guardian = {'Father': 0, 'Mother': 1, 'Other': 2}[guardian]
        schoolsup = 1 if schoolsup == 'Yes' else 0
        famsup = 1 if famsup == 'Yes' else 0
        paid = 1 if paid == 'Yes' else 0
        activities = 1 if activities == 'Yes' else 0
        nursery = 1 if nursery == 'Yes' else 0
        higher = 1 if higher_edu == 'Yes' else 0
        internet = 1 if internet == 'Yes' else 0
        romantic = 1 if romantic == 'Yes' else 0

        # Prepare input dataframe
        input_data = pd.DataFrame({
            'sex_F': [sex], 'sex_M': [1 - sex],
            'age': [age],
            'address_R': [1 - address], 'address_U': [address],
            'famsize_GT3': [1 - famsize], 'famsize_LE3': [famsize],
            'Pstatus_A': [1 - Pstatus], 'Pstatus_T': [Pstatus],
            'Mjob_at_home': [1 if Mjob == 0 else 0], 'Mjob_health': [1 if Mjob == 1 else 0],
            'Mjob_other': [1 if Mjob == 2 else 0], 'Mjob_services': [1 if Mjob == 3 else 0],
            'Mjob_teacher': [1 if Mjob == 4 else 0],
            'Fjob_at_home': [1 if Fjob == 0 else 0], 'Fjob_health': [1 if Fjob == 1 else 0],
            'Fjob_other': [1 if Fjob == 2 else 0], 'Fjob_services': [1 if Fjob == 3 else 0],
            'Fjob_teacher': [1 if Fjob == 4 else 0],
            'reason_course': [1 if reason == 1 else 0], 'reason_home': [1 if reason == 0 else 0],
            'reason_other': [1 if reason == 2 else 0], 'reason_reputation': [1 if reason == 3 else 0],
            'guardian_father': [1 if guardian == 0 else 0], 'guardian_mother': [1 if guardian == 1 else 0],
            'guardian_other': [1 if guardian == 2 else 0],
            'schoolsup_no': [1 - schoolsup], 'schoolsup_yes': [schoolsup],
            'famsup_no': [1 - famsup], 'famsup_yes': [famsup],
            'paid_no': [1 - paid], 'paid_yes': [paid],
            'activities_no': [1 - activities], 'activities_yes': [activities],
            'nursery_no': [1 - nursery], 'nursery_yes': [nursery],
            'higher_no': [1 - higher], 'higher_yes': [higher],
            'internet_no': [1 - internet], 'internet_yes': [internet],
            'romantic_no': [1 - romantic], 'romantic_yes': [romantic],
            'Medu': [Medu], 'Fedu': [Fedu], 'studytime': [studytime], 'failures': [failures],
            'famrel': [famrel], 'freetime': [freetime], 'goout': [goout],
            'Dalc': [Dalc], 'Walc': [Walc], 'health': [health],
            'G3': [G3], 'GPA': [GPA], 'absences': [absences], 'traveltime': [traveltime]
        })

        # Ensure input columns match training columns
        input_data = input_data[expected_columns]

        # Predict when button is clicked
        if st.button("🚀 Predict"):
            prediction = model.predict(input_data)
            prob = model.predict_proba(input_data)[0][1]
            result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"

            st.success(f"🎯 Prediction for {student_name}: {result}")
            st.metric("📈 Probability of Passing", f"{prob:.2%}")
            st.metric("📘 GPA", f"{GPA:.2f}")

            # Charts
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            sns.barplot(x=['Fail', 'Pass'], y=[1 - prob, prob], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title("Pass/Fail Probability")

            axes[1].pie([1 - prob, prob], labels=['Fail', 'Pass'], autopct='%1.1f%%', colors=['#f44336', '#4CAF50'])
            axes[1].set_title("Pass/Fail Pie Chart")

            st.pyplot(fig)

            # Prepare downloadable report
            report_text = f"""
Student Report - {student_name}
-------------------------------
Prediction: {result}
Probability of Passing: {prob:.2%}
GPA: {GPA:.2f}
Final Grade (G3): {G3}
Age: {age}
Absences: {absences}
"""
            st.download_button("📄 Download Report", data=report_text, file_name=f"{student_name}_report.txt", mime='text/plain')

# Standard Python entry point
if __name__ == '__main__':
    main()
