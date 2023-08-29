import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title='ROV Prediction', page_icon=':chart_with_upwards_trend:', layout='wide', initial_sidebar_state='collapsed')

# Define the light blue background color
background_color = '#e6f3ff'
st.markdown(f"""<style>body{{background-color: {background_color};}}</style>""", unsafe_allow_html=True)

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    available_features = df.columns.tolist()

    selected_categorical_features = st.multiselect('Select Categorical Columns', [f for f in available_features if df[f].dtype == 'object'])
    selected_numerical_features = st.multiselect('Select Numerical Columns', [f for f in available_features if df[f].dtype != 'object' and df[f].dtype != 'datetime64[ns]'])
    
    if not selected_categorical_features and not selected_numerical_features:
        st.write("Please select at least one feature.")
    else:
        X_categorical = df[selected_categorical_features]
        X_numerical = df[selected_numerical_features]
        
        # Convert categorical columns using one-hot encoding
        encoder = OneHotEncoder(sparse=False, drop='first')
        X_categorical_encoded = encoder.fit_transform(X_categorical)
        X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoder.get_feature_names(selected_categorical_features))
        
        # Concatenate one-hot encoded categorical and numerical features
        X = pd.concat([X_categorical_encoded_df, X_numerical], axis=1)
        
        y = df['Fiber_Cut']  # Replace 'target_column' with your actual target column name
        
        # Split dataset into train and test sets, stratified by the target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Initialize XGBoost model
        xgb_model = xgb.XGBClassifier(random_state=42)
        
        # Fit the model
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        xgb_preds = xgb_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, xgb_preds)
        recall = recall_score(y_test, xgb_preds)
        precision = precision_score(y_test, xgb_preds)
        f1 = f1_score(y_test, xgb_preds)
        
        # Formula for metrics
        accuracy_formula = "Accuracy = (TP + TN) / (TP + TN + FP + FN)"
        recall_formula = "Recall = TP / (TP + FN)"
        precision_formula = "Precision = TP / (TP + FP)"
        f1_formula = "F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"
        
        # Create a DataFrame to store the metrics
        model_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Recall', 'Precision', 'F1-Score'],
            'Value': [accuracy, recall, precision, f1],
            'Formula': [accuracy_formula, recall_formula, precision_formula, f1_formula]
        })
        
        st.header("ROV Prediction")
        st.write("XGBoost Model Metrics:")
        st.write(model_metrics)

        # Display confusion matrix as table
        st.subheader("Confusion Matrix")
        confusion_mat = confusion_matrix(y_test, xgb_preds)
        confusion_df = pd.DataFrame(confusion_mat)
        confusion_df

         # Add headers to rows and columns
        row_headers = ['Predicted 0', 'Predicted 1']
        col_headers = ['Actual 0', 'Actual 1']
    
        confusion_df = pd.DataFrame(confusion_mat, columns=col_headers, index=row_headers)
        st.table(confusion_df.style.set_table_attributes("style='display:inline'"))

        # Display confusion matrix plot
        #st.subheader("Confusion Matrix Plot")
        #plt.figure(figsize=(4, 3))
        #sns.heatmap(confusion_mat, annot=True, fmt="d", confusion_matap="Blues", cbar=False, annot_kws={"size": 5})
        #plt.xlabel("Predicted" , fontsize=3)
        #plt.ylabel("Actual" , fontsize=3)
        #plt.title("Confusion Matrix", fontsize=4)
        # Adjust plot size
        #ax = plt.gca()
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.25, box.height * 0.25])
        #st.pyplot()
        
        
        # Display feature importances
        st.subheader("Key Influencers on Accuracy:")
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': xgb_model.feature_importances_
        })
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        st.write(feature_importances)

        # Removing warning from below bar plots
        st.set_option('deprecation.showPyplotGlobalUse', False)

       # Display feature importance bar plot (top 25 features)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importances[:25], x='Importance', y='Feature')
        plt.title("Top 25 Key Influencers on Accuracy")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.xticks(rotation=45, ha="right")
        st.pyplot()
