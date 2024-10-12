import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, log_loss, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import pickle
import os
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)

st.set_page_config(
    page_title="🤖Automate ML🤖",
    page_icon="",
    layout="centered"
)

st.title("🤖No Code 🤖 ML Model Training🤖")
page = option_menu("CHOOSE BELOW", ["Train Model","Best Model Performance"], 
                   icons=["bi bi-file-earmark-binary", "bi bi-graph-up"], 
                   menu_icon="bi bi-menu-button-wide", default_index=0)
                   
dataset_list = os.listdir(f"{parent_dir}/data")

dataset = st.selectbox("Select a dataset from the dropdown", dataset_list, index=None)

if dataset is not None:
    dataset_path = f"{parent_dir}/data/{dataset}"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        st.error("The selected dataset does not exist")
else:
    st.error("Please select a dataset")

if 'pd' in locals() and 'df' in locals():
    if df is not None:
        st.dataframe(df.head())

        # Remove ID column if it exists
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        if id_columns:
            df = df.drop(id_columns, axis=1)

        # Remove columns with non-numeric values
        non_numeric_columns = df.select_dtypes(include=['object']).columns
        df = df.drop(non_numeric_columns, axis=1)

        # Handle missing values
        def handle_missing_values(df):
            imputer = SimpleImputer(strategy='mean')
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            return df_imputed

        df = handle_missing_values(df)

        col1, col2, col3, col4 = st.columns(4)

        scaler_type_list = ["standard", "minmax", "robust", "maxabs", "mean", "median"]

        model_dictionary = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": XGBClassifier(),
            "LightGBM Classifier": LGBMClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier()
        }

        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Model name", value=f"{scaler_type} {selected_model}")
        # Hyperparameter Tuning
        if selected_model == "Logistic Regression":
            C = st.slider("C", 0.1, 10.0, 1.0)
            model_dictionary[selected_model] = LogisticRegression(C=C)
        elif selected_model == "Support Vector Classifier":
            C = st.slider("C", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            model_dictionary[selected_model] = SVC(C=C, kernel=kernel)
        elif selected_model == "Random Forest Classifier":
            n_estimators = st.slider("n_estimators", 10, 100, 50)
            max_depth = st.slider("max_depth", 5, 20, 10)
            model_dictionary[selected_model] = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        elif selected_model == "XGBoost Classifier":
            learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1)
            n_estimators = st.slider("n_estimators", 10, 100, 50)
            max_depth = st.slider("max_depth", 5, 20, 10)
            model_dictionary[selected_model] = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
        elif selected_model == "LightGBM Classifier":
            learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1)
            n_estimators = st.slider("n_estimators", 10, 100, 50)
            max_depth = st.slider("max_depth", 5, 20, 10)
            model_dictionary[selected_model] = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
        elif selected_model == "AdaBoost Classifier":
            n_estimators = st.slider("n_estimators", 10, 100, 50)
            learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1)
            model_dictionary[selected_model] = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        elif selected_model == "Gradient Boosting Classifier":
            n_estimators = st.slider("n_estimators", 10, 100, 50)
            learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1)
            max_depth = st.slider("max_depth", 5, 20, 10)
            model_dictionary[selected_model] = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

        # Oversampling
        oversampling_technique = st.selectbox("Oversampling Technique", ["None", "SMOTE", "ADASYN"])

        if st.button("Train Model"):
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            # Oversampling
            if oversampling_technique == "SMOTE":
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
            elif oversampling_technique == "ADASYN":
                adasyn = ADASYN(random_state=42)
                X, y = adasyn.fit_resample(X, y)

            # Scaling
            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif scaler_type == "robust":
                scaler = RobustScaler()
            elif scaler_type == "maxabs":
                scaler = MaxAbsScaler()
            elif scaler_type == "mean":
                scaler = StandardScaler(with_mean=True)
            elif scaler_type == "median":
                scaler = StandardScaler(with_std=False)

            X = scaler.fit_transform(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = model_dictionary[selected_model]
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            log_loss_value = log_loss(y_test, y_pred)

            # Print the evaluation metrics
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            st.write("F1 Score:", f1)
            st.write("ROC AUC:", roc_auc)
            st.write("Balanced Accuracy:", balanced_accuracy)
            st.write("Log Loss:", log_loss_value)

            # Plot the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.colorbar()
            st.pyplot(plt)

            # Plot the ROC curve
            if roc_auc >= 0.9:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
                plt.legend(loc="lower right")
                st.pyplot(plt)
            else:
                # Plot the precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve (area = %0.2f)' % roc_auc)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall curve')
                plt.legend(loc="lower right")
                st.pyplot(plt)

            # Plot the confusion matrix with TP, TN, FP, FN
            plt.figure(figsize=(8, 6))
            plt.bar(['TP', 'TN', 'FP', 'FN'], [cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]], color=['green', 'blue', 'red', 'orange'])
            plt.xlabel('Quadrant')
            plt.ylabel('Count')
            plt.title('Confusion Matrix Quadrants')
            st.pyplot(plt)
            # Save the model
            model_dir = f"{parent_dir}/models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            with open(f"{model_dir}/{model_name}.pkl", "wb") as f:
                pickle.dump(model, f)
            st.success("Model saved successfully")

                # Store the model performances in an Excel sheet
            model_performance = {
                    "Dataset Name": [dataset],
                    "Scaler Type": [scaler_type],
                    "Algorithm Used": [selected_model],
                    "Model Name": [model_name],
                    "Accuracy": [accuracy],
                    "Precision": [precision],
                    "Recall": [recall],
                    "F1 Score": [f1],
                    "ROC AUC": [roc_auc],
                    "Balanced Accuracy": [balanced_accuracy],
                    "Log Loss": [log_loss_value]
            }

            model_performance_df = pd.DataFrame(model_performance)

            model_performance_file_path = f"{parent_dir}/model_performance.xlsx"

            if os.path.exists(model_performance_file_path):
                  existing_model_performance_df = pd.read_excel(model_performance_file_path)
                  updated_model_performance_df = pd.concat([existing_model_performance_df, model_performance_df])
                  updated_model_performance_df.to_excel(model_performance_file_path, index=False)
            else:
                  model_performance_df.to_excel(model_performance_file_path, index=False)

            st.success("Model performance stored in Excel sheet")
            # Create a new page

if page == "Best Model Performance":
    # Read the Excel file
    model_performance_file_path = f"{parent_dir}/model_performance.xlsx"
    if os.path.exists(model_performance_file_path):
        model_performance_df = pd.read_excel(model_performance_file_path)
                      
        
        # Display the best model
        best_model = model_performance_df.loc[model_performance_df["Accuracy"].idxmax()]
        st.write("Best Model:")
        st.write("Model Name:", best_model["Model Name"])
        st.write("Accuracy:", best_model["Accuracy"])
        st.write("Precision:", best_model["Precision"])
        st.write("Recall:", best_model["Recall"])
        st.write("F1 Score:", best_model["F1 Score"])
        st.write("ROC AUC:", best_model["ROC AUC"])
    else:
        st.error("No model performance data available")