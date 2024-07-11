import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pickle
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score





# Paths to CSV files
original_file_path = '/Users/tarunayikkara/Documents/Data Scientest/Energy Project/Streamlit/eco2mix-regional-cons-def.csv'
energy_revised_file_path = '/Users/tarunayikkara/Documents/Data Scientest/Energy Project/Streamlit/energy_revised.csv'
temperature_file_path = '/Users/tarunayikkara/Documents/Data Scientest/Energy Project/Streamlit/temperature-2.csv'
merged_file_path = '/Users/tarunayikkara/Documents/Data Scientest/Energy Project/Streamlit/merged_df.csv'

# Load the data
energy = pd.read_csv(original_file_path, sep=';')

# Rename columns
new_column_names = ['INSEE Region code', 'Region', 'Nature', 'Date', 'Time',
                    'Date - Time', 'Consumption (MW)', 'Thermal (MW)', 'Nuclear (MW)',
                    'Wind (MW)', 'Solar (MW)', 'Hydraulic (MW)', 'Pumping (MW)',
                    'Bioenergies (MW)', 'Ech. physiques (MW)', 'Stockage batterie',
                    'Battery destorage', 'Onshore wind', 'Offshore wind',
                    'TCO Thermal (%)', 'TCH Thermal (%)', 'TCO Nuclear (%)',
                    'TCH Nuclear (%)', 'TCO Wind (%)', 'TCH Wind (%)',
                    'TCO Solar (%)', 'TCH Solar (%)', 'TCO Hydropower (%)',
                    'TCH Hydraulic (%)', 'TCO Bioenergy (%)', 'TCH Bioenergy (%)',
                    'Column 30']
energy.columns = new_column_names

# Clean column names
def clean_column_names(df):
    df.columns = (df.columns
                  .str.strip()  # Remove leading/trailing whitespace
                  .str.replace(' ', '_')  # Replace spaces with underscores
                  .str.lower())  # Convert to lowercase
    return df

energy = clean_column_names(energy)

# Data cleaning and preprocessing
energy_revised = energy[['insee_region_code', 'region', 'nature', 'date_-_time',
                         'consumption_(mw)', 'thermal_(mw)', 'nuclear_(mw)', 'wind_(mw)',
                         'solar_(mw)', 'hydraulic_(mw)', 'pumping_(mw)', 'bioenergies_(mw)',
                         'ech._physiques_(mw)']]
energy_revised = energy_revised.drop_duplicates()

# Converting date_time column to datetime format
energy_revised['date_-_time'] = pd.to_datetime(energy_revised['date_-_time'], utc=True).dt.tz_convert('UTC')

# Cleaning wind_(mw) column
garbage_values = ['ND', '-']
energy_revised['wind_(mw)'] = energy_revised['wind_(mw)'].replace(garbage_values, np.nan).astype(float)
energy_revised['wind_(mw)'].fillna(energy_revised['wind_(mw)'].median(), inplace=True)

# Filling missing values in other columns with median
for c in energy_revised.columns:
    if energy_revised[c].isnull().any():
        energy_revised[c].fillna(energy_revised[c].median(), inplace=True)


# Energy revised new columns
energy_revised['New_date'] = pd.to_datetime(energy_revised['date_-_time']).dt.date
energy_revised['New_year'] = pd.to_datetime(energy_revised['date_-_time']).dt.year
energy_revised['New_month'] = pd.to_datetime(energy_revised['date_-_time']).dt.month


# Loading temperature data
temp = pd.read_csv(temperature_file_path, sep=';')
temp_new_columns = ['date_time', 'date', 'insee_region_code', 'region', 'tmin_c',
                    'tmax_c', 'tavg_c']
temp.columns = temp_new_columns
temp = clean_column_names(temp)


#Loading Merged data 
merged_df = pd.read_csv(merged_file_path, sep=',')


def Modelling():
    st.title("Model Comparison")
 
    data = {
            'Models': ['Linear Regression', 'Random Forest Regressor', 'Decision Tree Regressor', 'Lasso', 'Lasso CV', 'Ridge'],
            'R² train': [0.914, 0.959, 0.959, 0.916, 0.914, 0.911],
            'R² test': [0.916, 0.956, 0.954, 0.917, 0.915, 0.913],
            'MAE train': [470.61, 291.125, 297.03, 472.90, 470.95, 485.442],
            'MAE test': [468.51, 305.474, 316.28, 471.88, 469.24, 484.63],
            'MSE train': [365833.46, 183021.921, 185997.132, 380524.08, 367168.12, 400175.585],
            'MSE test': [364843.03, 200151.253, 208030.288, 379947.81, 366471.99, 399987.20],
            'RMSE train': [604.84, 427.810, 431.273, 616.86, 605.95, 632.594],
            'RMSE test': [604.02, 447.382, 456.103, 616.399, 605.37, 632.44]
        }
    df = pd.DataFrame(data)
    st.write(df)

    if st.button('Random Forest Regressor'):


        # Step 2: Instantiate the model with a maximum depth
        # Preprocess the data
        model = pd.read_csv(merged_file_path, sep=',')
        model = model.drop(['nature', 'date_-_time', 'New_date', 'datetime_'], axis=1)
        feats = model.drop('consumption_(mw)', axis=1)
        target = model['consumption_(mw)']
        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)

        # OneHotEncoding
        oneh = OneHotEncoder(drop='first', sparse_output=False)
        cat = ['region_x']
        encoded_train = oneh.fit_transform(X_train[cat])
        encoded_train_model = pd.DataFrame(encoded_train, columns=oneh.get_feature_names_out(cat), index=X_train.index)
        encoded_test = oneh.transform(X_test[cat])
        encoded_test_model = pd.DataFrame(encoded_test, columns=oneh.get_feature_names_out(cat), index=X_test.index)
        X_train = X_train.drop(cat, axis=1).join(encoded_train_model)
        X_test = X_test.drop(cat, axis=1).join(encoded_test_model)

        # Scaling
        sc = StandardScaler()
        num = ['thermal_(mw)', 'nuclear_(mw)', 'wind_(mw)', 'solar_(mw)', 'hydraulic_(mw)', 'pumping_(mw)', 'bioenergies_(mw)', 'ech._physiques_(mw)', 'tavg_(°c)', 'New_year', 'New_month']
        X_train[num] = sc.fit_transform(X_train[num])
        X_test[num] = sc.transform(X_test[num])

        # Random Forest Regressor
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)
        y_pred_test = rf_regressor.predict(X_test)
        y_pred_train = rf_regressor.predict(X_train)

            # Save the trained model
        with open('rf_regressor.pkl', 'wb') as file:
            pickle.dump(rf_regressor, file)

        # Save the scaler
        with open('scaler.pkl', 'wb') as file:
            pickle.dump(sc, file)

    

        # Evaluation
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(y_train, y_pred_train)

        st.write(f'Test Data - Mean Absolute Error (MAE): {mae_test}')
        st.write(f'Test Data - Mean Squared Error (MSE): {mse_test}')
        st.write(f'Test Data - Root Mean Squared Error (RMSE): {rmse_test}')
        st.write(f'Test Data - R-squared (R2): {r2_test}')
        st.write(f'Train Data - Mean Absolute Error (MAE): {mae_train}')
        st.write(f'Train Data - Mean Squared Error (MSE): {mse_train}')
        st.write(f'Train Data - Root Mean Squared Error (RMSE): {rmse_train}')
        st.write(f'Train Data - R-squared (R2): {r2_train}')

        # Plotting
        fig_test, ax_test = plt.subplots()
        ax_test.scatter(y_test, y_pred_test, alpha=0.5, label='Predicted vs Actual (Test)')
        ax_test.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit (y=x)')
        ax_test.set_xlabel('Actual Consumption (MW)')
        ax_test.set_ylabel('Predicted Consumption (MW)')
        ax_test.set_title('Actual vs Predicted Consumption (Test Data)')
        ax_test.legend()
        st.pyplot(fig_test)

        fig_train, ax_train = plt.subplots()
        ax_train.scatter(y_train, y_pred_train, alpha=0.5, label='Predicted vs Actual (Train)')
        ax_train.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Ideal Fit (y=x)')
        ax_train.set_xlabel('Actual Consumption (MW)')
        ax_train.set_ylabel('Predicted Consumption (MW)')
        ax_train.set_title('Actual vs Predicted Consumption (Train Data)')
        ax_train.legend()
        st.pyplot(fig_train)


        feature_importances = rf_regressor.feature_importances_
        features = X_train.columns

        # Create a DataFrame for better visualization
        feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

        # Sort the DataFrame by importance
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

        # Display feature importances
        st.subheader('Feature Importances')
        st.write(feature_importances_df)

        # Plot feature importances
        fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
        ax_importance.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
        ax_importance.set_xlabel('Importance')
        ax_importance.set_ylabel('Feature')
        ax_importance.set_title('Feature Importances')
        ax_importance.invert_yaxis()  # Invert y axis to have the most important feature on top
        st.pyplot(fig_importance)


def Prediction():
    def load_model(merged_file_path):
        with open(merged_file_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def get_features(bioenergies_mw, ech_physiques_mw, thermal_mw, hydraulic_mw, tavg_c, nuclear_mw, pumping_mw, solar_mw, wind_mw, tmin_c, tmax_c):
        features = np.array([bioenergies_mw, ech_physiques_mw, thermal_mw, hydraulic_mw, tavg_c, nuclear_mw, pumping_mw, solar_mw, wind_mw, tmin_c, tmax_c])
        return features.reshape(1, -1)

    def predict_surface_temperature(features):
        model = load_model()
        prediction = model.predict(features)
        return np.round(prediction, 3)

    st.header("Prediction")
    st.subheader('Prediction Simulation with Random Forest Regressor')

    # Load the DataFrame
    data = pd.read_csv(merged_file_path, sep=',')
    df3 = data.copy()

    # Get the minimum and maximum values for each feature from the DataFrame
    bioenergies_mw_min, bioenergies_mw_max = df3['bioenergies_(mw)'].min(), df3['bioenergies_(mw)'].max()
    ech_physiques_mw_min, ech_physiques_mw_max = df3['ech._physiques_(mw)'].min(), df3['ech._physiques_(mw)'].max()
    thermal_mw_min, thermal_mw_max = df3['thermal_(mw)'].min(), df3['thermal_(mw)'].max()
    hydraulic_mw_min, hydraulic_mw_max = df3['hydraulic_(mw)'].min(), df3['hydraulic_(mw)'].max()
    tavg_c_min, tavg_c_max = df3['tavg_(°c)'].min(), df3['tavg_(°c)'].max()
    nuclear_mw_min, nuclear_mw_max = df3['nuclear_(mw)'].min(), df3['nuclear_(mw)'].max()
    pumping_mw_min, pumping_mw_max = df3['pumping_(mw)'].min(), df3['pumping_(mw)'].max()
    solar_mw_min, solar_mw_max = df3['solar_(mw)'].min(), df3['solar_(mw)'].max()
    wind_mw_min, wind_mw_max = df3['wind_(mw)'].min(), df3['wind_(mw)'].max()
    tmin_c_min, tmin_c_max = df3['tmin_(°c)'].min(), df3['tmin_(°c)'].max()
    tmax_c_min, tmax_c_max = df3['tmax_(°c)'].min(), df3['tmax_(°c)'].max()

    # Feature inputs
    col1, col2 = st.columns(2)

    with col1:
        bioenergies_mw = st.slider("Bioenergies (MW)", min_value=float(bioenergies_mw_min), max_value=float(bioenergies_mw_max), value=bioenergies_mw_min)
        ech_physiques_mw = st.slider("Ech. Physiques (MW)", min_value=float(ech_physiques_mw_min), max_value=float(ech_physiques_mw_max), value=ech_physiques_mw_min)
        thermal_mw = st.slider("Thermal (MW)", min_value=float(thermal_mw_min), max_value=float(thermal_mw_max), value=thermal_mw_min)
        hydraulic_mw = st.slider("Hydraulic (MW)", min_value=float(hydraulic_mw_min), max_value=float(hydraulic_mw_max), value=hydraulic_mw_min)
        tavg_c = st.slider("Avg Temperature (°C)", min_value=float(tavg_c_min), max_value=float(tavg_c_max), value=tavg_c_min)
        nuclear_mw = st.slider("Nuclear (MW)", min_value=float(nuclear_mw_min), max_value=float(nuclear_mw_max), value=nuclear_mw_min)

    with col2:
        pumping_mw = st.slider("Pumping (MW)", min_value=float(pumping_mw_min), max_value=float(pumping_mw_max), value=pumping_mw_min)
        solar_mw = st.slider("Solar (MW)", min_value=float(solar_mw_min), max_value=float(solar_mw_max), value=solar_mw_min)
        wind_mw = st.slider("Wind (MW)", min_value=float(wind_mw_min), max_value=float(wind_mw_max), value=wind_mw_min)
        tmin_c = st.slider("Min Temperature (°C)", min_value=float(tmin_c_min), max_value=float(tmin_c_max), value=tmin_c_min)
        tmax_c = st.slider("Max Temperature (°C)", min_value=float(tmax_c_min), max_value=float(tmax_c_max), value=tmax_c_min)

    # Add a button for prediction
    if st.button("Predict"):
        features = get_features(bioenergies_mw, ech_physiques_mw, thermal_mw, hydraulic_mw, tavg_c, nuclear_mw, pumping_mw, solar_mw, wind_mw, tmin_c, tmax_c)
        prediction = predict_surface_temperature(features)
        st.write(f"The prediction of the energy consumption is: {prediction[0]}")
# Sidebar

st.markdown(
    """
    <style>
    .css-1d391kg {  /* This is the class for the main sidebar */
        background-color: rgba(255, 255, 255, 0.1) !important;  /* Adjust the transparency here */
    }
    .css-1v3fvcr {  /* This is the class for the main block within the sidebar */
        padding: 10px;
    }
    .css-1d391kg h2 { /* Adjust the font size for the title */
        font-size: 20px;
        font-weight: bold;
    }
    .css-1d391kg img {
        max-width: 100px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#st.sidebar.image("/Users/tarunayikkara/Documents/Data Scientest/Energy Project/Streamlit/energy_autodraw.png")
#st.sidebar.markdown('<div class="sidebar-title">Energy and Temperature Analysis</div>', unsafe_allow_html=True)
st.sidebar.title("Energy and Temperature Analysis")


# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Visualization", "Modelling", "Prediction", "Conclusion"])
st.sidebar.markdown(
    """
    - **Course** : Data Analyst
    - **Format** : Bootcamp
    - **Month** : May 2024
    - **Group** :
        - Martina RIETING
        - Tarun AYIKKARA
        - Johannes HEROLD
    """
)


# Introduction Section
if selection == "Introduction":
    st.title("Energy Consumption and Production Analysis in France")
    st.header("Project Description")
    st.write("""
    The primary goal of this project is to analyze the phasing between energy consumption and production at both the national and departmental levels in France. 
    The project aims to understand the correlation between energy consumption and temperature, evaluate different energy sources, and determine the total consumption by region. 
    Furthermore, the project will employ a machine learning model to predict energy consumption based on various features such as energy sources, temperature, and more.
    """)
    st.header("Data Source")
    st.write("""
    The data for this project is sourced from the ODRE (Open Data Energy Networks), which provides comprehensive information on energy consumption and production. 
    This dataset includes detailed records by sector, updated every half hour since 2013.
    """)
    st.header("Project Phases")
    st.write("""
    **Data Collection and Preprocessing:**
    - Retrieve energy consumption and production data from the ODRE database.
    - Clean and preprocess the data to ensure it is suitable for analysis, including handling missing values and normalizing the data.
    - Extract relevant features such as energy sources, temperature, and geographical regions.

    **Exploratory Data Analysis (EDA):**
    - Perform an in-depth analysis of the data to understand consumption and production patterns.
    - Visualize the phasing between energy consumption and production at both national and departmental levels.
    - Analyze the relationship between energy consumption and temperature.

    **Regional Analysis:**
    - Break down the energy consumption data by region to identify trends and anomalies.
    - Compare the energy consumption across different regions and understand the impact of local factors such as climate and industrial activities.

    **Machine Learning Model Development:**
    - Select appropriate machine learning algorithms to predict energy consumption.
    - Train the model using historical data, incorporating features such as energy sources, temperature, and regional information.
    - Validate the model to ensure its accuracy and reliability.

    **Prediction and Forecasting:**
    - Use the trained model to predict future energy consumption patterns.
    - Assess the model’s performance and refine it as needed to improve prediction accuracy.

    **Reporting and Visualization:**
    - Develop comprehensive reports and visualizations to communicate findings effectively.
    - Create interactive dashboards to allow stakeholders to explore the data and predictions.

    **Expected Outcomes:**
    - A detailed understanding of the phasing between energy consumption and production in France.
    - Insights into the impact of temperature on energy consumption.
    - Identification of regional consumption patterns and their driving factors.
    - A reliable machine learning model capable of predicting energy consumption based on multiple features.
    - Tools and visualizations to support decision-making and policy formulation in the energy sector.
    """)

# Data Exploration Section
elif selection == "Data Exploration":
    st.header("Data Exploration")
    
    # Display Data Summary
    st.subheader("Data Summary")
    st.write("### Original Energy Consumption Data")
    st.write(energy.head())

    st.write("### Revised Energy Data")
    st.write(energy_revised.head())

    # Display Basic Statistics
    st.subheader("Basic Statistics")
    st.write("### Original Energy Consumption Data")
    st.write(energy.describe())

    st.write("### Revised Energy Data")
    st.write(energy_revised.describe())

    # Display temperature data
    st.write("### Temperature Data")
    st.write(temp.head())

    #Merged data
    st.write('### Merged data(Energy and Temperature)')
    st.write(merged_df.head())

    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA) - Energy Revised")

    st.write("### Data Information")
    buffer = io.StringIO()
    energy_revised.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA) - Temperature")

    st.write("### Data Information")
    buffer = io.StringIO()
    temp.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA) - Merged data ( Final Data set)")

    st.write("### Data Information")
    buffer = io.StringIO()
    merged_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("### Describe Numerical Columns of the merged data set")
    st.write(merged_df.describe())

    st.write("### Describe Categorical Columns of the merged data set")
    st.write(merged_df.describe(include='object'))

    st.write("### Histograms of Numerical Columns")
    for col in energy_revised.select_dtypes(include='number').columns:
        st.write(f"Histogram of {col}")
        fig, ax = plt.subplots()
        sns.histplot(energy_revised[col], ax=ax)
        st.pyplot(fig)

    st.write("### Boxplots of Numerical Columns")
    for col in energy_revised.select_dtypes(include='number').columns:
        st.write(f"Boxplot of {col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=energy_revised[col], ax=ax)
        st.pyplot(fig)

# Visualization Section
elif selection == "Visualization":
    st.header("Visualization")


    st.subheader("Histograms for Consumption by Region Codes")

    rc = energy_revised.sort_values(by='insee_region_code').groupby('region')

# Initialize an empty list to store dropdown options
    dropdown_options = []

# Create a dictionary to store all figures
    figures = {}

    # Create histograms for each region using Plotly and store in figures dictionary
    for name, group in rc:
        fig = px.histogram(group, x='consumption_(mw)', title=f'Histogram for insee_region_code: {name}',
                        labels={'consumption_(mw)': 'Consumption (MW)', 'count': 'Frequency'})
        figures[name] = fig
        dropdown_options.append(name)

# Create a dropdown menu in Streamlit
    selected_region = st.selectbox('Select Region', options=dropdown_options)

    # Display the selected histogram using Plotly
    if selected_region:
        st.plotly_chart(figures[selected_region])


    st.subheader("Energy Consumption Over Year")
    # Create a Seaborn line plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='New_year', y='consumption_(mw)', data=energy_revised)
    plt.title('Energy Consumption Over Year')
    # Display the plot in Streamlit
    st.pyplot(plt)

    st.subheader("Energy Consumption Over Month")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='New_month', y='consumption_(mw)', data=energy_revised)
    plt.title('Energy Consumption Over Month')
    st.pyplot(plt)


    st.subheader("Bar Plots for Top and Bottom 5 Regions")
    region_consumption = energy_revised.groupby('region')['consumption_(mw)'].sum()
    top_5_regions = region_consumption.nlargest(5)
    flop_5_regions = region_consumption.nsmallest(5)
    fig_top = px.bar(top_5_regions, x=top_5_regions.index, y='consumption_(mw)', title='Top 5 Regions by Energy Usage')
    fig_top.update_traces(marker_color='green')
    fig_flop = px.bar(flop_5_regions, x=flop_5_regions.index, y='consumption_(mw)', title='Flop 5 Regions by Energy Usage')
    fig_flop.update_traces(marker_color='red')
    st.plotly_chart(fig_top)
    st.plotly_chart(fig_flop)


    st.subheader("Density Plot for Energy Consumption")
    fig = px.density_heatmap(energy_revised, x='consumption_(mw)', y='region', title='Density of Energy Consumption')
    st.plotly_chart(fig)

   
    # Streamlit app
    st.subheader("Inverse Relationship Between Temperature and Consumption Over Time")

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot temperature
    sns.lineplot(data=merged_df, x='New_month', y='tavg_(°c)', marker='o', ax=ax1, color='red', label='Temperature')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Month')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.tick_params(axis='x', rotation=45)

    # Create a twin Axes sharing the xaxis
    ax2 = ax1.twinx()

    # Plot consumption
    sns.lineplot(data=merged_df, x='New_month', y='consumption_(mw)', marker='x', ax=ax2, linestyle='--', color='green', label='Consumption')
    ax2.set_ylabel('Consumption (MW)')


    # Display plot in Streamlit
    st.pyplot(fig)



    


# Modelling Section
elif selection == "Modelling":
    st.header("Data Model")
    st.header('Linear Regression')
    st.write('Performance Summary: Consistent performance with high R² values for both training (0.914) and test (0.916) sets.')
    st.write('Error Analysis: Moderate prediction errors with MAE around 470 MW and RMSE around 604 MW, indicating some large prediction errors.')
    st.write('Fit and Generalization: Good fit and generalization, but higher errors compared to some other models.')
    st.write('Overall Interpretation: Linear Regression captures the overall trend well but may not be the most accurate model in terms of minimizing prediction errors.')

    st.header('Random Forest Regressor')
    st.write('Performance Summary: Best overall performance with the highest R² scores (Train: 0.959, Test: 0.956) and lowest errors.')
    st.write('Error Analysis: Low errors with MAE around 291 MW for training and 305 MW for testing, and RMSE around 428 MW for training and 447 MW for testing.')
    st.write('Fit and Generalization: Excellent predictive power and generalization ability.')
    st.write('Overall Interpretation: Handles non-linear relationships and interactions between features well, making it the most accurate model for this dataset.')

    st.header('Decision Tree Regressor')
    st.write('Performance Summary: High R² scores (Train: 0.959, Test: 0.954), similar to Random Forest but slightly lower on the test set.')
    st.write('Error Analysis: Low errors with MAE around 297 MW for training and 316 MW for testing, and RMSE around 431 MW for training and 456 MW for testing.')
    st.write('Fit and Generalization: Very good predictive power, but slightly overfitted compared to Random Forest.')
    st.write('Overall Interpretation: Performs well with non-linear relationships but shows slight signs of overfitting.')

    st.header('Lasso')
    st.write('Performance Summary: Consistent R² values (Train: 0.916, Test: 0.917) indicating a good fit and generalization.')
    st.write('Error Analysis: Moderate prediction errors with MAE around 472 MW and RMSE around 617 MW.')
    st.write('Fit and Generalization: Similar performance to Linear Regression with slight regularization.')
    st.write('Overall Interpretation: Provides no significant advantage over Linear Regression in this scenario, indicating moderate prediction accuracy.')

    st.header('Lasso CV')
    st.write('Performance Summary: R² values (Train: 0.914, Test: 0.915) similar to Linear Regression and Lasso.')
    st.write('Error Analysis: Moderate prediction errors with MAE around 471 MW and RMSE around 606 MW.')
    st.write('Fit and Generalization: Cross-validation for regularization does not significantly improve performance over standard Lasso.')
    st.write('Overall Interpretation: Similar to Lasso, with no significant improvement over Linear Regression.')

    st.header('Ridge')
    st.write('Performance Summary: Slightly lower R² values (Train: 0.911, Test: 0.913) compared to other linear models.')
    st.write('Error Analysis: Higher prediction errors with MAE around 485 MW and RMSE around 633 MW.')
    st.write('Fit and Generalization: Performs the worst among the linear models in terms of error metrics.')
    st.write('Overall Interpretation: Provides the least accurate predictions among the linear models, indicating higher errors and slightly lower predictive power.')
    
    Modelling()

       
# Prediction Section
elif selection == "Prediction":
    Prediction()

# Add a button for prediction
#if st.button("Predict"):
    #features = get_features(bioenergies_mw, ech_physiques_mw, thermal_mw, hydraulic_mw, tavg_c, nuclear_mw, insee_region_code_x, region_x_Normandie, pumping_mw, solar_mw, wind_mw, region_x_Grand_Est, tmin_c, tmax_c, region_x_Provence_Alpes_Cote_d_Azur)
    #prediction = predict_surface_temperature(features)
    #st.write(f"The prediction of the energy consumption is: {prediction[0]}")

# Conclusion Section
elif selection == "Conclusion":

    # Title
    st.title("Conclusion and Recommendations")

    # Conclusion
    st.header("Conclusion")
    st.write("""
    In conclusion, the Random Forest Regressor emerges as the top-performing model for predicting energy consumption based on the given metrics. However, further improvements can be achieved by enhancing feature engineering, refining model tuning, and incorporating additional data sources such as weather information. By following these recommendations, the project can advance towards developing a robust predictive model that accurately forecasts energy consumption and supports informed decision-making in energy management and policy.
    """)

    # Recommendations for Improving Model Performance
    st.header("Recommendations for Improving Model Performance")

    # Feature Engineering
    st.subheader("1. Feature Engineering")
    st.markdown("""
    - **Temporal Features:** Incorporate additional time-related features such as hour of the day, day of the week, seasonality indicators (e.g., month, quarter).
    - **Weather Data:** Include weather-related variables such as temperature, humidity, precipitation, wind speed, which can significantly impact energy consumption.
    """)

    # Model Tuning
    st.subheader("2. Model Tuning")
    st.markdown("""
    - Perform hyperparameter tuning for models such as Random Forest and Decision Tree to optimize performance further.
    - Use techniques like cross-validation to select the best parameters and avoid overfitting.
    """)

    # Ensemble Methods
    st.subheader("3. Ensemble Methods")
    st.markdown("""
    - Explore ensemble methods like Gradient Boosting Machines (GBM), AdaBoost, or stacking to combine the strengths of different models and improve overall performance.
    """)

    # Data Quality and Preprocessing
    st.subheader("4. Data Quality and Preprocessing")
    st.markdown("""
    - Ensure data cleanliness and handle missing values appropriately (e.g., imputation with median, mean values).
    - Scale numerical features if necessary (e.g., using StandardScaler) to ensure models perform well with variables of different scales.
    """)

    # Regularization Techniques
    st.subheader("5. Regularization Techniques")
    st.markdown("""
    - Experiment with different regularization strengths in Lasso and Ridge Regression to find a balance between bias and variance.
    - Consider ElasticNet regularization to combine L1 (Lasso) and L2 (Ridge) penalties for better performance.
    """)

    # Cross-Validation
    st.subheader("6. Cross-Validation")
    st.markdown("""
    - Implement robust cross-validation strategies to evaluate model performance thoroughly and ensure it generalizes well to unseen data.
    """)

    # Domain Knowledge Integration
    st.subheader("7. Domain Knowledge Integration")
    st.markdown("""
    - Collaborate with domain experts to incorporate domain-specific knowledge and insights into feature selection and model interpretation.
    - Understand the impact of policy changes, economic factors, or technological advancements on energy consumption trends.
    """)
