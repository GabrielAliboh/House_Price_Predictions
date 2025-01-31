import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.markdown("""#  House Price Prediction in Nigeria Using Supervised Machine Learning 
            
This datasets contains Houses listings in Nigeria and their prices based on Location and other parameters.

Datashape (24326, 8)

Parameters:

- bedrooms -> number of bedrooms in the houses
- bathrooms -> number of bathrooms in the houses
- toilets -> number of toilets 
- parking_space
- title -> house type
- town -> town in which the house is located
- state -> state within Nigeria in which the house is located and finally
- price -> the target column.


""")


# Importing necessary python libraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



# Loading in the Dataset
pd.read_csv('nigeria_houses_data.csv')


houses = pd.read_csv('nigeria_houses_data.csv')



### Exploring / Inspecting Data


list_state = houses.state.unique()

for state in list_state:
    town_list = houses[houses["state"]==state].groupby("town")["price"].mean().index.to_list()
    print(f"{state}: {town_list}\n")

st.markdown("""#### Brief overview of the dataset""")

st.write(houses.head(4))


houses.info()

st.markdown("""##### The summary statistics shows that:

The maximum bedrooms, bathrooms, parking_space and toilets are 9.
There are 25 states in the dataset.
There are 7 unique title in the dataset.
There are 189 town in the dataset.
The highest apartment price is 1,800,000,000,000 naira.
The lowest apartment price is 90,000 """)
st.markdown("""###### The types of houses in the Data used""")

st.write(houses['title'].unique())
# List of each unique item in the column 'title'

st.markdown("""### List of each unique item in the column 'state'""")

st.write(pd.Series(houses['state'].unique()))

houses.isnull().sum()


houses[houses.duplicated()]



houses.drop_duplicates(inplace=True)
# Dropping All Duplicate Values

houses.shape


st.markdown("""There were 10558 duplicated rows in the dataset
Shape with duplicates: (24326, 8)
Shape without duplicates: (13768, 8)""")

# apartments with less than 4 bedrooms with 9 toilets
houses[(houses['toilets'] == 9) & (houses["bedrooms"] < 4) ].head()
houses.drop(houses[(houses['toilets'] == 9) & (houses["bedrooms"] < 4)].index.to_list(), axis=0, inplace=True)

# apartments with less than 4 bedrooms with 8 toilets
houses[(houses['toilets'] == 8) & (houses["bedrooms"] < 3) ].head()
# dropping these extreme rows [ 2525,  4169,  4176, 13626, 14047, 14263, 14675, 15333, 15965, 16971, 18217, 20737, 20947, 21264]
houses.drop(houses[(houses['toilets'] == 8) & (houses["bedrooms"] < 3) ].index.to_list(), axis=0, inplace=True)

# apartments with less than 4 bedrooms with 8 toilets
houses[(houses['toilets'] == 7) & (houses["bedrooms"] < 3) ].head()
# dropping this extreme row 14917
houses.drop(houses[(houses['toilets'] ==7) & (houses["bedrooms"] < 3) ].index.to_list(), axis=0, inplace=True)

# apartments with less than 4 bedrooms with 8 toilets
houses[(houses['toilets'] == 6) & (houses["bedrooms"] < 2)].head()
# dropping this extreme row 7287
houses.drop(houses[(houses['toilets'] == 6) & (houses["bedrooms"] < 2) ].index.to_list(), axis=0, inplace=True)

# apartments with 9 bedrooms having less than 4 toilets
houses[(houses['bedrooms'] == 9) & (houses["toilets"] < 4)].head()
# dropping these extreme rows
houses.drop(houses[(houses['bedrooms'] == 9) & (houses["toilets"] < 4)].index.to_list(), axis=0, inplace=True)

# apartments with 8 bedrooms having less than 3 toilets
houses[(houses['bedrooms'] == 8) & (houses["toilets"] <3)].head()
# dropping these extreme rows
houses.drop(houses[(houses['bedrooms'] == 8) & (houses["toilets"] < 4)].index.to_list(), axis=0, inplace=True)

# apartments with 7 bedrooms having less than 3 toilets
houses[(houses['bedrooms'] == 7) & (houses["toilets"] <3)].head()
# dropping these extreme rows
houses.drop(houses[(houses['bedrooms'] == 7) & (houses["toilets"] < 3)].index.to_list(), axis=0, inplace=True)

houses[(houses['bathrooms'] == 8) & (houses["bedrooms"] < 3)].head()
# dropping this extreme row 9786
houses.drop(houses[(houses['bathrooms'] == 8) & (houses["bedrooms"] < 3)].index.to_list(), axis=0, inplace=True)

# apartments with 9 bedrooms having less than 4 bathrooms
houses[(houses['bedrooms'] == 9) & (houses["bathrooms"] < 4)].head()
# dropping these extreme rows
houses.drop(houses[(houses['bedrooms'] == 9) & (houses["bathrooms"] < 4)].index.to_list(), axis=0, inplace=True)

# apartments with 8 bedrooms having less than 4 bathrooms
houses[(houses['bedrooms'] == 8) & (houses["bathrooms"] < 4)].head()
# dropping this extreme row
houses.drop(houses[(houses['bedrooms'] == 8) & (houses["bathrooms"] < 4)].index.to_list(), axis=0, inplace=True)

# apartments with 7 bedrooms having less than 3 bathrooms
houses[(houses['bedrooms'] == 7) & (houses["bathrooms"] < 3)].head()
# dropping this extreme row
houses.drop(houses[(houses['bedrooms'] == 7) & (houses["bathrooms"] < 3)].index.to_list(), axis=0, inplace=True)

# apartments with 6 bedrooms having less than 3 bathrooms
houses[(houses['bedrooms'] == 6) & (houses["bathrooms"] < 3)].head()
# dropping these extreme rows
houses.drop(houses[(houses['bedrooms'] == 6) & (houses["bathrooms"] < 3)].index.to_list(), axis=0, inplace=True)

# apartments where bathrooms > 6 and toilets < 3 or bathrooms < 3 and toilets > 6
houses[((houses['bathrooms'] > 6) & (houses["toilets"] < 3)) | ((houses['bathrooms'] < 3) & (houses["toilets"] > 6))].head()
# dropping these extreme rows
houses.drop(houses[((houses['bathrooms'] > 6) & (houses["toilets"] < 3)) | ((houses['bathrooms'] < 3) & (houses["toilets"] > 6))].index.to_list(),
        inplace=True, axis=0)


records = houses['state'].value_counts()
st.write(records)


# remove the states with few entries
houses1 = houses[~houses['state'].isin(records[records < 10].index)]

houses1['state'].value_counts()



houses1.info()

sns.boxplot(x=houses1["price"]/1e9, orient="h")
plt.xlabel("Price (in billion naira)")
plt.title("Boxplot of Price of Apartment in Nigeria");



# Removing outliers in price (prices below or above the 10th and 90th percentile respectively)

low, high = houses1["price"].quantile([0.1,0.90])
mask = houses1["price"].between(low, high)
houses1 = houses1[mask]

houses1.shape


fig, ax = plt.subplots()
sns.boxplot(x=houses1["price"] / 1e6, orient="h", color="lightblue", ax=ax)
ax.set_xlabel("Price (in million naira)")
ax.set_title("Boxplot of Price of Apartment in Nigeria")

# Display in Streamlit
st.pyplot(fig)



# Second Visualization: Bar Chart
st.subheader("Average Price by State")
fig2, ax2 = plt.subplots()
houses1.groupby('state')['price'].mean().sort_values(ascending=False).plot(kind='bar', ax=ax2)
ax2.set_ylabel("Average Price (in naira)")
ax2.set_xlabel("State")
ax2.set_title("Average Apartment Prices by State in Nigeria")
st.pyplot(fig2)

st.markdown("""###### Descriptive statistics of houses by State """)
# Aggregation on state 
house_agg = houses1.groupby("state")["price"].agg(["count", "median", "mean", "std"]).sort_values(ascending=False, by="mean")

# Aggregation with state with more than 10 records
house_agg = house_agg[house_agg["count"] > 10]

st.write(house_agg)

# Third Visualization: Barplot with Mean and Median Prices
st.subheader("Mean and Median Prices by State")
# Aggregate data
house_agg = houses1.groupby("state")["price"].agg(["mean", "median"])
bar_data = pd.melt(house_agg.reset_index().iloc[:, [0, 1, 2]], id_vars="state")

fig3, ax3 = plt.subplots()
sns.barplot(
    x=bar_data["state"],
    y=bar_data["value"] / 1e6,
    hue=bar_data["variable"],
    palette=sns.color_palette("muted", 2),
    ax=ax3
)
ax3.set_xlabel("State")
ax3.set_ylabel("Price (in million naira)")
ax3.set_title("Average House Prices in Nigeria By State")
ax3.tick_params(axis='x', rotation=90)
st.pyplot(fig3)



st.subheader("Most Expensive Towns in Nigeria")
fig4, ax4 = plt.subplots()
houses1.groupby('town')['price'].mean().sort_values(ascending=False).head(20).plot(
    kind='bar', title="Most Expensive Towns In Nigeria", ax=ax4
)
ax4.set_ylabel("Average Price (in naira)")
ax4.set_xlabel("Town")
st.pyplot(fig4)

# Fifth Visualization: Barplot of House Title Distribution
st.subheader("Distribution of House Titles")
fig5, ax5 = plt.subplots()
houses1["title"].value_counts(normalize=True).sort_values().plot.barh(
    xlabel="Frequency (count)", color="lightblue", ylabel="Title", title="Barplot of House Title", ax=ax5
)
st.pyplot(fig5)

# Sixth Visualization: Barplot of House Title and Mean Price
st.subheader("House Title vs. Average Price")
fig6, ax6 = plt.subplots()
(houses1.groupby("title")["price"].mean() / 1e6).sort_values().plot.barh(
    xlabel="Mean Price (in million naira)", ylabel="Title", color="lightblue",
    title="Average House Price and House Title", ax=ax6
)
st.pyplot(fig6)

# Seventh Visualization: Correlation Between Space and House Price
st.subheader("Correlation Between Space and House Price")
correlation_matrix = houses1.select_dtypes("number").corr()
fig7, ax7 = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax7)
st.pyplot(fig7)

def categorize_as_other(state, state_houses1):
    # get the counts of each town

    state_count = state_houses1["town"].value_counts()
    
    # get towns less than 20
    less_than_20_town = state_count[state_count < 20].index.to_list()
    dict_less_than_20_town = {town: f"Others_{state.lower()}" for town in less_than_20_town}
    
    # replacing each town with Others
    state_houses1.loc[:,"town"] = state_houses1["town"].replace(dict_less_than_20_town)
    
    return state_houses1


# Extract Abuja records
abuja_houses1 = houses1[houses1["state"] == "Abuja"].drop(columns="state")
print(houses1.shape)
abuja_houses1.head()


#Categorizing towns with less than 20 records as Others
abuja_houses1 = categorize_as_other("abuja", abuja_houses1)
abuja_houses1["town"].unique()


# Splitting The Dataset Into Training & Testing
X_abj, y_abj = abuja_houses1.drop(columns="price"), abuja_houses1["price"]

# Importing necessary Machine Learning Libraries
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from ipywidgets import Dropdown, IntSlider, interact
import warnings
#from sklearn.model_selection import train_test_split


X_train_abj, X_test_abj, y_train_abj, y_test_abj = train_test_split(X_abj, y_abj, test_size=0.2, random_state=42)
X_abj2 = X_abj.copy()


encoder = OrdinalEncoder()

# Encode the categorical features (title and town)
X_abj[['title', 'town']] = encoder.fit_transform(X_abj[['title', 'town']])

# Split data into training and testing sets
X_train_abj, X_test_abj, y_train_abj, y_test_abj = train_test_split(X_abj, y_abj, test_size=0.2, random_state=42)

# Initialize the GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=42, alpha=1e-10, max_depth=4, n_estimators=200)

# Fit the model
model.fit(X_train_abj, y_train_abj)

# Predictions
y_pred_train = model.predict(X_train_abj)
y_pred_test = model.predict(X_test_abj)

# Evaluate the model
print(f"R2 Score (training): {r2_score(y_train_abj, y_pred_train):,.2f}")
print(f"MAE (training): N{mean_absolute_error(y_train_abj, y_pred_train):,.2f}")
print(f"MAE (test): N{mean_absolute_error(y_test_abj, y_pred_test):,.2f}")



st.write(X_abj)
st.title("House Price Prediction")
st.write(X_abj.bedrooms.max())
# Assuming 'title' is one of the columns in X_abj
title_column = X_abj2['title']  # Replace 'title' with the actual column name if different
town_column = X_abj2['town']
# Now using st.radio() with the unique values from that column

# Create a form for input
with st.form(key='prediction_form'):
   bedrooms = st.number_input("Number of Bedrooms", min_value=X_abj.bedrooms.min(), max_value=X_abj.bedrooms.max(), value=1.0)
   bathrooms = st.number_input("Number of Bathrooms", min_value=X_abj.bathrooms.min(), max_value=X_abj.bathrooms.max(), value=1.0)
   toilets = st.number_input("Number of toilet", min_value=X_abj.toilets.min(), max_value=X_abj.toilets.max(), value=1.0)
   parking_space = st.number_input("Number of parking space", min_value=X_abj.parking_space.min(), max_value=X_abj.parking_space.max(), value=1.0)
   title = st.radio('Choose the type of house', list(title_column.unique()))
   town = st.radio('Choose the type of house', list(town_column.unique()))
   submit_button = st.form_submit_button("Make Prediction")

title_encoder = OrdinalEncoder()
town_encoder = OrdinalEncoder()
title_encoded = title_encoder.fit_transform([[title]])[0][0]  # Transforming 'title'
town_encoded = town_encoder.fit_transform([[town]])[0][0]

# Handle prediction when the form is submitted
if submit_button:
       data = np.array([bedrooms,bathrooms,toilets,parking_space,title_encoded,town_encoded]).reshape(1,-1)
       predicted_price = model.predict(data)
       st.write(f"The house should be around N{predicted_price[0]:,.2f}")