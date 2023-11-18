# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
df = pd.read_csv('mumbai house price prediction\Mumbai1_with_risk.csv')

# Mapping for risk: 'Low' -> 0, 'High' -> 1
risk_mapping = {'Low': 0, 'High': 1}

# Apply the mapping to create a new 'Risk_numeric' column
df['Risk_numeric'] = df['Risk'].map(risk_mapping)

# Display the DataFrame with the new 'Risk_numeric' column
print(df[['Risk', 'Risk_numeric']])

# Data Preprocessing
# Handle missing values, encode categorical variables, etc.

# Define features
features = df[['Area', 'No. of Bedrooms', 'Gymnasium', 'Lift Available', 'Car Parking',
               'Maintenance Staff', '24x7 Security', 'Children\'s Play Area', 'Clubhouse',
               'Intercom', 'Landscaped Gardens', 'Indoor Games', 'Gas Connection',
               'Jogging Track', 'Swimming Pool']]

# Create a new 'Risk' column
df['Risk'] = df.apply(lambda row: 'High' if row['Gymnasium'] == 0 or row['Lift Available'] == 0 or
                                         row['Car Parking'] == 0 or row['24x7 Security'] == 0 else 'Low', axis=1)

# Save the DataFrame back to the CSV file with the new 'Risk' column
df.to_csv('mumbai house price prediction\Mumbai1_with_risk.csv', index=False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, df['Risk'], test_size=0.2, random_state=42)

# Build a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
risk_predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, risk_predictions)
conf_matrix = confusion_matrix(y_test, risk_predictions)
class_report = classification_report(y_test, risk_predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
df = pd.read_csv('mumbai house price prediction\Mumbai1_with_risk.csv')

# Data Preprocessing
# Handle missing values, encode categorical variables, etc.

# Define features and target variable
features = df[['Area', 'No. of Bedrooms', 'Gymnasium', 'Lift Available', 'Car Parking',
               'Maintenance Staff', '24x7 Security', 'Children\'s Play Area', 'Clubhouse',
               'Intercom', 'Landscaped Gardens', 'Indoor Games', 'Gas Connection',
               'Jogging Track', 'Swimming Pool']]

target = df['Risk']  # Assuming you have a 'Risk' column indicating high or low risk

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
df = pd.read_csv('mumbai house price prediction\Mumbai1_with_risk.csv')

# Data Preprocessing
# Handle missing values, encode categorical variables, etc.

# Define features and target variable
features = df[['Area', 'No. of Bedrooms', 'Gymnasium', 'Lift Available', 'Car Parking',
               'Maintenance Staff', '24x7 Security', 'Children\'s Play Area', 'Clubhouse',
               'Intercom', 'Landscaped Gardens', 'Indoor Games', 'Gas Connection',
               'Jogging Track', 'Swimming Pool']]

target = df['Risk']  # Assuming you have a 'Risk' column indicating high or low risk

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
risk_predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, risk_predictions)
conf_matrix = confusion_matrix(y_test, risk_predictions)
class_report = classification_report(y_test, risk_predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Create a scatter plot with 'Area' and 'Risk'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Area', y='Risk', data=df, hue='Risk', palette='viridis', s=100)
plt.title('Scatter Plot of Area and Risk')
plt.xlabel('Area')
plt.ylabel('Risk')
plt.show()

# Make predictions on the test set
risk_predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, risk_predictions)
conf_matrix = confusion_matrix(y_test, risk_predictions)
class_report = classification_report(y_test, risk_predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')



risk_mapping = {'Low': 0, 'High': 1}
df['Risk_numeric'] = df['Risk'].map(risk_mapping)

# Sort the DataFrame by 'Risk' in descending order
top_risk_entries = df.sort_values(by='Risk_numeric', ascending=False).head(10)

# Display the top 10 highest-risk entries
print(top_risk_entries)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame df with 'Location', 'Risk', 'Price', and 'Risk_numeric' columns
# If your 'Risk' column contains string values like 'High' and 'Low'
# Convert it to a numeric representation for sorting

risk_mapping = {'Low': 0, 'High': 1}
df['Risk_numeric'] = df['Risk'].map(risk_mapping)

# Sort the DataFrame by 'Risk' in descending order
top_risk_entries = df.sort_values(by='Risk_numeric', ascending=False).head(10)

# Create a line plot with 'Location' and 'Price', colored by 'Risk'
plt.figure(figsize=(12, 6))
sns.lineplot(x='Location', y='Price', data=top_risk_entries, hue='Risk', palette='Set1', marker='o')
plt.title('Line Plot of Top 10 Highest-Risk vs Location vs Price')
plt.xlabel('Location')
plt.ylabel('Price')
plt.legend(title='Risk', loc='upper right')
plt.show()



import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame df with 'Location', 'Price', 'No. of Bedrooms', and 'Gas Connection' columns
features = df[['Price', 'No. of Bedrooms', 'Location', 'Gas Connection']]

# Convert categorical variables to numerical using one-hot encoding
features = pd.get_dummies(features, columns=['Location', 'Gas Connection'], drop_first=True)

# Normalize the features
normalized_features = (features - features.mean()) / features.std()

# Determine the number of clusters (you can choose an optimal number based on your business requirements)
num_clusters = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(normalized_features)

from mpl_toolkits.mplot3d import Axes3D


from sklearn.preprocessing import LabelEncoder

# Label encode the 'Location' column
label_encoder = LabelEncoder()
df['Location_encoded'] = label_encoder.fit_transform(df['Location'])
# Assuming you have a DataFrame df with 'Risk' column
risk_mapping = {'Low': 0, 'High': 1}
df['Risk_numeric'] = df['Risk'].map(risk_mapping)


# Create the 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with 'Price', 'No. of Bedrooms', 'Location_encoded', colored by 'Cluster'
scatter = ax.scatter(df['Price'], df['No. of Bedrooms'], df['Location_encoded'], c=df['Cluster'], cmap='viridis', s=100)

# Set labels and title
ax.set_xlabel('Price')
ax.set_ylabel('No. of Bedrooms')
ax.set_zlabel('Location (Encoded)')
ax.set_title('3D Scatter Plot of Customer Segmentation')

# Add a legend
legend = ax.legend(*scatter.legend_elements(), title='Cluster', loc='upper right')
ax.add_artist(legend)

# Show the plot
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset (assuming you have a DataFrame named 'df' with relevant property features and prices)
# df = pd.read_csv('your_property_data.csv')

# Select features and target variable
features = df[['Area', 'No. of Bedrooms', 'Gymnasium', 'Lift Available', 'Car Parking',
               'Maintenance Staff', '24x7 Security', 'Children\'s Play Area', 'Clubhouse',
               'Intercom', 'Landscaped Gardens', 'Indoor Games', 'Gas Connection',
               'Jogging Track', 'Swimming Pool']]

target = df['Price']  # Assuming you have a 'Price' column indicating property prices

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
price_predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, price_predictions)
r2 = r2_score(y_test, price_predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize predicted vs actual prices
plt.scatter(y_test, price_predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Property Prices')
plt.show()

corr_matrix = features.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()









import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

# Assuming you have a DataFrame df with 'Location', 'Price', and 'Risk' columns
# and 'Risk' column contains string values like 'High' and 'Low'

# Create Risk_numeric column
risk_mapping = {'Low': 0, 'High': 1}
df['Risk_numeric'] = df['Risk'].map(risk_mapping)

# Label encode the 'Location' column
label_encoder = LabelEncoder()
df['Location_encoded'] = label_encoder.fit_transform(df['Location'])

# Filter low-risk places
low_risk_places = df[df['Risk'] == 'Low']

# Sort low-risk places by price in descending order and select the top 250
top_low_risk_places = low_risk_places.sort_values(by='Price', ascending=False).head(250)

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with 'Location_encoded', 'Price', and 'Risk_numeric'
scatter = ax.scatter(top_low_risk_places['Location_encoded'],
                     top_low_risk_places['Price'],
                     top_low_risk_places['Risk_numeric'],
                     c=top_low_risk_places['Price'], cmap='viridis', s=50, edgecolors='k')

# Set labels and title
ax.set_xlabel('Location (Encoded)')
ax.set_ylabel('Price')
ax.set_zlabel('Risk (Numeric)')
ax.set_title('Top 250 Low-Risk Places with Prices')

# Add color bar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Price')

# Show the plot
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have a DataFrame df with 'Location', 'Price', and 'Risk' columns
# and 'Risk' column contains string values like 'High' and 'Low'

# Create Risk_numeric column
risk_mapping = {'Low': 0, 'High': 1}
df['Risk_numeric'] = df['Risk'].map(risk_mapping)

# Filter low-risk places
low_risk_places = df[df['Risk'] == 'Low']

# Sort low-risk places by price in descending order and select the top 250
top_low_risk_places = low_risk_places.sort_values(by='Price', ascending=False).head(250)

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

