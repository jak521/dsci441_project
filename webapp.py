import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from io import BytesIO

# Step 1: Load the data
@st.cache
def load_data():
    df = pd.read_csv("../../pew_data/2023-24 RLS Public Use File Feb 19.csv") 
    df_exclude_refused = df.fillna(99)
    df_tree = df_exclude_refused[(df_exclude_refused['RELPER'] != 99) & (df_exclude_refused['HAPPY'] != 99) & (df_exclude_refused['CURREL'] != 900000) & (df_exclude_refused['INC_SDT1'] != 99) & (df_exclude_refused['FERTREC'] != 99) & (df_exclude_refused['INC_SDT1'] != 99) & (df_exclude_refused['USGEN'] != 99)]
    df_tree['CURREL_SEGMENTED'] = df_tree['CURREL'].apply(
        lambda x: 'Catholic' if x == 10000 else 
                'Protestant' if x == 1000 else 
                'Jewish' if x == 50000 else 
                'Religiously unaffiliated' if x == 100000 else 
                'Refused' if x == 99 | x == 900000 else
                'Other'  # Handling the cases where the religion doesn't fit the specified ones
    )
    df_tree = df_tree[df_tree['CURREL_SEGMENTED'] != 'Refused']
    df_tree['HAPPY_ONE_THREE'] = df_tree['HAPPY'].apply(lambda x: 1 if x == 1 else 0)
    df_tree['CURREL_SEGMENTED_ENC'] = df_tree['CURREL_SEGMENTED'].astype('category').cat.codes
    return df_tree


df_tree = load_data()

# Step 2: Feature Selection
X = df_tree[['RELPER', 'CURREL_SEGMENTED_ENC', 'USGEN', 'INC_SDT1', 'FERTREC']]
y = df_tree['HAPPY_ONE_THREE']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Step 4: Initialize and Train the Decision Tree Classifier
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# Step 5: Predict and Evaluate the Model
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Step 6: Displaying the Model Evaluation Metrics
st.title("Decision Tree Model for Happiness Prediction")
st.subheader("Model Evaluation")
st.write(f"### Accuracy: {accuracy * 100:.2f}%")
st.write(f"### Classification Report:\n{class_report}")

# Step 7: Decision Tree Visualization
st.subheader("Decision Tree Visualization")

# Create the plot and save it to a BytesIO object to display in Streamlit
fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
plot_tree(tree_model, 
          feature_names=X.columns, 
          class_names=['Not Happy', 'Very Happy'], 
          filled=True, 
          rounded=True)

# Save the figure to a buffer
buf = BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)

# Display the figure in Streamlit
st.image(buf)

# Optional: Save the image to a file (if needed)
# plt.savefig("decision_tree_highres.png", bbox_inches='tight')


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Define the toggle (Selectbox for plot selection)
plot_option = st.selectbox('Choose plot', ['RELPER vs HAPPY', 'USGEN vs RELPER'])

# Create the plots based on selection
if plot_option == 'RELPER vs HAPPY':
    # Create RELPER vs HAPPY proportion bar plot
    prop_plot = df_tree.groupby(['RELPER', 'HAPPY']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)  # Proportions
    prop_plot.plot(kind='bar', stacked=True, color=['#006400', '#99FF99', '#FF6347'])
    plt.title("Proportions of Happiness by RELPER")
    plt.ylabel("Proportion")
    plt.xticks(ticks=[0,1,2,3], labels=['Very Religious', 'Somewhat Religious', 'Not Too Religious', 'Not at All Religious'])
    plt.legend(title='Happiness', labels=['Very Happy', 'Pretty Happy', 'Not Too Happy'], bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

elif plot_option == 'USGEN vs RELPER':
    # Create USGEN vs RELPER proportion bar plot
    prop_plot = df_tree.groupby(['USGEN', 'RELPER']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)  # Proportions
    prop_plot.plot(kind='bar', stacked=True, color=['#006400', '#99FF99', '#FFB6C1', '#FF6347'])
    plt.title("Proportions of RELPER by USGEN")
    plt.ylabel("Proportion")
    plt.xticks(ticks=[0,1,2], labels=['Immigrant', 'Child of Immigrant(s)', 'Neither'])
    plt.legend(title='Religion', labels=['Very Religious', 'Somewhat Religious', 'Not Too Religious', 'Not at all Religious'], bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)
