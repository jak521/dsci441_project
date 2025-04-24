import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
    df_tree['HAPPY_INVERTED'] = df_tree['HAPPY'].replace({1: 3, 2: 2, 3: 1})  # Swap 1 and 3, keep 2 the same
    df_tree['SUCCESS'] = np.where(
        df_tree[['INC_SDT1', 'HAPPY_INVERTED', 'EDUCREC']].eq(99).any(axis=1),
        np.nan,  # Set to NaN if any value is 99
        df_tree['INC_SDT1'] + df_tree['HAPPY_INVERTED'] + df_tree['EDUCREC']
    )   

    df_tree = df_tree.dropna(subset=['SUCCESS'])
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
plot_option = st.selectbox('Choose plot', ['RELPER vs HAPPY', 'USGEN vs RELPER', 'RELPER vs SUCCESS', 'CURREL vs HAPPY', 'CURREL vs SUCCESS'])

# Create the plots based on selection
if plot_option == 'RELPER vs HAPPY':
    # Create RELPER vs HAPPY proportion bar plot
    prop_plot = df_tree.groupby(['RELPER', 'HAPPY']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)  # Proportions
    prop_plot.plot(kind='bar', stacked=True, color=['#006400', '#99FF99', '#FF6347'])
    plt.title("Proportions of Happiness by Extent of Religion")
    plt.ylabel("Proportion")
    plt.xticks(ticks=[0,1,2,3], labels=['Very Religious', 'Somewhat Religious', 'Not Too Religious', 'Not at All Religious'])
    plt.legend(title='Happiness', labels=['Very Happy', 'Pretty Happy', 'Not Too Happy'], bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

elif plot_option == 'USGEN vs RELPER':
    # Create USGEN vs RELPER proportion bar plot
    prop_plot = df_tree.groupby(['USGEN', 'RELPER']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)  # Proportions
    prop_plot.plot(kind='bar', stacked=True, color=['#006400', '#99FF99', '#FFB6C1', '#FF6347'])
    plt.title("Proportions of Extent of Religion by Immigration Status")
    plt.ylabel("Proportion")
    plt.xticks(ticks=[0,1,2], labels=['Immigrant', 'Child of Immigrant(s)', 'Neither'])
    plt.legend(title='Religion', labels=['Very Religious', 'Somewhat Religious', 'Not Too Religious', 'Not at all Religious'], bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)


elif plot_option == 'RELPER vs SUCCESS':
    # Create RELPER vs SUCCESS proportion bar plot
    prop_plot = df_tree.groupby(['RELPER', 'SUCCESS']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)  # Proportions
    num_bins = 13
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap (green -> red)
    colors = [cmap(i / num_bins) for i in range(num_bins)]
    prop_plot.plot(kind='bar', stacked=True, color=colors)
    plt.title("Proportions of Success by Extent of Religion")
    plt.ylabel("Proportion")
    plt.xticks(ticks=[0,1,2,3], labels=['Very Religious', 'Somewhat Religious', 'Not Too Religious', 'Not at All Religious'])
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)


elif plot_option == 'CURREL vs HAPPY':
    # Create RELPER vs SUCCESS proportion bar plot
    prop_plot = df_tree.groupby(['CURREL_SEGMENTED', 'HAPPY']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)  # Proportions
    prop_plot.plot(kind='bar', stacked=True, color=['#006400', '#99FF99', '#FF6347'])
    plt.title("Proportions of Happiness by Current Religion")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(title='Happiness', labels=['Very Happy', 'Pretty Happy', 'Not Too Happy'], bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

elif plot_option == 'CURREL vs SUCCESS':
    # Create RELPER vs SUCCESS proportion bar plot
    prop_plot = df_tree.groupby(['CURREL_SEGMENTED', 'SUCCESS']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)  # Proportions
    num_bins = 13
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap (green -> red)
    colors = [cmap(i / num_bins) for i in range(num_bins)]
    prop_plot.plot(kind='bar', stacked=True, color=colors)
    plt.title("Proportions of Success by Current Religion")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)


# Features to consider
features = ['RELPER', 'CURREL', 'FERTREC', 'INC_SDT1', 'SUCCESS', 'USGEN', 'PARTY', 'REGION', 'ATTNDPERRLS', 'RELIMP', 'GOD', 'HVN', 'PRAY', 'RTRT', 'CHILDREN', 'BIRTHDECADE', 'RACECMB', 'IDEO', 'INTFREQ', 'GENDER']

# Step 1: Let user pick a subset of features
selected_features_pool = st.multiselect("Select features to explore:", features, default=features)

# Generate all combinations
feature_combos = []
for r in range(1, len(selected_features_pool) + 1):
    feature_combos.extend(itertools.combinations(selected_features_pool, r))

combo_names = [', '.join(combo) for combo in feature_combos]

df_tree = df_tree.copy()
for feature in selected_features_pool:
    df_tree = df_tree[df_tree[feature] != 99]
# Also make sure to filter 'CURREL' != 900000 if still needed
df_tree = df_tree[df_tree['CURREL'] != 900000]
# also filter our output of HAPPY
df_tree = df_tree[df_tree['HAPPY'] != 99]

# -------- Balanced Logistic Regression --------
st.subheader('Logistic Regression (Balanced Weights)')
selected_combo_bal = st.selectbox('Choose Features (Balanced):', combo_names, key='balanced_combo')
selected_features_bal = selected_combo_bal.split(', ')

X_bal = df_tree[selected_features_bal]
y_bal = df_tree['HAPPY_ONE_THREE']

X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)
log_reg_bal = LogisticRegression(class_weight='balanced', random_state=9, max_iter=1000)
log_reg_bal.fit(X_train_bal, y_train_bal)

y_pred_bal = log_reg_bal.predict(X_test_bal)
accuracy_bal = accuracy_score(y_test_bal, y_pred_bal)
st.write(f'**Accuracy (Balanced):** {accuracy_bal * 100:.2f}%')

cm_bal = confusion_matrix(y_test_bal, y_pred_bal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Blues', xticklabels=['Unhappy', 'Happy'], yticklabels=['Unhappy', 'Happy'])
plt.title('Confusion Matrix (Balanced)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)

# -------- Unbalanced Logistic Regression --------
st.subheader('Logistic Regression (Unbalanced)')
selected_combo_unbal = st.selectbox('Choose Features (Unbalanced):', combo_names, key='unbalanced_combo')
selected_features_unbal = selected_combo_unbal.split(', ')

X_unbal = df_tree[selected_features_unbal]
y_unbal = df_tree['HAPPY_ONE_THREE']

X_train_unbal, X_test_unbal, y_train_unbal, y_test_unbal = train_test_split(X_unbal, y_unbal, test_size=0.3, random_state=42)
log_reg_unbal = LogisticRegression(random_state=9, max_iter=1000)
log_reg_unbal.fit(X_train_unbal, y_train_unbal)

y_pred_unbal = log_reg_unbal.predict(X_test_unbal)
accuracy_unbal = accuracy_score(y_test_unbal, y_pred_unbal)
st.write(f'**Accuracy (Unbalanced):** {accuracy_unbal * 100:.2f}%')

cm_unbal = confusion_matrix(y_test_unbal, y_pred_unbal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_unbal, annot=True, fmt='d', cmap='Oranges', xticklabels=['Unhappy', 'Happy'], yticklabels=['Unhappy', 'Happy'])
plt.title('Confusion Matrix (Unbalanced)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)

st.subheader("Decision Tree Plot Explorer")

# Tree combo selector
selected_combo_tree = st.selectbox("Choose Features for Decision Tree:", combo_names, key='tree_combo')
selected_features_tree = selected_combo_tree.split(', ')

# Define X and y
X_tree = df_tree[selected_features_tree]
y_tree = df_tree['HAPPY_ONE_THREE']

# Fit the decision tree
tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
tree_model.fit(X_tree, y_tree)

# Export to graphviz
dot_data = export_graphviz(
    tree_model,
    out_file=None,
    feature_names=selected_features_tree,
    class_names=['Unhappy', 'Happy'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Render graph in Streamlit
st.graphviz_chart(dot_data)