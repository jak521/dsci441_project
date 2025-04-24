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

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.header('Faith, Happiness, and Success:Exploring Socioeconomic and Religious Trends in the U.S.')

with st.expander("ℹ️ Click here to view/download the Excel legend"):
    st.write("This file defines the feature acronyms and maps numeric values to their qualitative meanings used in the plots below.")
    with open("legend.xlsx", "rb") as file:
        st.download_button(
            label="Download Legend Mapping File",
            data=file,
            file_name="legend_mapping.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


    # ----------- PLOTS: Y-axis is always HAPPY ------------
st.subheader("Proportion Plots: Feature vs. Happiness")

happy_features = ['RELPER', 'CURREL_SEGMENTED', 'FERTREC', 'INC_SDT1', 'SUCCESS', 'USGEN', 'PARTY', 'REGION', 'ATTNDPERRLS', 'RELIMP', 'GOD', 'HVN', 'PRAY', 'RTRT', 'CHILDREN', 'BIRTHDECADE', 'RACECMB', 'IDEO', 'INTFREQ', 'GENDER']  # features to compare with HAPPY
x_happy = st.selectbox("Select x-axis feature (vs. HAPPY):", happy_features, key="happy_plot")
df_tree = df_tree[df_tree[x_happy] != 99]
# Also make sure to filter 'CURREL' != 900000 if still needed
df_tree = df_tree[df_tree['CURREL'] != 900000]
# also filter our output of HAPPY
df_tree = df_tree[df_tree['HAPPY'] != 99]

if x_happy in df_tree.columns:
    prop_plot = df_tree.groupby([x_happy, 'HAPPY']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)

    # Choose color palette
    happy_colors = ['#006400', '#99FF99', '#FF6347']

    # Create plot
    prop_plot.plot(kind='bar', stacked=True, color=happy_colors)
    plt.title(f"Proportions of Happiness by {x_happy}")
    plt.ylabel("Proportion")
    if x_happy == 'RELPER':
        plt.xticks(ticks=[0,1,2,3], labels=['Very Religious', 'Somewhat Religious', 'Not Too Religious', 'Not at All Religious'])
    elif x_happy == 'USGEN':
        plt.xticks(ticks=[0,1,2], labels=['Immigrant', 'Child of Immigrant(s)', 'Neither'])
    else:
        plt.xticks(rotation=0)
    plt.legend(title='Happiness', labels=['Very Happy', 'Pretty Happy', 'Not Too Happy'], bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)


# ----------- PLOTS: Y-axis is always SUCCESS ------------
st.subheader("Proportion Plots: Feature vs. Success")

success_features = ['RELPER', 'CURREL_SEGMENTED', 'FERTREC', 'INC_SDT1', 'USGEN', 'PARTY', 'REGION', 'ATTNDPERRLS', 'RELIMP', 'GOD', 'HVN', 'PRAY', 'RTRT', 'CHILDREN', 'BIRTHDECADE', 'RACECMB', 'IDEO', 'INTFREQ', 'GENDER']  # features to compare with SUCCESS
x_success = st.selectbox("Select x-axis feature (vs. SUCCESS):", success_features, key="success_plot")

df_tree = df_tree[df_tree[x_success] != 99]
# Also make sure to filter 'CURREL' != 900000 if still needed
df_tree = df_tree[df_tree['CURREL'] != 900000]
# also filter our output of HAPPY
df_tree = df_tree[df_tree['HAPPY'] != 99]

if x_success in df_tree.columns:
    prop_plot = df_tree.groupby([x_success, 'SUCCESS']).size().unstack().fillna(0)
    prop_plot = prop_plot.div(prop_plot.sum(axis=1), axis=0)

    # Gradient color palette
    num_bins = prop_plot.shape[1]
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / num_bins) for i in range(num_bins)]

    # Create plot
    prop_plot.plot(kind='bar', stacked=True, color=colors)
    plt.title(f"Proportions of Success by {x_success}")
    plt.ylabel("Proportion")
    if x_success == 'RELPER':
        plt.xticks(ticks=[0,1,2,3], labels=['Very Religious', 'Somewhat Religious', 'Not Too Religious', 'Not at All Religious'])
    else:
        plt.xticks(rotation=0)
    plt.legend(title='Success', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)



# Features to consider
features = ['RELPER', 'CURREL_SEGMENTED', 'FERTREC', 'INC_SDT1', 'SUCCESS', 'USGEN', 'PARTY', 'REGION', 'ATTNDPERRLS', 'RELIMP', 'GOD', 'HVN', 'PRAY', 'RTRT', 'CHILDREN', 'BIRTHDECADE', 'RACECMB', 'IDEO', 'INTFREQ', 'GENDER']

# Step 1: Let user pick a subset of features
selected_features_pool = st.multiselect("Select features to explore:", features, default=[])

# Generate all combinations
feature_combos = []
for r in range(1, len(selected_features_pool) + 1):
    feature_combos.extend(itertools.combinations(selected_features_pool, r))

combo_names = [', '.join(combo) for combo in feature_combos]

for feature in selected_features_pool:
    df_tree = df_tree[df_tree[feature] != 99]
# Also make sure to filter 'CURREL' != 900000 if still needed
df_tree = df_tree[df_tree['CURREL'] != 900000]
# also filter our output of HAPPY
df_tree = df_tree[df_tree['HAPPY'] != 99]

# If df_tree is empty after filtering, show warning and stop
if df_tree.empty:
    st.warning("You've selected too many features. Reduce your selection for data to populate.")
elif len(selected_features_pool) == 0:
    st.warning("Please select at least one feature to display model outputs.")
else:
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

    # Tree depth selector
    max_depth = st.slider("Select max depth of the Decision Tree:", min_value=1, max_value=10, value=5, step=1)

    # Tree combo selector
    selected_combo_tree = st.selectbox("Choose Features for Decision Tree:", combo_names, key='tree_combo')
    selected_features_tree = selected_combo_tree.split(', ')

    # Define X and y
    X_tree = df_tree[selected_features_tree]
    y_tree = df_tree['HAPPY_ONE_THREE']

    # Fit the decision tree
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
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
