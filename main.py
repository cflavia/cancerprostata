import streamlit as st

try:
    # Streamlit < 0.65
    from streamlit.ReportThread import get_report_ctx

except ModuleNotFoundError:
    try:
        # Streamlit > 0.65
        from streamlit.report_thread import get_report_ctx

    except ModuleNotFoundError:
        try:
            # Streamlit > ~1.3
            from streamlit.script_run_context import get_script_run_ctx as get_report_ctx

        except ModuleNotFoundError:
            try:
                # Streamlit > ~1.8
                from streamlit.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx

            except ModuleNotFoundError:
                # Streamlit > ~1.12
                from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx

import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import lime
from lime import lime_tabular
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

if __name__ == '__main__':
    values = ["None", "Shap", "Lime"]
    choose_option = st.sidebar.selectbox(
        "Choose a option",
        values, index = values.index("None")
    )

    data = pd.read_csv('https://raw.githubusercontent.com/tipemat/dataset/master/swenounou.csv', header=0)
    data = data.dropna()

    if (choose_option == "None"):
        #dataset
        st.title('Cancerul de prostata')
        st.write('Setului de date analizat si utilizat in predictia cancerului de prostata')
        #print dataset
        st.write(data)

        #prezent statistic of EXHP
        plt = sns.countplot(data['EXHP'], label="Count")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Numarul de pacienti care sufera de cancer de prostata")
        st.pyplot(plt = plt)

    X = data.loc[:, data.columns != 'EXHP']
    y = data.loc[:, data.columns == 'EXHP']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if (choose_option == "Shap"):
        #predictie
        X100 = shap.utils.sample(X, 100)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        shap_values = shap.TreeExplainer(model).shap_values(X_train)
        plt = shap.summary_plot(shap_values, X_train, plot_type="bar")
        st.pyplot(plt=plt)

        plt = shap.summary_plot(shap_values, X)
        st.pyplot(plt=plt)

        svm = sklearn.svm.SVC(kernel='rbf', probability=True)
        svm.fit(X_train, y_train)
        explainer = shap.KernelExplainer(svm.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test.iloc[0, :])
        st.write(X_test.iloc[0,:])

        X_select = data.iloc[: , 1:8]
        X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=0.2, random_state=42)
        X100 = shap.utils.sample(X, 100)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        shap_values = shap.TreeExplainer(model).shap_values(X_train)
        plt = shap.summary_plot(shap_values, X_train, plot_type="bar")
        st.pyplot(plt=plt)

        plt = shap.summary_plot(shap_values, X_select)
        st.pyplot(plt=plt)

        svm = sklearn.svm.SVC(kernel='rbf', probability=True)
        svm.fit(X_train, y_train)
        explainer = shap.KernelExplainer(svm.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test.iloc[0, :])
        st.write(X_test.iloc[0, :])

    if(choose_option == "Lime"):
        X_train = X_train.astype('int')
        y_train = y_train.astype('int')
        X_test = X_test.astype('int')

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
        clf.fit(X_train, y_train)

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=[0, 1],
            mode='classification'
        )
        exp = explainer.explain_instance(
            data_row=X_test.iloc[0],
            predict_fn=clf.predict_proba
        )
        exp.as_pyplot_figure()
        exp.show_in_notebook()
        st.pyplot()

        gb = GradientBoostingClassifier()
        gb.fit(X_train, y_train)

        st.write("Test  Accuracy : %.2f" % gb.score(X_test, y_test))
        st.write("Train Accuracy : %.2f" % gb.score(X_train, y_train))
        st.write("Confusion Matrix : ")
        st.write(confusion_matrix(y_test, gb.predict(X_test)))
        st.write("Classification Report")
        st.write(classification_report(y_test, gb.predict(X_test)))

        exp = explainer.explain_instance(
            data_row=X_test.iloc[0],
            predict_fn=gb.predict_proba
        )
        exp.as_pyplot_figure()
        exp.show_in_notebook()
        st.pyplot()
