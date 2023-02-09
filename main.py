import shutil
from collections import Counter

import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import shap
import lime
from lime import lime_tabular
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

def fillmissing(data, feature, method):
  if method == "mode":
    data[feature] = data[feature].fillna(data[feature].mode()[0])
  elif method == "median":
    data[feature] = data[feature].fillna(data[feature].median())
  else:
    data[feature] = data[feature].fillna(data[feature].mean())

def initializeWeights(L_in, L_out):
  epsilon_init = 0.12
  W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
  return W

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel="linear"))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c="gray", edgecolors=(0, 0, 0))
    plt.scatter(
        X[zero_class, 0],
        X[zero_class, 1],
        s=160,
        edgecolors="b",
        facecolors="none",
        linewidths=2,
        label="Class 1",
    )
    plt.scatter(
        X[one_class, 0],
        X[one_class, 1],
        s=80,
        edgecolors="orange",
        facecolors="none",
        linewidths=2,
        label="Class 2",
    )

    plot_hyperplane(
        classif.estimators_[0], min_x, max_x, "k--", "Boundary\nfor class 1"
    )
    plot_hyperplane(
        classif.estimators_[1], min_x, max_x, "k-.", "Boundary\nfor class 2"
    )
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - 0.5 * max_x, max_x + 0.5 * max_x)
    plt.ylim(min_y - 0.5 * max_y, max_y + 0.5 * max_y)
    if subplot == 2:
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.legend(loc="upper left")


if __name__ == '__main__':
    values = ["None", "Shap", "Lime", "BMI Calculator"]
    st.set_option('deprecation.showPyplotGlobalUse', False)
    choose_option = st.sidebar.selectbox(
        "Choose a option",
        values, index = values.index("None")
    )

    data = pd.read_csv('https://raw.githubusercontent.com/tipemat/dataset/master/swenounou.csv', header=0)
    data = data.dropna()

    if (choose_option == "None"):
        #dataset
        st.title('Cancerul de prostata')
        st.write("Cancerul de prostată este o formă de cancer care se dezvoltă în prostată, o glandă aflată la intersecția aparatului urinar cu cel genital la bărbați. Mare majoritate a cancerelor de prostată se dezvoltă lent, cu toate acestea, există cazuri în care cancerul de prostată se dezvoltă în mod agresiv.")
        st.write("Cancerul de prostată tinde să se dezvolte la bărbații cu vârsta de peste cincizeci de ani și, deși este una dintre cele mai răspândite tipuri de cancer la bărbați mulți nu au simptome, nu se tratează, și în cele din urmă mor din alte cauze. Acest lucru este din cauza faptului că cancerul de prostată este, în majoritatea cazurilor, cu dezvoltare lentă, asimptomatică și oamenii în vârstă adesea vor muri din cauze care nu au legătură cu cancerul de prostată, cum ar fi inimă / boli circulatorii, pneumonie , alte tipuri de cancer care nu au legătură cu prostata, sau din cauza vârstei înaintate. Pe de altă parte, cazurile de cancer de prostată mai agresive reprezintă cauza pentru mai multe decese legate de cancer în rândul bărbaților din Statele Unite, decât orice alt cancer, cu excepția cancerului pulmonar. Aproximativ două treimi din cazurile de cancer de prostată sunt cu dezvoltare lentă, iar o treime cu dezvoltare mai agresivă și mai rapidă.")
        st.write(" ")
        st.write("Pentru predictia cancerului de prostata am utilizat un set de date, care cuprinde date de la pacienti reali")
        st.write('Setului de date analizat si utilizat in predictia cancerului de prostata')
        #print dataset
        st.write(data)

        #prezent statistic of EXHP
        fig, ax = plt.subplots(figsize=(8, 4))
        plt = sns.countplot(data=data, x=data['EXHP'], label="Count")
        st.write("Numarul de pacienti care sufera de cancer de prostata")
        st.pyplot(fig)

    y = data.loc[:, data.columns == 'EXHP']

    if (choose_option == "Shap"):
        #predictie
        st.title("Shap prediction")
        st.subheader(
            "Shap = SHapley Additive exPlanations - is a method based on cooperative game theory and used to increase transparency and interpretability of machine learning models.")

        btn_aleg_comp = st.multiselect("Choose the fragments you want to analyze: ",
                                          ["Fragm1", "Fragm2", "Fragm3", "Fragm4", "Fragm5",
                                           "Fragm6", "Fragm7", "Fragm8", "Fragm9", "Fragm10", "Fragm11", "Fragm12"],
                                          ["Fragm1", "Fragm2", "Fragm3", "Fragm4", "Fragm5",
                                           "Fragm6", "Fragm7", "Fragm8", "Fragm9", "Fragm10", "Fragm11", "Fragm12"]
                                          )
        test_size_choose = st.slider(label="Choose the percentage of data to be tested:",
                                     min_value=10,
                                     max_value=40,
                                     value=10,
                                     step=1)
        if (btn_aleg_comp):
            X_select = data[btn_aleg_comp]
            X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size = test_size_choose, random_state=42)
            st.write('The chosen prediction model is: RandomForest.'
                     '\n- We used Shapley values to explain any machine learning model;'
                     '\n- We used SHAP algorithm, because it is a local feature attribution technique that explains every prediction from the model as a sum of individual feature contributions;'
                     '\n- SHAP values explain the raw predictions from the leaf nodes of the trees.')
            st.write("* A density scatter plot of SHAP values for each feature to identify how much impact each feature has on the model output. Features are sorted by the sum of the SHAP value magnitudes across all samples.")
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            shap_values = shap.TreeExplainer(model).shap_values(X_train)
            shap.summary_plot(shap_values, X_train, plot_type="bar")
            st.pyplot()

#             st.write('Blue color indicates that x value decreased the prediction and red color indicates that y value increased the prediction.')
#             st.write("If we can se, we get grey colored points for categorical data as the integer encoded values can not be always used to arrange it from low to high.")
#             shap_values = shap.TreeExplainer(model).shap_values(X_train)
#             shap.summary_plot(shap_values, X_select)
#             st.pyplot()

            svm = sklearn.svm.SVC(kernel='rbf', probability=True)
            svm.fit(X_train, y_train)
            explainer = shap.KernelExplainer(svm.predict_proba, X_train)
            shap_values = explainer.shap_values(X_test.iloc[0, :])
            st.write(X_test.iloc[0,:])

            y_shap = LabelEncoder().fit_transform(y)
            oversample = SMOTEENN()
            X_select_shap, y_shap = oversample.fit_resample(X_select, y)
            counter = Counter(y_shap)
            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_select_shap, y_shap, test_size=test_size_choose,
                                                                random_state=42)
            st.write('Explain model used normalization method.')
            model = RandomForestRegressor()
            model.fit(X_train_1, y_train_1)
            shap_values = shap.TreeExplainer(model).shap_values(X_train_1)
            shap.summary_plot(shap_values, X_train_1, plot_type="bar")
            st.pyplot()

            #shap.summary_plot(shap_values, X_select_shap)
            #st.pyplot()

#             svm = sklearn.svm.SVC(kernel='rbf', probability=True)
#             svm.fit(X_train_1, y_train_1)
#             explainer = shap.KernelExplainer(svm.predict_proba, X_train_1)
#             shap_values = explainer.shap_values(X_test_1.iloc[0, :])
#             st.write(X_test_1.iloc[0, :])

    if(choose_option == "Lime"):
        st.title("LIME prediction")
        st.subheader("LIME = Local Interpretable Model-agnostic Explanations - is used to explain predictions of a machine learning model.")

        test_size_choose = st.slider(label="Choose the percentage of data to be tested:",
                                     min_value=10,
                                     max_value=40,
                                     value=10,
                                     step=1)
        btn_aleg_comp = st.multiselect("Choose the fragments you want to analyze: ",
                                       ["Fragm1", "Fragm2", "Fragm3", "Fragm4", "Fragm5",
                                        "Fragm6", "Fragm7", "Fragm8", "Fragm9", "Fragm10", "Fragm11", "Fragm12"],
                                       ["Fragm1", "Fragm2", "Fragm3", "Fragm4", "Fragm5",
                                        "Fragm6", "Fragm7", "Fragm8", "Fragm9", "Fragm10", "Fragm11", "Fragm12"]
                                       )
        if (btn_aleg_comp):
            st.write("From the plots, you can see the intervals between which each characteristic should be in order to obtain the best possible prediction.")
            st.write("* MLPClassifier algorithm is used when training our model.")
            X_select = data[btn_aleg_comp]
            X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size = test_size_choose, random_state = 42)
            X_train = X_train.astype('int')
            y_train = y_train.astype('int')
            X_test = X_test.astype('int')
            y_test = y_test.astype('int')

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
            st.set_option('deprecation.showPyplotGlobalUse', False)
            exp.as_pyplot_figure()
            st.pyplot()
            if (accuracy_score(y_test, clf.predict(X_test)) > 70):
                st.write("We can see that the prediction was successful, with a result:")
            else:
                st.write("We can see that the prediction was not successful, with a result:")
            st.write("Test  Accuracy : %.2f" % accuracy_score(y_test, clf.predict(X_test)))
            st.write("Train  Accuracy : %.2f" % accuracy_score(y_train, clf.predict(X_train)))

            st.write("\n\n* GradientBoostingClassifier algorithm is used when training our model.")
            gb = GradientBoostingClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size = test_size_choose, random_state = 42)
            X_train = X_train.astype('int')
            y_train = y_train.astype('int')
            X_test = X_test.astype('int')
            y_test = y_test.astype('int')
            gb.fit(X_train, y_train)
            exp = explainer.explain_instance(
                data_row=X_test.iloc[0],
                predict_fn=gb.predict_proba
            )
            exp.as_pyplot_figure()
            exp.show_in_notebook()
            st.pyplot()
            if (accuracy_score(y_test, gb.predict(X_test)) > 70):
                st.write("We can see that the prediction was successful, with a result:")
            else:
                st.write("We can see that the prediction was not successful, with a result:")
            st.write("Test  Accuracy : %.2f" % accuracy_score(y_test, gb.predict(X_test)))
            st.write("Train  Accuracy : %.2f" % accuracy_score(y_train, gb.predict(X_train)))

            st.write("\n\n* RandomForestClassifier algorithm is used when training our model.")
            rf = RandomForestClassifier(n_estimators=100)
            X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=test_size_choose, random_state=42)
            X_train = X_train.astype('int')
            y_train = y_train.astype('int')
            X_test = X_test.astype('int')
            y_test = y_test.astype('int')
            rf.fit(X_train, y_train)

            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=X_train.columns,
                class_names=[0, 1],
                mode='classification'
            )

            exp = explainer.explain_instance(
                data_row=X_test.iloc[0],
                predict_fn=rf.predict_proba
            )
            exp.as_pyplot_figure()
            exp.show_in_notebook()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            if (accuracy_score(y_test, rf.predict(X_test)) > 70):
                st.write("We can see that the prediction was successful, with a result:")
            else:
                st.write("We can see that the prediction was not successful, with a result:")
            st.write("Test  Accuracy : %.2f" % accuracy_score(y_test, rf.predict(X_test)))
            st.write("Train  Accuracy : %.2f" % accuracy_score(y_train, rf.predict(X_train)))

            st.write("\n\n* LogisticRegression algorithm is used when training our model.")
            lr = LogisticRegression()
            X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size = test_size_choose, random_state = 42)
            X_train = X_train.astype('int')
            y_train = y_train.astype('int')
            X_test = X_test.astype('int')
            y_test = y_test.astype('int')
            lr.fit(X_train, y_train)

            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=X_train.columns,
                class_names=[0, 1],
                mode='classification'
            )

            exp = explainer.explain_instance(
                data_row=X_test.iloc[0],
                predict_fn=lr.predict_proba
            )
            exp.as_pyplot_figure()
            exp.show_in_notebook()
            st.pyplot()
            if (accuracy_score(y_test, lr.predict(X_test)) > 70):
                st.write("We can see that the prediction was successful, with a result:")
            else:
                st.write("We can see that the prediction was not successful, with a result:")
            st.write("Test  Accuracy : %.2f" % accuracy_score(y_test, lr.predict(X_test)))
            st.write("Train  Accuracy : %.2f" % accuracy_score(y_train, lr.predict(X_train)))

            y_trans = LabelEncoder().fit_transform(y)
            oversample = SMOTE()
            X_select_trans, y_trans = oversample.fit_resample(X_select, y)
            counter = Counter(y_trans)
            st.write("\n\n* LogisticRegression algorithm is used when training our model when the dataset is normalize.")
            lr = LogisticRegression()
            X_train, X_test, y_train, y_test = train_test_split(X_select_trans, y_trans, test_size = test_size_choose, random_state = 42)
            X_train = X_train.astype('int')
            y_train = y_train.astype('int')
            X_test = X_test.astype('int')
            y_test = y_test.astype('int')
            lr.fit(X_train, y_train)

            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=X_train.columns,
                class_names=[0, 1],
                mode='classification'
            )

            exp = explainer.explain_instance(
                data_row=X_test.iloc[0],
                predict_fn=lr.predict_proba
            )
            exp.as_pyplot_figure()
            exp.show_in_notebook()
            st.pyplot()
            if (accuracy_score(y_test, lr.predict(X_test)) > 70):
                st.write("We can see that the prediction was successful, with a result:")
            else:
                st.write("We can see that the prediction was not successful, with a result:")
            st.write("Test  Accuracy : %.2f" % accuracy_score(y_test, lr.predict(X_test)))
            st.write("Train  Accuracy : %.2f" % accuracy_score(y_train, lr.predict(X_train)))


    if (choose_option == 'BMI Calculator'):
        st.title('BMI Calculator')
        st.subheader('=  Body Mass Index')
        st.write("<div style='text-align: justify;font-size: 16px;'><br><br>Pentru calculul Indicelui de Masă Corporală se utilizează următoarea formulă:<br><br></div>",unsafe_allow_html=True)
        st.write(r"$$\color{orange}BMI=\frac{greutate}{înălțime^2}$$")
        col1, col2 = st.columns(2)
        greutate = col1.text_input("Greutate (kg)", 50)
        inaltime = col2.text_input("Înălțime (m)",1.6)
        btn_calc=st.button("Calculează")
        if (',' in greutate):
          greutate= greutate.split(',')[0] + '.' + greutate.split(',')[1]
        if(' 'in greutate):
          greutate = greutate.split(' ')[0] + greutate.split(' ')[1]
        if (',' in inaltime):
          inaltime= inaltime.split(',')[0] + '.' + inaltime.split(',')[1]
        if (' ' in inaltime):
          inaltime = inaltime.split(' ')[0] + inaltime.split(' ')[1]
        try:
          float(greutate)
          float(inaltime)
        except ValueError:
          st.write("Vă rugăm să introduceți date valide.")
        if(btn_calc and float(greutate)>0.0 and float(inaltime)>0.0):
          if (',' in greutate):
            intreg=greutate.split(',')[0]+'.'+greutate.split(',')[1]
          rezultatBMI=float(greutate)/((float(inaltime))*(float(inaltime)))
          st.write("<div style='text-align: justify;font-size: 16px; color: orange'><br><b>Rezultat:</b></div>",unsafe_allow_html=True)
          st.write("BMI-ul dumneavoastră este: % .2f." %(rezultatBMI))
          if(float(rezultatBMI)<18.5):
            st.write("\n* **Subponderal**"
                 "\n* Un risc ridicat de sănătate, dacă nu alegi o dietă sănătoasă, bogată în nutrienţi.")
          if (float(rezultatBMI) >= 18.5 and float(rezultatBMI) < 25):
            st.write("\n* **Greutate normală** - aveți o greutate perfectă.")
          if (float(rezultatBMI) >= 25 and float(rezultatBMI) <30):
            st.write("\n* **Supraponderal** - aveți o greutate moderată.")
          if (float(rezultatBMI) >= 30 and float(rezultatBMI) <35):
            st.write("\n* **Obezitate (gradul I)** - recomandare: să eliminați dulciurile şi alimentele nesănătoase din alimentație.")
          if (float(rezultatBMI) >= 35and float(rezultatBMI) <40):
            st.write("\n* **Obezitate (gradul II)** - greutatea vă afectează sănătatea.")
          if (float(rezultatBMI) >=40):
            st.write("\n* **Obezitate morbidă** - greutatea vă afectează grav sănătatea.")
        #st.image("resources/bmi-categories.jpeg")
        #st.write("<div  style='font-size: 12px;text-align: center'>Sursă imagine: <a href='https://www.bodycureclinic.fit/images/bmi.jpeg'>https://www.bodycureclinic.fit/images/bmi.jpeg</a></div><br>", unsafe_allow_html=True)
        st.write("\nCreșterea greutății corporale crește riscul apariției unor probleme de sănătate, cum ar fi:"
             "\n* bolile cardiovasculare;"
             "\n* insuficiența cardiacă;"
             "\n* hipertensiunea arterială;"
             "\n* infarctul miocardic;"
             "\n* accidentul vascular cerebral;"
             "\n* afecțiunile articulațiilor;"
             "\n* unele **tipuri de diabet**;"
             "\n* unele tipuri de cancer.")
        
