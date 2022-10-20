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
    values = ["None", "Shap", "Lime", "Multi-class Classification"]
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

            st.write('Blue color indicates that x value decreased the prediction and red color indicates that y value increased the prediction.')
            st.write("If we can se, we get grey colored points for categorical data as the integer encoded values can not be always used to arrange it from low to high.")
            shap_values = shap.TreeExplainer(model).shap_values(X_train)
            shap.summary_plot(shap_values, X_select, max_display = 5, show = False)
            st.pyplot()

#             svm = sklearn.svm.SVC(kernel='rbf', probability=True)
#             svm.fit(X_train, y_train)
#             explainer = shap.KernelExplainer(svm.predict_proba, X_train)
#             shap_values = explainer.shap_values(X_test.iloc[0, :])
#             st.write(X_test.iloc[0,:])

#             y_shap = LabelEncoder().fit_transform(y)
#             oversample = SMOTE()
#             X_select_shap, y_shap = oversample.fit_resample(X_select, y)
#             counter = Counter(y_shap)
#             X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_select_shap, y_shap, test_size=test_size_choose,
#                                                                 random_state=42)
#             st.write('Explain model used normalization method.')
#             model = RandomForestRegressor()
#             model.fit(X_train_1, y_train_1)
#             shap_values = shap.TreeExplainer(model).shap_values(X_train_1)
#             plt = shap.summary_plot(shap_values, X_train_1, plot_type="bar")
#             st.pyplot(plt=plt)

#             plt = shap.summary_plot(shap_values, X_select_shap)
#             st.pyplot(plt=plt)

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
            exp.as_pyplot_figure()
            exp.show_in_notebook()
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


    if (choose_option == 'Multi-class Classification'):
        st.title('Multi-class Classification')
        st.subheader('=  a classification task with more than two classes')
        data = data.sample(frac=1).reset_index(drop=True)
        Y = data['EXHP']
        X = data.drop(['EXHP'], axis=1)

        y = LabelEncoder().fit_transform(y)

        fig, ax = plt.subplots(figsize=(8, 4))
        counter = Counter(y)
        for k, v in counter.items():
            per = v / len(y) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
        plt.bar(counter.keys(), counter.values())
        plt.show()
        st.pyplot(fig)

        y = LabelEncoder().fit_transform(y)
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        counter = Counter(y)
        for k, v in counter.items():
            per = v / len(y) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
        plt.bar(counter.keys(), counter.values())
        plt.show()
        st.pyplot(plt=plt)

        svc = SVC()
        o_vs_r = OneVsRestClassifier(svc)
        o_vs_r.fit(X, y)
        y_pred = o_vs_r.predict(X)
        plt.plot(y_pred, X, 'ro')
        plt.plot(y, X, 'yo')
        st.pyplot(plt=plt)
        st.write("Accuracy for prediction:", sklearn.metrics.accuracy_score(y, y_pred))
        st.write("Precision for prediction:", sklearn.metrics.precision_score(y, y_pred))

        data = data.sample(frac=1).reset_index(drop=True)
        Y = data['EXHP']
        X = data.drop(['EXHP'], axis=1)
        encoder = OneHotEncoder()
        encoded_Y = encoder.fit(Y.values.reshape(-1, 1))
        encoded_Y = encoded_Y.transform(Y.values.reshape(-1, 1)).toarray()
        train_ratio = 0.70
        validation_ratio = 0.15
        test_ratio = 0.15

        trainX, testX, trainY, testY = train_test_split(X, encoded_Y, test_size=1 - train_ratio)
        valX, testX, valY, testY = train_test_split(testX, testY,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

        re_transformed_array_trainY = encoder.inverse_transform(trainY)
        unique_elements, counts_elements = np.unique(re_transformed_array_trainY, return_counts=True)
        unique_elements_and_counts_trainY = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
        unique_elements_and_counts_trainY.columns = ['unique_elements', 'count']

        re_transformed_array_valY = encoder.inverse_transform(valY)
        unique_elements, counts_elements = np.unique(re_transformed_array_valY, return_counts=True)
        unique_elements_and_counts_valY = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
        unique_elements_and_counts_valY.columns = ['unique_elements', 'count']

        re_transformed_array_trainY = encoder.inverse_transform(trainY)
        unique_elements, counts_elements = np.unique(re_transformed_array_trainY, return_counts=True)
        unique_elements_and_counts_trainY = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
        unique_elements_and_counts_trainY.columns = ['unique_elements', 'count']
        st.write(unique_elements_and_counts_trainY)

        re_transformed_array_testY = encoder.inverse_transform(testY)
        unique_elements, counts_elements = np.unique(re_transformed_array_testY, return_counts=True)
        unique_elements_and_counts_testY = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
        unique_elements_and_counts_testY.columns = ['unique_elements', 'count']
        y_part = [trainY, valY, testY]
        for y_part in y_part:
            re_transformed_array = encoder.inverse_transform(y_part)
            unique_elements, counts_elements = np.unique(re_transformed_array, return_counts=True)
            unique_elements_and_counts = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
            unique_elements_and_counts.columns = ['unique_elements', 'count']
        list_trainY = unique_elements_and_counts_trainY['unique_elements'].to_list()
        list_valY = unique_elements_and_counts_valY['unique_elements'].to_list()
        list_testY = unique_elements_and_counts_testY['unique_elements'].to_list()
        input_shape = trainX.shape[1]
        n_batch_size = 20
        n_steps_per_epoch = int(trainX.shape[0] / n_batch_size)
        n_validation_steps = int(valX.shape[0] / n_batch_size)
        n_test_steps = int(testX.shape[0] / n_batch_size)
        n_epochs = 25
        num_classes = trainY.shape[1]
        st.write('Input Shape: ' + str(input_shape))
        st.write('Dataset Size: ' + str(n_batch_size))
        st.write('Number of Classes: ' + str(num_classes))

        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        checkpoint_no = 'model_pred'
        model_name = 'Bird_ANN_2FC_F64_64_epoch_25'
        checkpoint_dir = './' + checkpoint_no
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        keras_callbacks = [ModelCheckpoint(filepath=checkpoint_dir + '/' + model_name,
                                           monitor='val_loss', save_best_only=True, mode='auto')]
        trainX = trainX.astype('int')
        trainY = trainY.astype('int')
        testX = testX.astype('int')
        testY = testY.astype('int')
        valX = valX.astype('int')
        valY = valY.astype('int')
        history = model.fit(trainX,
                            trainY,
                            steps_per_epoch=n_steps_per_epoch,
                            epochs=n_epochs,
                            batch_size=n_batch_size,
                            validation_data=(valX, valY),
                            validation_steps=n_validation_steps,
                            callbacks=[keras_callbacks])
        hist_df = pd.DataFrame(history.history)
        hist_df['epoch'] = hist_df.index + 1
        cols = list(hist_df.columns)
        cols = [cols[-1]] + cols[:-1]
        hist_df = hist_df[cols]
        hist_df.to_csv(checkpoint_no + '/' + 'history_df_' + model_name + '.csv')
        values_of_best_model = hist_df[hist_df.val_loss == hist_df.val_loss.min()]

        class_assignment = dict(zip(y, encoded_Y))
        df_temp = pd.DataFrame([class_assignment], columns=class_assignment.keys())
        df_temp = df_temp.stack()
        df_temp = pd.DataFrame(df_temp).reset_index().drop(['level_0'], axis=1)
        df_temp.columns = ['Category', 'Allocated Number']
        df_temp.to_csv(checkpoint_no + '/' + 'class_assignment_df_' + model_name + '.csv')
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        st.pyplot()

        model_reloaded = load_model(checkpoint_no + '/' + model_name)
        root_directory = os.getcwd()
        checkpoint_dir = os.path.join(root_directory, checkpoint_no)
        model_name_temp = os.path.join(checkpoint_dir, model_name + '.h5')
        model_reloaded.save(model_name_temp)
        folder_name_temp = os.path.join(checkpoint_dir, model_name)
        shutil.rmtree(folder_name_temp, ignore_errors=True)

        best_model = load_model(model_name_temp)
        test_loss, test_acc = best_model.evaluate(testX,
                                                  testY,
                                                  steps=n_test_steps)
        st.write('Test Accuracy for model:', test_acc)
        y_pred = model.predict(testX)
        st.write('Train Accuracy for model: ', acc[0])
