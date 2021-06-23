
# import libraries

from django.shortcuts import render
import joblib
import numpy as np


def home(request):
    return render(request , 'home.html')


def result(request):
    #

    # extract model
    model = joblib.load('finalize.sav')

    # collect features from home page

    ls = []
    ls.append(float(request.GET['sepal_length']))
    ls.append(float(request.GET['sepal_width']))
    ls.append(float(request.GET['petal_length']))
    ls.append(float(request.GET['petal_width']))
    print(ls)
    print(type(ls[0]))

    # convert list into array
    ls = np.array(ls)
    ls = ls.reshape(1 , -1)
    print(ls)

    # prediction
    yp = model.predict(ls)
    print(yp)   # [2]

    y_pred = []
    if yp[0] == 0:
        y_pred.append('Iris-setosa')
    elif yp[0] == 1:
        y_pred.append('Iris-versicolor')
    else:
        y_pred.append('Iris-virginica')

    print(y_pred)

    return render(request, 'result.html' , {'predict' : y_pred[0]})
