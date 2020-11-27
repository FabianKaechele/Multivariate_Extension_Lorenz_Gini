import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
from matplotlib.colors import LightSource

####--------------------------------####
# This is the corresponing Python code to the paper XXXcitationXXX.
####--------------------------------####
def gini(X):
    """
             Calculates univariate Gini-coeffienct of one dimensional data array

             Input:
                 X:        distribution of single feature over people

             Output:
                univariate Gini coefficient
             """

    sorted_X = np.array(X.copy())
    sorted_X.sort()
    n = X.size
    help1 = 2.0 / n
    help2 = (n + 1.0) / n
    weighted_sum = sum([(i + 1) * j for i, j in enumerate(sorted_X)])

    return help1 * weighted_sum / (sorted_X.sum()) - help2


def x_stern(X):
    """
             Calculates X* out of values from X for arbitrary dimensions, as defined in formula (6)


             Input:
                 X:        distribution of features over people

             Output:
                df:         X* values


             """

    X1 = np.copy(X)
    df = pd.DataFrame(X1)
    df.reset_index(inplace=True, drop=True)
    columns = list(df)

    for column in columns:
        sorted_arr = np.array(df[column].copy())
        sorted_arr.sort()
        sorted_arr1 = np.array(df[column])
        sorted_arr1 = (st.rankdata(sorted_arr1, method='max') - 1).astype(int)
        sum = np.sum(df[column])
        cum_sum = np.cumsum(sorted_arr)
        X_stern = cum_sum / sum
        X_stern_sortiert = np.empty(shape=np.size(df[column]))
        help = np.size(df[column])
        for i in range(0, help):
            X_stern_sortiert[i] = X_stern[sorted_arr1[i]]
        df[column] = X_stern_sortiert

    return df


def mult_gini(X_stern_values):
    """
             Calculates multivariate Gini coefficient, calculated by formula (7)

             Input:
                 X_stern_values:        Matrix with X^* values

             Output:
                megc:                    multivariate extension of the Gini coefficient


             """
    d = np.size(X_stern_values, axis=1)
    n = np.size(X_stern_values, axis=0)
    X_stern_values = np.array(X_stern_values)
    inte = 0
    for i in range(0, n):
        inner = 0
        for j in range(0, d):
            if inner == 0:
                inner = (1 - X_stern_values[i, j])
            else:
                inner = inner * (1 - X_stern_values[i, j])

        inte = inte + inner
    inte = inte / n
    a = math.factorial(d + 1)
    b = inte * a - 1
    megc = (inte * math.factorial(d + 1) - 1) / (math.factorial(d + 1) - 1)
    return megc


def emp_dist(C, Y, X_stern):
    """
             Calculates the empirical distribution of X^* in given grid to draw the MEILC

             Input:
                 C:        grid positions in first direction
                 Y:        grid positions in second direction
                 X:        values of X^*

             Output:
                T:


             """

    grid = np.size(C, axis=0)
    T = np.zeros(shape=(grid, grid))
    X_copy = np.copy(X_stern)
    n = np.size(X_copy, axis=0)
    X_helper = np.copy(X_stern)
    for i in range(0, grid):
        for j in range(0, grid):
            X_helper[:, 0] = np.where(X_copy[:, 0] < C[i], 1, 0)
            X_helper[:, 1] = np.where(X_copy[:, 1] < Y[j], 1, 0)
            X_helper1 = np.where(np.sum(X_helper, axis=1) == 2, 1, 0)
            T[i, j] = np.sum(X_helper1) / n
    return T


def mult_lorenz(datamult):
    """
                Draw 2-dimensional Lorenz curve and safe it to working directory as .png

                Input:
                    data:        Input data

                Output:


                """
    # check dimensions
    if np.size(datamult, axis=1) != 2:
        raise ValueError('Data should be 2 dimsional to plot MEILC')

    # calculate X^* values
    X_stern_values = x_stern(datamult)

    # calculate MEGC and univariate Ginis
    megc = mult_gini(X_stern_values)
    gini_x = gini(datamult.iloc[:, 0])
    gini_y = gini(datamult.iloc[:, 1])

    # create grid and calculate cdf of X*
    X_stern = np.array(X_stern_values)
    C = np.arange(0, 1, 0.01)
    Y = np.arange(0, 1, 0.01)
    C, Y = np.meshgrid(C, Y)
    Z = np.array(emp_dist(C[0, :], Y[:, 0], X_stern))

    # curve starts at 0
    Z[:, 0] = 0
    Z[0, :] = 0

    # plot MEILC
    white = np.ones((Z.shape[0], Z.shape[1], 3))
    blue = white * np.array([0.01, 0.01, 0.01])
    light = LightSource(132, -20)
    illuminated_surface = light.shade_rgb(blue, Z * 10, fraction=1)
    fig = plt.figure(figsize=(11, 7))
    fig.patch.set_facecolor('white')
    ax = plt.axes(projection='3d')
    ax.patch.set_facecolor('white')
    surf = ax.plot_surface(C, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False,
                           facecolors=illuminated_surface)
    ax.set_zlim3d(0, 1)
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.view_init(elev=20, azim=232)
    hfont = {'fontname': 'Times New Roman'}
    ax.set_xlabel('\nFeature 2'+'\n Gini=' + str(round(gini_y,2)))
    ax.set_ylabel('\nFeature 1 ' +'\n Gini=' + str(round(gini_x,2)))
    ax.set_zlabel('Proportion', fontsize=10,**hfont)
    ax.set_title('MEILC with MEGC = '+str(round(megc,2)),fontsize=10,**hfont)
    plt.savefig('Mult_Lorenz.png', format='png',bbox_inches="tight", dpi=1200)
    plt.show()


