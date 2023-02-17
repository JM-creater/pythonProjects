import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import the dataset
data_df = pd.read_csv('grade_salary.csv')

# Dividing the dataset into 2 components
x = data_df.iloc[:, 0:3].values
y = data_df.iloc[:, 3:4].values

# split our dataset to get training set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y_train)

# Visualizing the Polynomial Regression results
def viz_linear():
    plt.title('Salary with GWA of 3rd and 4th year college (Polynomial)')
    plt.xlabel('3rd year GWA coland 4th year GWA')
    plt.ylabel('Salary after 5 years')
    plt.title('Plot of Salary with GWA of 3rd and 4th year (Linear)')
    plt.scatter(data_df['gwa_3rd_year_college'], data_df['gwa_4th_year_college'], data_df['salary_after_5_years'], color='black', marker='.')
    plt.show()
viz_linear()

# test_model function to test the model
def test_model():
    test_mod = [[2, 4.5, 4.8]]
    model_salary = lin_reg_2.predict(poly_reg.fit_transform(test_mod))
    print("%.2f" % model_salary)
test_model()

















