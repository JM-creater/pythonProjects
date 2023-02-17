import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# import the dataset
data_df = pd.read_csv('grade_salary.csv')

# Fitting Linear Regression to the dataset
ml = LinearRegression()
ml.fit(data_df[['gender', 'gwa_3rd_year_college', 'gwa_4th_year_college']].values, data_df.salary_after_5_years)

# Visualizing the Polynomial Regression results
def viz_linear():
    data_df['prediction_salary'] = ml.predict(data_df[['gender', 'gwa_3rd_year_college', 'gwa_4th_year_college']])
    plt.title('Salary with GWA of 3rd and 4th year college (Linear)')
    plt.xlabel('3rd year GWA and 4th year GWA')
    plt.ylabel('Salary after 5 years')
    plt.scatter(data_df.gender, data_df.salary_after_5_years, color='black', marker='.')
    plt.plot(data_df.gender, data_df.prediction_salary)
    plt.show()
viz_linear()

# test_model functon to test the model
def test_model():
    model_salary = ml.predict([[2, 4.5, 4.8]])
    print("%.2f" % model_salary)
test_model()











