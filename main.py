import pandas as pd
# from ydata_profiling import ProfileReport

data = pd.read_csv('diabetes.csv')
target = 'Outcome'

# profile = ProfileReport(data, title='Diabetes Data Profiler', explorative=True)
# profile.to_file('Diabetes_Data_Profiler.html')