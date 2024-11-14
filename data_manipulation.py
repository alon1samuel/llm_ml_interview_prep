"""
Load a CSV file into a Pandas DataFrame and display the first 5 rows.
Use PySpark to filter rows where a column value is greater than a threshold.
Perform a group-by operation and calculate the average of each group.
Merge two DataFrames on a common column.
Fill missing values in a DataFrame with the column mean.
Use Pandas to create a pivot table from a dataset.
Write a PySpark job to count distinct values in a column.
Remove duplicate rows in a Pandas DataFrame.
Add a new calculated column to a DataFrame using a vectorized operation.
Compare the performance of .apply() vs. vectorized operations in Pandas.

"""

import pandas as pd
import pyspark 


# Load a CSV file into a Pandas DataFrame and display the first 5 rows.

path = "data/employment_data.csv"
df = pd.read_csv(path)
print(df.head())

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Practise').getOrCreate()

# df_spark = spark.createDataFrame(df)

# print(df_spark.show(5))


# Use PySpark to filter rows where a column value is greater than a threshold.

# df_spark.filter(df_spark.Data_value > 80000).show()


len(df)

# Perform a group-by operation and calculate the average of each group.

df.columns
df.head()
print(df.groupby('Period')['Data_value'].mean().head())

# Merge two DataFrames on a common column.

# Data for the first DataFrame
data1 = {
    "ID": [1, 2, 3, 4],
    "Name": ["John", "Alice", "Bob", "Emma"],
    "Age": [25, 30, 22, 28]
}

# Data for the second DataFrame
data2 = {
    "ID": [1, 2, 3, 5],
    "Department": ["Engineering", "HR", "Marketing", "Sales"],
    "Salary": [70000, 50000, 60000, 55000]
}

# Create the DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

print(df1.merge(df2, how='left', on='ID'))

# Fill missing values in a DataFrame with the column mean.

filled_df = df.copy()
filled_df['Data_value'] = filled_df['Data_value'].fillna(value=df['Data_value'].mean())

# Use Pandas to create a pivot table from a dataset.


df.pivot(index='Period', columns='Series_reference', values='Data_value').head()


# Write a PySpark job to count distinct values in a column.

df_spark = spark.createDataFrame(df)

len(df_spark.select('Period').distinct().collect())
len(df['Period'].unique())


# Remove duplicate rows in a Pandas DataFrame.

print(len(df))
print(len(df.drop_duplicates()))

df[['Period', 'UNITS']].drop_duplicates()


# Add a new calculated column to a DataFrame using a vectorized operation.

df.info()
df['Magnitude'].describe()
df['magnitude_value'] = df['Data_value']* df['Magnitude']


# Compare the performance of .apply() vs. vectorized operations in Pandas.

from time import time

start_time = time()
df['magnitude_value'] = df['Data_value']* df['Magnitude']
print(f"{(time()-start_time):.6f}")

start_time = time()
df['magnitude_value'] = df.apply(lambda row: row['Data_value']*row['Magnitude'], axis=1)
print(f"{(time()-start_time):.6f}")




print()