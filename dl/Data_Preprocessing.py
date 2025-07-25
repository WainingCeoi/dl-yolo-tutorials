import pandas as pd


'''
Quick Notes:
Before passing data into numpy or tensor, etc. We read data using Pandas from CSV files.
Some variables might need to convert from string to number, e.g. float, int, etc.
If we don't won't convert string into dummy variables, cause dummy variables might increase the number of variables.
Thus, we can use below ways to do so which are not using "pd.to_dummy"
'''

# Read original dataset and store as DataFrame
df = pd.read_csv(r"./Datasets/Iris.csv")
print(df)

'''
Method 1:
Really Not Recommended.
1. Manual type in the elements name and indicate their numerical number, might cause error.
2. If variable contains tons of different type of elements, the workload will be horrible.
3. Not easy to convert enumerate type back to it original name.
   e.g. After training AI/ML model, we might need to convert enumerate type variables back to it original type.
'''
df1 = df.copy()
variety_dic = {"Setosa": 0,
               "Versicolor": 1,
               "Virginica": 2}
df1["variety"] = df1["variety"].map(variety_dic)
print(df1)


'''Method 2:
Recommended.
1. Automatic encode the object as an enumerated type or categorical variable.
2. No manual errors.
3. Easy to reverse convertion process. 
'''
# Factorize
df2 = df.copy()
code, uniques = pd.factorize(df2["variety"])
df2["variety"] = code
print(df2)

# Take a glance of "uniques"
print(uniques)

# Reverse Factorize
original_variety = uniques[code]

print(original_variety)

print(uniques[0])
print(uniques[1])
print(uniques[2])