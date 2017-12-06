# Quick tutorial for DataFrame

## Introduction

Next to Matplotlib and NumPy, Pandas is one of the most widely used Python libraries in data science. It is mainly used for data munging, and with good reason: it’s very powerful and flexible, among many other things. It makes the least sexy part of the "sexiest job of the 21st Century" a bit more pleasant.

## 1. How to create a DataFrame ?

You can either start from scratch to define your data or you can convert other data structure to DataFrame.

### Numpy Array:

We will see here how to convert NumPy Array to DataFrame.

```python
data = np.array([['','Col1','Col2'],
                ['Row1',1,2],
                ['Row2',3,4]])
                
print(pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:]))
```

Other examples:

```python
# Take a 2D array as input to your DataFrame 
my_2darray = np.array([[1, 2, 3], [4, 5, 6]])
print(pd.DataFrame(my_2darray))

# Take a dictionary as input to your DataFrame 
my_dict = {1: ['1', '3'], 2: ['1', '2'], 3: ['2', '4']}
print(pd.DataFrame(my_dict))

# Take a DataFrame as input to your DataFrame 
my_df = pd.DataFrame(data=[4,5,6,7], index=range(0,4), columns=['A'])
print(pd.DataFrame(my_df))

# Take a Series as input to your DataFrame
my_series = pd.Series({"United Kingdom":"London", "India":"New Delhi", "United States":"Washington", "Belgium":"Brussels"})
print(pd.DataFrame(my_series))
```

### CSV File:

### From Scratch:

### Information of a DataFrame
```
df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))
```
* Shape:
Dimension of the Dataframe (width & height)
```python
df.shape()
```
* Index:
```python
df.index()
```
* Length:
Height of the dataframe
```python
len(df.index)
```

### Fundamental DataFrame Manipulations:

We will cover the basic operations that you can do on your newly made DataFrame: adding, selecting, deleting, renaming, … You name it!

## 2. How To Select an Index or Column From a Pandas DataFrame ?

### Select an element:

The most important ones to remember are, without a doubt, loc and iloc. 

```python
# Using `iloc[]`
print(df.iloc[0][0])

# Using `loc[]`
print(df.loc[0]['A'])

# Using `at[]`
print(df.at[0,'A'])

# Using `iat[]`
print(df.iat[0,0])

# Using `get_value(index, column)`
print(df.get_value(0, 'A'))
```

  * loc works on labels of your index. This means that if you give in loc[2], you look for the values of your DataFrame that have an index labeled 2.
  * iloc works on the positions in your index. This means that if you give in iloc[2], you look for the values of your DataFrame that are at index ’2`.

### Select Rows and Columns:

```python
# Use `iloc[]` to select row `0`
print(df.iloc[_])

# Use `loc[]` to select column `'A'`
print(df.loc[:,'_'])
```

## 3. How To Add an Index, Row or Column to a Pandas DataFrame?

### Index:

When you create a DataFrame, you have the option to add input to the ‘index’ argument to make sure that you have the index that you desire. When you don’t specify this, your DataFrame will have, by default, a numerically valued index that starts with 0 and continues until the last row of your DataFrame.

However, even when your index is specified for you automatically, you still have the power to re-use one of your columns and make it your index. You can easily do this by calling set_index() on your DataFrame.

```python
# Set 'C' as the index of your DataFrame
df.set_index('C')
```
### Adding a Row:

```python
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2.5, 12.6, 4.8], columns=[48, 49, 50])
df.loc[2]=[11,12,13]
```

### Adding a Column:

```python
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['A', 'B', 'C'])
df.loc['D'] = pd.Series(['5', '6'], index=df.index)
```

## 4. How to Delete Indices, Rows or Columns From a Pandas Data Frame?

## 5. How to Rename the Index or Columns of a Pandas DataFrame?

## 6. How To Format The Data in Your Pandas DataFrame?

### Replacing All Occurrences of a String in a DataFrame

### Removing Parts From Strings in the Cells of Your DataFrame

### Splitting Text in a Column into Multiple Rows in a DataFrame

### Applying A Function to Your Pandas DataFrame’s Columns or Rows

## 7. How To Create an Empty DataFrame?

## 8. Does Pandas Recognize Dates When Importing Data?

## 9. When, Why And How You Should Reshape Your Pandas DataFrame?

## 10. How To Iterate Over a Pandas DataFrame?

## 11. How To Write a Pandas DataFrame to a File
