# Quick tutorial for Data Frame

## Introduction

Next to Matplotlib and NumPy, Pandas is one of the most widely used Python libraries in data science. It is mainly used for data munging, and with good reason: it’s very powerful and flexible, among many other things. It makes the least sexy part of the "sexiest job of the 21st Century" a bit more pleasant.

### How to create a dataframe ?

You can either start from scratch to define your data or you can convert other data structure to DataFrame.

#### Numpy Array:

We will see here how to convert NumPy Array to DataFrame.

```python
data = np.array([['','Col1','Col2'],
                ['Row1',1,2],
                ['Row2',3,4]])
                
print(pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:]))
```
