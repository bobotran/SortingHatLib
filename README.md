# SortingHatInf

SortingHatInf is a library that implements ML-based feature type inference as seen in the paper [here](https://adalabucsd.github.io/papers/2021_SortingHat_SIGMOD.pdf). Feature type inference is the task of predicting the feature types of the columns of a given dataset.

## Feature Types
### SortingHat
- `numeric`
- `categorical`
- `datetime`
- `sentence`
- `url`
- `embedded-number`
- `list`
- `not-generalizable`
- `context-specific`

### Expanded
Same as SortingHat except:
- `numeric` mapped to `integer` or `floating`
- `categorical` mapped to `boolean` if Boolean

### ARFF (loose)
- `Nominal-specification` (Categorical)
- `INTEGER`
- `REAL` (Float)
- `STRING`
- `IGNORE` (Not-Generalizable)

## API Documentation
`get_sortinghat_types(df: pd.DataFrame) -> List[str]` returns a list of the predicted SortingHat feature types on the columns of the specified Pandas dataframe  
Ex. `infer_sh = sortinghatinf.get_sortinghat_types(df)`  
<pre><code>> infer_sh  
> [  
> &nbsp;  'COL_TYPE_1',  
> &nbsp;  'COL_TYPE_2',  
> &nbsp;  ...  
> ]  
</code></pre>

`get_expanded_feature_types(df: pd.DataFrame) -> List[str]` returns a list of the predicted SortingHat feature types on the columns of the specified Pandas dataframe mapped to the expanded types  
Ex. `infer_exp = sortinghatinf.get_expanded_types(df)`  
<pre><code>> infer_exp  
> [    
> &nbsp;  'COL_TYPE_1',  
> &nbsp;  'COL_TYPE_2',  
> &nbsp;  ...   
> ]  
</code></pre>

`get_feature_types_as_arff(df: pd.DataFrame) -> Tuple[List[Tuple[str, Union[str, List[str]]]], List[str]]` returns the predicted SortingHat feature types mapped to the loose ARFF types and the original predicted SortingHat feature types  
Ex. `infer_arff, infer_sh = sortinghatinf.get_expanded_types(df)`  
<pre><code>> infer_arff  
> [  
> &nbsp;  ('COL_NAME_1', ['POSSIBLE_VALUE_1', 'POSSIBLE_VALUE_2', ...]), # NOMINAL  
> &nbsp;  ('COL_NAME_2', 'INTEGER'), # INTEGER  
> &nbsp;  ('COL_NAME_3', 'FLOAT'), # REAL  
> &nbsp;  ('COL_NAME_4', 'STRING'), # STRING  
> &nbsp;  ('COL_NAME_5', 'IGNORE'), # IGNORE  
> &nbsp;  ...  
> ]  
</code></pre>

**Note**: Because ARFF expects a string list for categorical features, columns discovered to be categorical should be converted to string. This function will report these columns with an error.

## Example Usage with OpenML
Here, we run feature type inference on a dataset obtained from OpenML.
Note: this can be done with any dataset loaded as a Pandas dataframe, but we use OpenML here as an example.

1. First ensure `pip`, `wheel`, and `setuptools` are up-to-date.
```
python -m pip install --upgrade pip setuptools wheel
``` 
2. Install the package using python-pip.
```
pip install sortinghatinf
```
3. Import the library.
```
import sortinghatinf
```

4. Install the OpenML python API.
```
pip install openml
```

5. Import the OpenML python library.
```
import openml
```

6. Load the 'Blood Transfusion Service Center' dataset from OpenML (dataset_id=31).
Note: This requires an OpenML account which you can setup by following this [link](https://docs.openml.org/Python-start/).
```
data = openml.datasets.get_dataset(dataset_id=31)
X, _, _, _ = data.get_data() # Loaded as Pandas dataframe
```

7. Infer the feature types for the data columns.
```
# Infer the SortingHat feature types.
infer_sh = sortinghatinf.get_sortinghat_types(X)

# Infer the expanded feature types.
infer_exp = sortinghatinf.get_expanded_feature_types(X)

# Infer the ARFF feature types.
# The function `get_feature_types_as_arff()` also returns the SortingHat feature types.
infer_arff, infer_sh = sortinghatinf.get_feature_types_as_arff(X)
```



