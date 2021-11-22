# SortingHatLib

SortingHatLib is a library that implements ML-based feature type inference as seen in the paper [here](https://adalabucsd.github.io/papers/2021_SortingHat_SIGMOD.pdf). Feature type inference is the task of predicting the feature types of the columns of a given dataset.

Library for ML feature type inference: https://github.com/pvn25/ML-Data-Prep-Zoo/tree/master/MLFeatureTypeInference.

## Feature Types
### SortingHat
- `Numeric`
- `Categorical`
- `Datetime`
- `Sentence`
- `URL`
- `Embedded Number`
- `List`
- `Not-Generalizable`
- `Context-Specific`

### Extended
Same as SortingHat except:
- `Numeric` mapped to `Integer` or `Floating`
- `Categorical` mapped to `Boolean` if Boolean

### ARFF
- `Integer`
- `Real` (Float)
- `Nominal-specification` (Categorical)
- `String`
- `Ignore`

## Example Usage with OpenML
Here, we run feature type inference on a dataset obtained from OpenML.
Note: this can be done with any dataset loaded as a Pandas dataframe, but we use OpenML here as an example.
1. Install the package using python-pip.
```
git clone https://github.com/pvn25/SortingHatLib.git
pip install SortingHatLib/
```
2. Import the library.
```
import sortinghat.pylib as pl
```

3. Install the OpenML python API.
```
pip install openml
```

4. Import the OpenML python library.
```
import openml
```

5. Load the 'Blood Transfusion Service Center' dataset from OpenML (dataset_id=31).
Note: This requires an OpenML account which you can setup by following this [link](https://docs.openml.org/Python-start/).
```
data = openml.datasets.get_dataset(dataset_id=31)
X, y, cat_ind, attr_names = data.get_data()
```

6. Infer the feature types for the data columns.
```
# Infer the SortingHat feature types.
infer_sh = pl.get_sortinghat_types(X)

# Infer the extended feature types.
infer_ext = pl.get_expanded_feature_types(X)

# Infer the ARFF feature types.
# The `get_feature_types_as_arff()` also returns the SortingHat feature types.
infer_arff, infer_sh = pl.get_feature_types_as_arff(X)
```



