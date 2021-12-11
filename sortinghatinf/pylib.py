import pandas as pd
import numpy as np
import os
from pandas.api.types import is_numeric_dtype
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import joblib
from .model_loading import decompress_pickle
from typing import List, Tuple, Union

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources')

rf_Filename = os.path.join(RESOURCES_DIR, "compressed_rf.pbz2")

Pickled_LR_Model = decompress_pickle(filename=rf_Filename)

del_pattern = r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}'
del_reg = re.compile(del_pattern)

delimeters = r"(,|;|\|)"
delimeters = re.compile(delimeters)

url_pat = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
url_reg = re.compile(url_pat)

email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b"
email_reg = re.compile(email_pat)

stop_words = set(stopwords.words('english'))

class_map = {
        0: 'numeric',
        1: 'categorical',
        2: 'datetime',
        3: 'sentence',
        4: 'url',
        5: 'embedded-number',
        6: 'list',
        7: 'not-generalizable',
        8: 'context-specific'
    }

### Private Functions ###

def summary_stats(dat, key_s):
    b_data = []
    for col in key_s:
        nans = np.count_nonzero(pd.isnull(dat[col]))
        dist_val = len(pd.unique(dat[col].dropna()))
        Total_val = len(dat[col])
        mean = 0
        std_dev = 0
        var = 0
        min_val = 0
        max_val = 0
        if is_numeric_dtype(dat[col]):
            mean = np.mean(dat[col])
            
            if pd.isnull(mean):
                mean = 0
                std_dev = 0
                #var = 0
                min_val = 0
                max_val = 0           
            else:
                data_col = dat[col]
                if pd.api.types.is_sparse(data_col):
                    data_col = data_col.sparse.to_dense()
                std_dev = np.std(data_col)
                var = np.var(data_col)
                min_val = float(np.min(data_col))
                max_val = float(np.max(data_col))
        b_data.append([Total_val, nans, dist_val, mean, std_dev, min_val, max_val])
    return b_data

def get_sample(dat, key_s):
    rand = []
    for name in key_s:
        rand_sample = list(pd.unique(dat[name]))
        rand_sample = rand_sample[:5]
        while len(rand_sample) < 5:
            rand_sample.append(list(pd.unique(dat[name]))[np.random.randint(len(list(pd.unique(dat[name]))))])
        rand.append(rand_sample[:5])
    return rand


# summary_stat_result has a structure like [[Total_val, nans, dist_va, ...], ...].
def get_ratio_dist_val(summary_stat_result):
    ratio_dist_val = []
    for r in summary_stat_result:
        ratio_dist_val.append(r[2]*100.0 / r[0])
    return ratio_dist_val

def get_ratio_nans(summary_stat_result):
    ratio_nans = []
    for r in summary_stat_result:
        ratio_nans.append(r[1]*100.0 / r[0])
    return ratio_nans

def featurize_and_extract(df):
    """
    Takes a dataframe and runs base featurization on it.
    Then runs feature extraction on the featurized data.
    Returns the featurized data.
    """
    dataFeaturized = featurize_file(df)
    dataFeaturized1 = feature_extraction(dataFeaturized)
    return dataFeaturized1

def featurize_file(df):
    stats = []
    attribute_name = []
    sample = []
    id_value = []
    i = 0

    castability = []
    number_extraction = []

    avg_tokens = []
    ratio_dist_val = []
    ratio_nans = []

    keys = list(df.keys())
    attribute_name.extend(keys)
    summary_stat_result = summary_stats(df, keys)
    stats.extend(summary_stat_result)
    samples = get_sample(df,keys)
    sample.extend(samples)

    ratio_dist_val.extend(get_ratio_dist_val(summary_stat_result))
    ratio_nans.extend(get_ratio_nans(summary_stat_result))


    csv_names = ['Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val', 'mean', 'std_dev', 'min_val',
                 'max_val', '%_dist_val', '%_nans', 'sample_1', 'sample_2', 'sample_3','sample_4','sample_5'
                ]
    golden_data = pd.DataFrame(columns = csv_names)
    
    for i in range(len(attribute_name)):
        val_append = []
        val_append.append(attribute_name[i])
        val_append.extend(stats[i])
        
        val_append.append(ratio_dist_val[i])
        val_append.append(ratio_nans[i])    
        
        val_append.extend(sample[i])

        golden_data.loc[i] = val_append

    curdf = golden_data
    
    for row in curdf.itertuples():
        is_list = False
        curlst = [row[11],row[12],row[13],row[14],row[15]]
        
        delim_cnt,url_cnt,email_cnt,date_cnt =0,0,0,0
        chars_totals,word_totals,stopwords,whitespaces,delims_count = [],[],[],[],[]
        
        for value in curlst: # sample values
            word_totals.append(len(str(value).split(' ')))
            chars_totals.append(len(str(value)))
            whitespaces.append(str(value).count(' '))
            
            if del_reg.match(str(value)):  delim_cnt += 1    
            if url_reg.match(str(value)):  url_cnt += 1
            if email_reg.match(str(value)):  email_cnt += 1
            
            delims_count.append(len(delimeters.findall(str(value))))        
            
            tokenized = word_tokenize(str(value))
            stopwords.append(len([w for w in tokenized if w in stop_words]))    
        
            try:
                _ = pd.Timestamp(value)
                date_cnt += 1
            # except ValueError: date_cnt += 0
            except: 
                date_cnt += 0    
        
        if delim_cnt > 2:  curdf.at[row.Index, 'has_delimiters'] = True
        else: curdf.at[row.Index, 'has_delimiters'] = False

        if url_cnt > 2:  curdf.at[row.Index, 'has_url'] = True
        else: curdf.at[row.Index, 'has_url'] = False
            
        if email_cnt > 2:  curdf.at[row.Index, 'has_email'] = True
        else: curdf.at[row.Index, 'has_email'] = False   
            
        if date_cnt > 2:  curdf.at[row.Index, 'has_date'] = True
        else: curdf.at[row.Index, 'has_date'] = False           
            
        curdf.at[row.Index, 'mean_word_count'] = np.mean(word_totals)
        curdf.at[row.Index, 'std_dev_word_count'] = np.std(word_totals)
        
        curdf.at[row.Index, 'mean_stopword_total'] = np.mean(stopwords)
        curdf.at[row.Index, 'stdev_stopword_total'] = np.std(stopwords)
        
        curdf.at[row.Index, 'mean_char_count'] = np.mean(chars_totals)    
        curdf.at[row.Index, 'stdev_char_count'] = np.std(chars_totals)
        
        curdf.at[row.Index, 'mean_whitespace_count'] = np.mean(whitespaces)
        curdf.at[row.Index, 'stdev_whitespace_count'] = np.std(whitespaces)    
        
        curdf.at[row.Index, 'mean_delim_count'] = np.mean(whitespaces)
        curdf.at[row.Index, 'stdev_delim_count'] = np.std(whitespaces)      
        
        if curdf.at[row.Index, 'has_delimiters'] and curdf.at[row.Index, 'mean_char_count'] < 100: curdf.at[row.Index, 'is_list'] = True    
        else: curdf.at[row.Index, 'is_list'] = False
        
        if curdf.at[row.Index, 'mean_word_count'] > 10: curdf.at[row.Index, 'is_long_sentence'] = True    
        else: curdf.at[row.Index, 'is_long_sentence'] = False    

    golden_data = curdf

    return golden_data	

vectorizerName = joblib.load(os.path.join(RESOURCES_DIR, "dictionaryName.pkl"))
vectorizerSample = joblib.load(os.path.join(RESOURCES_DIR, "dictionarySample.pkl"))

def process_stats(data):
    data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val','has_delimiters', 'has_url', 'has_email', 'has_date', 'mean_word_count',
       'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
       'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
       'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
       'is_list', 'is_long_sentence']] # drops Attribute_name and sample values
    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)
    return data1

def feature_extraction(data):
    data1 = process_stats(data)

    arr = data['Attribute_name'].values
    arr = [str(x) for x in arr]
    
    arr1 = data['sample_1'].values
    arr1 = [str(x) for x in arr1] # One row of values
    arr2 = data['sample_2'].values
    arr2 = [str(x) for x in arr2]
    arr3 = data['sample_3'].values
    arr3 = [str(x) for x in arr3]    

    X = vectorizerName.transform(arr)
    X1 = vectorizerSample.transform(arr1)
    X2 = vectorizerSample.transform(arr2)

    attr_df = pd.DataFrame(X.toarray())
    sample1_df = pd.DataFrame(X1.toarray())
    sample2_df = pd.DataFrame(X2.toarray())
    
    data2 = pd.concat([data1, attr_df], axis=1, sort=False)
    return data2


def load_RF(df):
    """
    Runs the Random Forest Classifier on the given data.
    """
    y_RF = Pickled_LR_Model.predict(df).tolist()
    return y_RF

### Public Functions ###

def get_sortinghat_types(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of the predicted SortingHat feature types on the columns of the specified Pandas dataframe 
    """
    dataFeaturized = featurize_and_extract(df)
    y_RF = load_RF(dataFeaturized) # Prediction
    return [class_map[y] for y in y_RF]

def get_expanded_feature_types(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of the predicted SortingHat feature types 
    on the columns of the specified Pandas dataframe 
    mapped to the expanded types 
    """
    sortinghat_types = get_sortinghat_types(df)

    expanded_types = []
    for i in range(len(sortinghat_types)):
        sortinghat_type = sortinghat_types[i]

        column_dtype = sortinghat_type
        if sortinghat_type == 'numeric':
            if pd.api.types.is_integer_dtype(df.iloc[:,i]):
                column_dtype = 'integer'
            else:
                column_dtype = 'floating'
        if sortinghat_type == 'categorical':
            if pd.api.types.is_bool_dtype(df.iloc[:,i]):
                column_dtype = 'boolean'
        
        expanded_types.append(column_dtype)
    return expanded_types


def get_feature_types_as_arff(df: pd.DataFrame) -> Tuple[List[Tuple[str, Union[str, List[str]]]], List[str]]:
    """
    Returns the predicted SortingHat feature types mapped to the loose ARFF types 
    and the original predicted SortingHat feature types  
    """
    sortinghat_types = get_sortinghat_types(df)

    attributes_arff = []
    # Collects categorical columns that need to be cast to string
    non_str_categoricals = []
    for i in range(len(sortinghat_types)):
        sortinghat_type = sortinghat_types[i]
        # Map Sortinghat feature types to arff
        column_name = str(df.iloc[:,i].name)
        if sortinghat_type == 'numeric':
            if pd.api.types.is_integer_dtype(df.iloc[:,i]):
                attributes_arff.append((column_name, 'INTEGER'))
            else:
                attributes_arff.append((column_name, 'REAL'))
        elif sortinghat_type == 'categorical':
            # for categorical feature, arff expects a list string. However, a
            # categorical column can contain mixed type and should therefore
            # raise an error asking to convert all entries to string.
            categories = df.iloc[:,i].astype('category').cat.categories
            categories_dtype = pd.api.types.infer_dtype(categories)
            if categories_dtype not in ("string", "unicode"):
                non_str_categoricals.append((column_name, categories_dtype))
            attributes_arff.append((column_name, categories.tolist()))
        elif sortinghat_type in ('datetime', 'sentence', 'url', 'embedded-number', 'list', 'context-specific'):
            attributes_arff.append((column_name, 'STRING'))
        elif sortinghat_type == 'not-generalizable':
            attributes_arff.append((column_name, 'IGNORE'))
        else:
            raise ValueError(
                "The column {} is of an unrecognized type {}.".format(column_name, sortinghat_type)
            )

    # Detected categorical columns that need to be cast to string
    if len(non_str_categoricals) > 0:
        raise ValueError(
                    "The column(s) {} of the dataframe were detected "
                    "to be categorical. Therefore, we require all values in "
                    "these columns to be string. Please "
                    "convert the entries which are not string. "
                    "Detected dtypes {} for these columns respectively.".format(
                        [c[0] for c in non_str_categoricals], 
                        [c[1] for c in non_str_categoricals])
                )

    return attributes_arff, sortinghat_types
