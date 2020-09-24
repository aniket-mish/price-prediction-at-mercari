import joblib
from scipy.sparse import csr_matrix, hstack
import pandas as pd

# Initialize Files
enc_c0 = joblib.load('backend/assets/files/category-1.pickle')
enc_c1 = joblib.load('backend/assets/files/category-2.pickle')
enc_c2 = joblib.load('backend/assets/files/category-3.pickle')

enc_n = joblib.load('backend/assets/files/name.pickle')
enc_t = joblib.load('backend/assets/files/text.pickle')

# Encode features
def get_encodings(data):
    
    category_0 = enc_c0.transform(data['category_0'].values)

    category_1 = enc_c1.transform(data['category_1'].values)

    category_2 = enc_c2.transform(data['category_2'].values)
    
    nums = csr_matrix(pd.get_dummies(data[['shipping', 'item_condition_id', 'is_expensive', 'is_luxurious']], sparse=True).values)

    name = enc_n.transform(data['name'].values)
    
    text = enc_t.transform(data['text'].values)

    data = hstack((category_0, category_1, category_2, nums, name, text)).tocsr().astype('float32')

    return data