import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler


def effect_encoder(data: pd.Series, data_is_transformed=False, delimiter="||"):
    """
    Encodes the effect each component of a list of values, which is mostly 
    similar to a binary encoder.

    In a binary encoder, values are encoded as [0, 1], whereas with the effect
    encoder, values are encoded as [-1, 1]. This encoding strategy stands to 
    benefit neural network convergence for datasets with lots of binary 
    variables, as empirically found in the following resource on neural network
    training (Specifically in the "Why not code binary inputs as 0 and 1?" 
    part): http://www.faqs.org/faqs/ai-faq/neural-nets/part2/

    Args:
        data (pd.Series) : the data
        data_is_transformed (bool) : whether the data is already encoded. If 
            data is boolean only (True, False), this should be set to True.
        delimiter (str) : if data is not transformed, this is the delimiter to
        transform it in case of multiple legs. Otherwise delimiter is ignored.
    Returns:
        If data is not transformed : transformer, df_out
        If data is already transformed : df_out
    """
    if not data_is_transformed:
        mlb = MultiLabelBinarizer()
        data_transformed = mlb.fit_transform(data.apply(
            lambda x: x.split(delimiter)))
        df_out = pd.DataFrame(
            data=data_transformed, 
            columns=mlb.classes_) \
            .replace({0: -1, 1: 1})
        return mlb, df_out
    else:
        return data.astype(int).replace({0: -1, 1: 1})


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    Custom StandardScaler class, which only scales certain columns within the 
    DataFrame. The transform() method returns the transformed DataFrame with
    the original column order.
    """
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

    def fit_transform(self, X, y=None, copy=None):
        self.fit(X, y)
        return self.transform(X, y)
