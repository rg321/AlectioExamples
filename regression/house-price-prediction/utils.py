# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:59:21 2020

@author: arun
"""


def preprocess(trainDF, testDF):
    trainDF.drop(["Id"], axis=1, inplace=True)
    testDF.drop(["Id"], axis=1, inplace=True)
    trainDF = trainDF[trainDF.GrLivArea < 4500]
    trainDF.reset_index(drop=True, inplace=True)
    trainDF["SalePrice"] = np.log1p(trainDF["SalePrice"])
    y = trainDF["SalePrice"].reset_index(drop=True)
    train_features = trainDF.drop(["SalePrice"], axis=1)
    test_features = testDF
    features = pd.concat([train_features, test_features]).reset_index(drop=True)

    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna("None"))

    features["LotFrontage"] = features.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numerics = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics.append(i)
    features.update(features[numerics].fillna(0))

    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numerics2 = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics2.append(i)
    skew_features = (
        features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
    )

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))

    features = features.drop(["Utilities", "Street", "PoolQC",], axis=1)

    features["YrBltAndRemod"] = features["YearBuilt"] + features["YearRemodAdd"]
    features["TotalSF"] = (
        features["TotalBsmtSF"] + features["1stFlrSF"] + features["2ndFlrSF"]
    )

    features["Total_sqr_footage"] = (
        features["BsmtFinSF1"]
        + features["BsmtFinSF2"]
        + features["1stFlrSF"]
        + features["2ndFlrSF"]
    )

    features["Total_Bathrooms"] = (
        features["FullBath"]
        + (0.5 * features["HalfBath"])
        + features["BsmtFullBath"]
        + (0.5 * features["BsmtHalfBath"])
    )

    features["Total_porch_sf"] = (
        features["OpenPorchSF"]
        + features["3SsnPorch"]
        + features["EnclosedPorch"]
        + features["ScreenPorch"]
        + features["WoodDeckSF"]
    )

    features["haspool"] = features["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    features["has2ndfloor"] = features["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
    features["hasgarage"] = features["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
    features["hasbsmt"] = features["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    features["hasfireplace"] = features["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)
    final_features = pd.get_dummies(features).reset_index(drop=True)

    X = final_features.iloc[: len(y), :]
    X_sub = final_features.iloc[len(y) :, :]
    X.shape, y.shape, X_sub.shape

    outliers = [30, 88, 462, 631, 1322]
    X = X.drop(X.index[outliers])
    y = y.drop(y.index[outliers])

    overfit = []
    for i in X.columns:
        counts = X[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(X) * 100 > 99.94:
            overfit.append(i)

    overfit = list(overfit)
    X = X.drop(overfit, axis=1)
    X_sub = X_sub.drop(overfit, axis=1)

    return X, y, X_sub
