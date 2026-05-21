"""Small script to carry out analysis of the relationship between 
F1 score scored by a language-specific model on a second language 
and the Distance Measure between the two languages."""

# ------------------------------------------------------------

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression as LiR

# ------------------------------------------------------------

# store Distance Measure in nested dictionary
lang2dist = {
    "eng": {
        "slk": 56.3,
        "eng": 0.0,
        "swe": 31.0,
        "nor": 33.6,
        "heb": 91.4,
        "rom": 57.4,
        "por": 56.5,
        "ger": 31.3,
        "chi": 88.5,
        "hrv": 52.5,
        "srb": 52.5,
        "dan": 24.6
    },
    "slk": {
        "slk": 0.0,
        "eng": 56.3,
        "swe": 42.7,
        "nor": 44.0,
        "heb": 85.0,
        "rom": 49.9,
        "por": 52.5,
        "ger": 50.4,
        "chi": 85.8,
        "hrv": 12.1,
        "srb": 9.3,
        "dan": 54.4
    },
    "dan": {
        "slk": 54.4,
        "eng": 24.6,
        "swe": 10.3,
        "nor": 17.7,
        "heb": 82.2,
        "rom": 54.3,
        "por": 49.4,
        "ger": 28.3,
        "chi": 87.0,
        "hrv": 55.3,
        "srb": 55.3,
        "dan": 0.0
    },
    "rom": {
        "slk": 49.9,
        "eng": 57.4,
        "swe": 47.2,
        "nor": 51.7,
        "heb": 78.4,
        "rom": 0.0,
        "por": 38.0,
        "ger": 52.6,
        "chi": 91.8,
        "hrv": 44.8,
        "srb": 47.1,
        "dan": 54.3
    },
    "chi": {
        "slk": 85.8,
        "eng": 88.5,
        "swe": 86.1,
        "nor": 86.1,
        "heb": 97.2,
        "rom": 91.8,
        "por": 88.6,
        "ger": 87.2,
        "chi": 0.0,
        "hrv": 86.0,
        "srb": 86.0,
        "dan": 87.0
    }
}

# helper function -> retrieve Distance Measure for given language pair
# normalizes values from 0-100 to 0-1
def add_dist(row):
    return lang2dist[row["targ_lang"]][row["test_lang"]] / 100

# helper function -> store the shift in F1 score from baseline to finetuned model
def F1_BLdiff(row):
    return row['F1'] - df_BL[df_BL['test_lang'] == row['test_lang']]['F1'].values[0]

# helper function -> extract p-value for LiR coefficients
def linearRegression(X, y):
    #fit the model and compute the coefficients
    model = LiR()
    model.fit(X, y)
    coeffs = model.coef_
    intercept = model.intercept_
    #now we have coefficients
    
    #generate the design matrix to calculate standard deviations & leverage
    dX = np.column_stack([np.ones(len(X)), X])#add a column of 1s at the start
    #compute RSS and then RSE (will be used for confidence interval calculation)
    yPred = model.predict(X)
    RSS = sum((yPred - y)**2)
    degF = X.shape[0] - X.shape[1] - 1#n - k - 1
    MSE = RSS/(degF)#normalize with degrees of freedom
    tempMat = np.linalg.inv(dX.T @ dX)#temporary matrix used for coefficient std calculation
    coeffVars = MSE*np.diag(tempMat)#compute the coefficient variances using the formula (extract the diagonal)
    coeffSTDs = np.sqrt(coeffVars)
    #now we have the standard deviations of the coefficients
    
    #95% confidence intervals for coefficients and intercept + pValues
    alph = 0.05
    t = intercept/coeffSTDs[0]#compute t-value for intercept
    pVal_intercept = 2*(1.0-stats.t.cdf(abs(t), degF))#1-cdf gives us the probability of getting a value larger, so we multiply by two to get the probability of getting a more extreme value   (also use abs for aways positive values)
    #now for the coefficients
    coeffSTDs = coeffSTDs[1:]#cut of the first element which was for the intercept
    for name, (std, coeff) in enumerate(zip(coeffSTDs, coeffs)):
        t = coeff/std#compute t-value for the coefficient
        pVal_coeff = 2*(1.0-stats.t.cdf(abs(t), degF))#1-cdf gives us the probability of getting a value larger, so we multiply by two to get the probability of getting a more extreme value   (also use abs for aways positive values)
    return pVal_intercept, pVal_coeff

# ------------------------------------------------------------

# load full csv, only retain evaluation of last fine-tuning epoch
df = pd.read_csv("evaluation_merged.csv")
df_15 = df[(df['epoch'] == 15)].copy()

# dataset only containing baseline performance
df_BL = df[(df['targ_lang'] == 'all') & (df['epoch'] == 3)]

# create new column with Distance Measure for the target-test language pair
df_15['dist_measure'] = df_15.apply(add_dist, axis= 1)

# create new column storing the shift in F1 score from baseline to finetuned model
df_15['F1_BLdiff'] = df_15.apply(F1_BLdiff, axis= 1)

# ------------------------------------------------------------

# obtain correlation values (both linear and non-linear)
pearson_corr = df_15[["F1_BLdiff", "dist_measure"]].corr("pearson").values[0][1]
spearman_corr = df_15[["F1_BLdiff", "dist_measure"]].corr("spearman").values[0][1]

# fit a model, obtain coefficients
model = LiR().fit(pd.DataFrame(df_15['dist_measure']), pd.DataFrame(df_15['F1_BLdiff']))
b_0 = model.intercept_
b_1 = model.coef_[0][0]

# obtain statistics: r2 and p-values for coefficients
r2 = model.score(pd.DataFrame(df_15['dist_measure']), pd.DataFrame(df_15['F1_BLdiff']))
pVal_intercept, pVal_coeff = linearRegression(df_15['dist_measure'].to_numpy().reshape(-1, 1), df_15['F1_BLdiff'].to_numpy().reshape(-1, 1))

# ------------------------------------------------------------

#print all results
print("\n\nResults of Language Distance - F1 Metric relationship analysis")
print("\nEach datapoint corresponds to a pair of languages:")
print("| value 1 > the language Distance Measure, normalized to be in range 0-1, where higher values indicate more difference")
print("| value 2 > the difference in F1 score on one language between the model trained on the other language, and the baseline model")
print("\nCorrelation:")
print(f"| spearman = {spearman_corr:.3f}")
print(f"| pearson =  {pearson_corr:.3f}")
print("\nLinear fit:")
print(f"| equation > F1 = {b_0[0]:.4f} + ({b_1:.4f})*DistanceMeasure")
print(f"| R^2 = {r2:.5f}")
print(f"| Intercept P-value =   {pVal_intercept[0]:.5f}")
print(f"| Coefficient P-value = {pVal_coeff[0]:.5f}\n\n")