import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
from itertools import combinations
import matplotlib.pyplot as plt


### Pre-step: Reading in z score and LD matrix ###
# Files to be read in
fIn_z = "zscore.csv"
fIn_LD = "LD.csv"

# Read in the data into pandas dataframes
df_z = pd.read_csv(fIn_z)
df_LD = pd.read_csv(fIn_LD)

# Rename the first columns
df_LD.rename(columns={'Unnamed: 0':'SNP_name'}, inplace=True)
df_z.rename(columns={'Unnamed: 0':'SNP_name'}, inplace=True)



### Part 1 - 3: The three parts are combined in the code below


## Part 1: Implement the efficient Bayes factor

snps = df_z["SNP_name"].tolist()
comb1 = combinations(snps, 1)
comb2 = combinations(snps, 2)
comb3 = combinations(snps, 3)

comb1_list = []
for i in list(comb1):
    comb1_list.append(i)
colnames1 = ["SNP1"]
df_comb1 = pd.DataFrame(comb1_list, columns=colnames1)

comb2_list = []
for i in list(comb2):
    comb2_list.append(i)
colnames2 = ["SNP1", "SNP2"]
df_comb2 = pd.DataFrame(comb2_list, columns=colnames2)

comb3_list = []
for i in list(comb3):
    comb3_list.append(i)
colnames3 = ["SNP1", "SNP2", "SNP3"]
df_comb3 = pd.DataFrame(comb3_list, columns=colnames3)


## For 1 SNP
# Iterate over each row
for index, rows in df_comb1.iterrows():
    # Create list for the current row
    one_SNP = [rows.SNP1]
    colnam1 = one_SNP.copy()
    colnam1.insert(0, "SNP_name")
    
    df_LD1 = df_LD[colnam1]
    
    df_LD1 = df_LD1[df_LD1['SNP_name'].isin(one_SNP)]
    df_z1 = df_z[df_z['SNP_name'].isin(one_SNP)]
    
    # Convert the dataframes to numpy arrays
    z_c1 = df_z1["V1"].to_numpy()
    R_cc1 = df_LD1.to_numpy()[:, 1:]
    
    # Initialize variables based on given information
    cov_CC1 = 2.49 * np.identity(1)
    zeros1 = np.zeros(1)
    
    # Numerator
    cov_num1 = R_cc1 + np.matmul(np.matmul(R_cc1, cov_CC1), R_cc1)    
    numerator1 = multivariate_normal.pdf(z_c1, mean=zeros1, cov=cov_num1)
    denominator1 = multivariate_normal.pdf(z_c1, mean=zeros1, cov=R_cc1)

    BF1 = numerator1 / denominator1
    df_comb1.loc[index, "BF3"] = BF1
    df_comb1.loc[index, "SNP2"] = "HELLO2"
    df_comb1.loc[index, "SNP3"] = "HELLO3"
    
df_comb1=df_comb1.reindex(columns= ['SNP1', 'SNP2', 'SNP3','BF3'])


## For 2 SNPs
# Iterate over each row
for index, rows in df_comb2.iterrows():
    # Create list for the current row
    two_SNPs =[rows.SNP1, rows.SNP2]
    colnam2 = two_SNPs.copy()
    colnam2.insert(0, "SNP_name")
    
    df_LD2 = df_LD[colnam2]
    
    df_LD2 = df_LD2[df_LD2['SNP_name'].isin(two_SNPs)]
    df_z2 = df_z[df_z['SNP_name'].isin(two_SNPs)]
    
    # Convert the dataframes to numpy arrays
    z_c2 = df_z2["V1"].to_numpy()
    R_cc2 = df_LD2.to_numpy()[:, 1:]
    
    # Initialize variables based on given information
    cov_CC2 = 2.49 * np.identity(2)
    zeros2 = np.zeros(2)
    
    cov_num2 = R_cc2 + np.matmul(np.matmul(R_cc2, cov_CC2), R_cc2)
    cov_num2 = cov_num2.astype('float64')
    if np.linalg.det(cov_num2) == 0:
        df_comb2.loc[index, "BF3"] = 1111111111111111111111111111
    else:
        numerator2 = multivariate_normal.pdf(z_c2, mean=zeros2, cov=cov_num2)
        denominator2 = multivariate_normal.pdf(z_c2, mean=zeros2, cov=R_cc2)

        BF2 = numerator2 / denominator2
        df_comb2.loc[index, "BF3"] = BF2
        df_comb2.loc[index, "SNP3"] = "HELLO3"

df_comb2=df_comb2.reindex(columns= ['SNP1', 'SNP2', 'SNP3','BF3'])
df_comb2 = df_comb2[df_comb2.BF3 != 1111111111111111111111111111]

## For 3 SNPs
# Iterate over each row
for index, rows in df_comb3.iterrows():
    # print(index)
    # Create list for the current row
    three_SNPs =[rows.SNP1, rows.SNP2, rows.SNP3]
    colnam3 = three_SNPs.copy()
    colnam3.insert(0, "SNP_name")

    df_LD3 = df_LD[colnam3]
    
    df_LD3 = df_LD3[df_LD3['SNP_name'].isin(three_SNPs)]
    df_z3 = df_z[df_z['SNP_name'].isin(three_SNPs)]

    # Convert the dataframes to numpy arrays
    z_c3 = df_z3["V1"].to_numpy()
    R_cc3 = df_LD3.to_numpy()[:, 1:]
    
    # Initialize variables based on given information
    cov_CC3 = 2.49 * np.identity(3)
    zeros3 = np.zeros(3)
    
    cov_num3 = R_cc3 + np.matmul(np.matmul(R_cc3, cov_CC3), R_cc3)
    cov_num3 = cov_num3.astype('float64')
    if np.linalg.det(cov_num3) == 0 or np.linalg.det(cov_num3) < 10**(-7):
        df_comb3.loc[index, "BF3"] = 1111111111111111111111111111
    else:
        numerator3 = multivariate_normal.pdf(z_c3, mean=zeros3, cov=cov_num3)
        denominator3 = multivariate_normal.pdf(z_c3, mean=zeros3, cov=R_cc3)
        BF3 = numerator3 / denominator3
        df_comb3.loc[index, "BF3"] = BF3


# Part 2: Implement prior calculation
m = 100
df_comb1["prior"] = ((1/m)**1) * ((m-1)/m)**(m-1)
df_comb2["prior"] = ((1/m)**2) * ((m-1)/m)**(m-2)
df_comb3["prior"] = ((1/m)**3) * ((m-1)/m)**(m-3)


df_comb1['posterior'] = df_comb1['BF3'] * df_comb1['prior']
df_comb2['posterior'] = df_comb2['BF3'] * df_comb2['prior']
df_comb3['posterior'] = df_comb3['BF3'] * df_comb3['prior']

frames = [df_comb1, df_comb2, df_comb3]
df_merged = pd.concat(frames)

# Part 3: Implement posterior inference
df_merged["posterior_normalized"] = df_merged["posterior"] / df_merged["posterior"].sum()
df_merged = df_merged.sort_values(by=['posterior_normalized'], ascending=False)

df_merged['sorted_configurations'] = df_merged.index

# Generating plot for Part 3
df_merged.plot.scatter(x="sorted_configurations", y="posterior_normalized", alpha=0.5)



### Part 4: Implement PIP & Visualize

snp_pip = {}
sum_snp_scores = 0
sum_all_scores = df_merged["posterior"].sum()

for snp in snps:
    # Extract rows containing "the" SNP
    df_filtered = df_merged[(df_merged == snp).any(axis=1)]
    # Sum the scores associated with that SNP
    sum_snp_scores = df_filtered["posterior"].sum()
    
    pip = sum_snp_scores / sum_all_scores
    
    # Add the result to a dictionary
    snp_pip[snp] = [pip]


df_result = pd.DataFrame(snp_pip)
df_result = df_result.T
df_result = df_result.reset_index()
df_result.columns = ['SNP_name', 'PIP']
df_result.to_csv("COMP565_A2_SNP_pip.csv.gz")

# Calculate -log10p value from z-scores
df_z["pVal"] = scipy.stats.norm.sf(abs(df_z["V1"]))
df_z["-log10p"] = - np.log10(df_z["pVal"])

# Merge the dataframes (PIP and -log10p) by SNP name.
# This ensures that the two plots that will be generated will have the
# same x-axis (the order in which SNPs are plotted are the same)
df_merged_p_pip = pd.merge(df_z, df_result, on=['SNP_name'])


# Plotting
fig, axes = plt.subplots(2,1)
df_merged_p_pip.plot.scatter(x="SNP_name", y="-log10p", alpha=0.5, ax=axes[0])
df_merged_p_pip.plot.scatter(x="SNP_name", y="PIP", alpha=0.5, ax=axes[1])