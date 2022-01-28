### Name: Yumika Shiba
### Student ID: 260863694

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap



# Part 1: Implementing collapsed Gibbs sampling LDA

# Reading in the data
f_mimic3 = "MIMIC3_DIAGNOSES_ICD_subset.csv"
f_d_icd = "D_ICD_DIAGNOSES_DATA_TABLE.csv"

df_m3 = pd.read_csv(f_mimic3, converters={'ICD9_CODE': lambda x: str(x)})
df_d = pd.read_csv(f_d_icd, converters={'ICD9_CODE': lambda x: str(x)})

D = 689  # # of unique patients
M = 389  # # of unique ICD-9 codes
K = 5
alpha = 1
beta = 0.001

df_m3['SUBJECT_ID'] = df_m3['SUBJECT_ID'].astype(str)
df_m3['ICD9_CODE'] = df_m3['ICD9_CODE'].astype(str)


SUBJECT_IDs = df_m3["SUBJECT_ID"].unique()
ICD9_CODEs = df_m3["ICD9_CODE"].unique()

df_sj = pd.DataFrame(SUBJECT_IDs, columns = ['SUBJECT_ID'])
df_ICD = pd.DataFrame(ICD9_CODEs, columns = ['ICD9_CODE'])
df_M3 = df_m3.copy()
for k in range(K):
    topic_no = str(k+1)
    df_sj[topic_no] = 0.0
    df_ICD[topic_no] = 0.0
    df_M3[topic_no] = 0.0

    

for itr in range(100):  ## Change to 100
    print("iteration: ", itr)
    
    # Enumerate over all of 1000 tokens
    for index, row in df_M3.iterrows():
        d = row['SUBJECT_ID']
        i = row['ICD9_CODE']
        
        
        # update K x 1 topic distribution of z_id
        gamma_idk = np.zeros(K)  # K x 1 vector - 5 x 1 vector
        
        idx_sj = df_sj.index[df_sj['SUBJECT_ID']==d].tolist()[0]
        idx_ICD = df_ICD.index[df_ICD['ICD9_CODE']==i].tolist()[0]
        
        
        # for each of the 5 topics
        for k in range(K): 
            topic_no = str(k+1)
                
            # count # of words that had been assigned to k 
            denom_subtract = df_M3[df_M3["ICD9_CODE"].str.contains(i)][topic_no].sum()
            
            if df_M3.at[index,topic_no] == 1:
                df_sj.at[idx_sj, topic_no] -= 1
                df_ICD.at[idx_ICD, topic_no] -= 1
            
            term1 = alpha + (df_sj.at[idx_sj, topic_no])
            term2 = (beta + (df_ICD.at[idx_ICD, topic_no])) / (M * beta + (df_ICD[topic_no].sum() - denom_subtract))
            gamma_idk[k] = term1 * term2
                
        
        # Normalize so that sums to 1
        gamma_idk = gamma_idk / gamma_idk.sum()
        
        # Sample a topic, z_id
        z_id = np.random.multinomial(1, gamma_idk, size=1)
        
        # Update the record of previously sampled topic for each token
        for k in range(K):
            topic_no = str(k+1)
            df_M3.at[index, topic_no] = z_id[0][k]

        # Update the sufficient statistics        
        for k in range(K):
            topic_no = str(k+1)

            if z_id[0][k] == 1:
                df_sj.at[idx_sj, topic_no] += 1
                df_ICD.at[idx_ICD, topic_no] += 1



# Create and Initialize the two matrices
THETA = pd.DataFrame(SUBJECT_IDs, columns = ['SUBJECT_ID'])
PHI = pd.DataFrame(ICD9_CODEs, columns = ['ICD9_CODE'])
ICD9_CODEs = df_m3["ICD9_CODE"].unique()

for k in range(K):
    topic_no = str(k+1)
    PHI[topic_no] = 0.0
    THETA[topic_no] = 0.0

    
    
# Q2. Visualizing the top ICD codes under each topic

# normalization 1
for k in range(K):
    topic_no = str(k+1)
    
    sum_w_nwk = df_ICD[topic_no].sum()  ### May be wrong
    
    for word in ICD9_CODEs:
        idx_ICD = df_ICD.index[df_ICD['ICD9_CODE']==word].tolist()[0]
        PHI.at[idx_ICD, topic_no] = (beta + df_ICD.at[idx_ICD, topic_no]) / ((M * beta) + sum_w_nwk)

        
# Join the ICD9_CODE and short name
PHI_name = pd.merge(PHI, df_d, on="ICD9_CODE", how='left')
PHI_name["ICD9_SHORT"] = PHI_name["ICD9_CODE"] + "-" + PHI_name["SHORT_TITLE"]
PHI_ICD9_SHORT = PHI_name[["ICD9_SHORT", "1", "2", "3", "4","5"]]


result = pd.DataFrame() # Initialize a dataframe

for i in range(5):
    topic_no = str(i+1)
    
    df_top10 = PHI_ICD9_SHORT.nlargest(10, topic_no)
    result = result.append(df_top10, ignore_index=True, sort=False)

result = result.fillna(0)
result = result.set_index('ICD9_SHORT')

# Plotting the heatmap
fig, ax = plt.subplots(figsize=(10, 30))
ax = heatmap(result, cmap="Reds", linecolor="white")
          

    
        
# Q3: Correlating topics with the target ICD codes

# normalization 2
for doc in SUBJECT_IDs:
    sum_n_dk = df_sj.loc[df_sj["SUBJECT_ID"]==doc].sum(axis=1)
    idx_sj = df_sj.index[df_sj['SUBJECT_ID']==doc].tolist()[0]
    
    for k in range(K):
        topic_no = str(k+1)
        THETA.at[idx_sj, topic_no] = (alpha + df_sj.at[idx_sj, topic_no]) / ((K * alpha) + sum_n_dk) 

THETA_sorted = THETA.sort_values(by=["SUBJECT_ID"])
df_m3_2 = df_m3.copy()
df_m3_2["331"] = np.where(df_m3_2["ICD9_CODE"].str.startswith("331"), 1.0, 0.0)
df_m3_2["332"] = np.where(df_m3_2["ICD9_CODE"].str.startswith("332"), 1.0, 0.0)
df_m3_2["340"] = np.where(df_m3_2["ICD9_CODE"].str.startswith("340"), 1.0, 0.0)
df_grouped = df_m3_2.groupby("SUBJECT_ID").sum()
df_grouped_sorted = df_grouped.sort_values(by=["SUBJECT_ID"])

df_Q3 = THETA_sorted.merge(df_grouped_sorted, on="SUBJECT_ID")

corr_matrix_np = np.zeros((5,4))
df_corr = pd.DataFrame(corr_matrix_np, columns=["topic", "331", "332", "340"])
df_corr["topic"] = df_corr["topic"].astype(str)

for k in range(K):
    topic_no = str(k+1)
    df_corr.at[k, "topic"] = "topic"+ str(topic_no)

lst_icd = ["331", "332", "340"]
lst_topics = [0,1,2,3,4]

for icd in lst_icd:
    for topic in lst_topics:
        df_corr.at[topic, icd] = df_Q3[icd].corr(df_Q3[str(topic+1)])

df_corr = df_corr.set_index('topic')

# Plotting the heatmap
fig_Q3, ax_Q3 = plt.subplots(figsize=(5, 3))
ax_Q3 = heatmap(df_corr, cmap="Reds", linecolor="white")



# Q4
df_m3_2 = df_M3.copy()
df_m3_2["331"] = np.where(df_m3_2["ICD9_CODE"].str.startswith("331"), 1.0, 0.0)
df_m3_2["332"] = np.where(df_m3_2["ICD9_CODE"].str.startswith("332"), 1.0, 0.0)
df_m3_2["340"] = np.where(df_m3_2["ICD9_CODE"].str.startswith("340"), 1.0, 0.0)
df_grouped = df_m3_2.groupby("SUBJECT_ID").sum()
df_grouped_3ICD = df_grouped[["331", "332", "340"]]


result2 = pd.DataFrame() # Initialize a dataframe

for i in range(5):
    topic_no = str(i+1)    
    df_top100 = THETA.nlargest(100, topic_no)
    result2 = result2.append(df_top100, ignore_index=True, sort=False)

result2 = result2.fillna(0)
result2 = result2.merge(df_grouped_3ICD, on="SUBJECT_ID", how="left")

result2_topics = result2[["SUBJECT_ID", "1", "2", "3", "4", "5"]]

result2_icds = result2[["SUBJECT_ID", "331", "332", "340"]]

result2_topics = result2_topics.set_index('SUBJECT_ID')
result2_icds = result2_icds.set_index('SUBJECT_ID')


# Plotting the heatmap
fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 30))
heatmap(result2_icds, cmap="Reds", linecolor="white", ax=ax1)
heatmap(result2_topics, cmap="Reds", linecolor="white", ax=ax2)