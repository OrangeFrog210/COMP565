import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


fIn_beta_marginal = "beta_marginal.csv"
fIn_LD = "LD.csv"

# Read in the data into pandas dataframes
df_beta_marginal = pd.read_csv(fIn_beta_marginal)
df_LD = pd.read_csv(fIn_LD)

# Rename the first columns
df_beta_marginal.rename(columns={'Unnamed: 0':'SNP_name'}, inplace=True)
df_LD.rename(columns={'Unnamed: 0':'SNP_name'}, inplace=True)

# initialize values
M = 100  # number of SNPs
N_train = 439
N_test = 50

# Create a list of the SNPs (100)
snp_lst = df_beta_marginal["SNP_name"].to_list()
print(len(snp_lst))


# initialize values for the hyperparameters
tao_e = 1
tao_beta = 200
pi = 0.01


df_LD_i = df_LD.copy()

# initialize values for the posterior estimates for all SNPs
df_LD_i["myu_"] = 0.0
df_LD_i["tao_"] = 1.0
df_LD_i["gamma_"] = 0.01


result_lst = []
for ite in range(10): # for 10 iterations
    print("iteration: ", ite)
    
    for snp_j in snp_lst:
        
        # For updating
        idx = df_LD_i[df_LD_i["SNP_name"] == snp_j].index[0]
        
        
        ## 1. E-step
        
        # Calculate tao_j
        tao_j = (N_train*tao_e) + tao_beta
        
        # 1. Update tao_j
        df_LD_i.at[idx,"tao_"] = tao_j        
        
        # Calculate myu_j
        beta_marginal_j_df = df_beta_marginal[df_beta_marginal["SNP_name"] == snp_j]
        beta_marginal_j = beta_marginal_j_df["V1"].values[0]
        
        df_sum = df_LD_i.copy()
        df_sum = df_sum[["SNP_name", snp_j, "gamma_", "myu_", "tao_"]]
        df_sum_i = df_sum[df_sum["SNP_name"] != snp_j]  # A datafram not containing a row for SNP j
        df_sum_i["gamma_myu_rij"] = df_sum_i["gamma_"] * df_sum_i["myu_"] * df_sum_i[snp_j]
        sum_i = df_sum_i["gamma_myu_rij"].sum()

        myu_j = N_train * (tao_e / tao_j) * (beta_marginal_j - sum_i)  ## unsure: beta_marginal_j
        
        # 2. Update myu_j
        df_LD_i.at[idx,"myu_"] = myu_j


        # Calculate gamma_j
        u_j = math.log(pi / (1-pi)) + (1/2)*(math.log(tao_beta/tao_j)) + (tao_j/2)*(myu_j**2)
        gamma_j = 1 / (1 + math.exp(-u_j))

        # 3. Update gamma_j
        df_LD_i.at[idx,"gamma_"] = gamma_j


    df_sum_E = df_sum.copy()
    df_sum_E.loc[(df_sum_E.gamma_ < 0.01), ("gamma_")] = 0.01
    df_sum_E.loc[(df_sum_E.gamma_ > 0.99), ("gamma_")] = 0.99
    
    
    ## 2. Maximization Step
    # Update tao_beta_inverse
    df_sum_E["term1"] = df_sum_E["gamma_"] * (df_sum_E["myu_"]**2 + df_sum_E["tao_"]**(-1))
    tao_beta = (df_sum_E["term1"].sum() / df_sum_E["gamma_"].sum())**(-1)
    
    # Update pi
    pi = df_sum_E["gamma_"].sum() / M
    arr_beta_marginal = df_beta_marginal["V1"].to_numpy()
    R = df_LD.to_numpy()[:, 1:]
    
    # Calculate 1st ELBO term
    tm1 = (N_train/2)*(math.log(tao_e))  # since tao_e=1 always in our case, tm1=0 always

    tm2 = -(tao_e/2) * N_train

    df_sum_E["elbo1_term3_temp1"] = df_sum_E["gamma_"]*df_sum_E["myu_"]
    gam_myu_elmwise = df_sum_E["elbo1_term3_temp1"].to_numpy()
    tm3 = tao_e * np.matmul(np.transpose(gam_myu_elmwise), N_train*arr_beta_marginal)

    df_sum_E["elbo1_term4"] = (-tao_e/2)*(df_sum_E["gamma_"])*(df_sum_E["myu_"]**2 + (1/df_sum_E["tao_"]))*N_train
    tm4 = df_sum_E["elbo1_term4"].sum()
    
    # Calculate 5th term in 1st ELBO term
    comb2 = combinations(snp_lst, 2)
    comb2_list = []
    for i in list(comb2):
        comb2_list.append(i)
    colnames2 = ["SNP1", "SNP2"]
    df_comb2 = pd.DataFrame(comb2_list, columns=colnames2)

    for index, rows in df_comb2.iterrows():
        # Create list for the current row
        two_SNPs =[rows.SNP1, rows.SNP2]
        colnam2 = two_SNPs.copy()
        colnam2.insert(0, "SNP_name")

        df_LD2 = df_LD[colnam2]

        df_LD2 = df_LD2[df_LD2['SNP_name'].isin(two_SNPs)]
        df_LD2 = df_LD2[df_LD2["SNP_name"] == two_SNPs[0]]
        rkj = df_LD2[two_SNPs[1]].values[0]


        idx0 = df_LD_i[df_LD_i["SNP_name"] == two_SNPs[0]].index[0]
        idx1 = df_LD_i[df_LD_i["SNP_name"] == two_SNPs[1]].index[0]

        # 1. Update tao_j
        df_LD_i.at[idx,"tao_"] = tao_j

        gm_j = df_sum_E.at[idx0,"gamma_"]
        my_j = df_sum_E.at[idx0,"myu_"]
        gm_k = df_sum_E.at[idx1,"gamma_"]
        my_k = df_sum_E.at[idx1,"myu_"]
        df_comb2.loc[index, "calc"] = gm_j*my_j*gm_k*my_k*(N_train*rkj)

    tm5_5_sum = df_comb2["calc"].sum()
    tm5 = - tao_e * (tm5_5_sum)
    
    elbo1 = tm1 + tm2 + tm3 + tm4 + tm5
    

    # Calculate 2nd ELBO term - more precisely, it's not the entire term. Omitting part that cancels out with 4th ELBO term.
    df_sum_E["elbo2_term2"] = -(tao_beta/2)*df_sum_E["gamma_"]*(df_sum_E["myu_"]**2 + df_sum_E["tao_"]**(-1))
    elbo2_2 = df_sum_E["elbo2_term2"].sum()
    elbo2 =  elbo2_2

    # Calculate 3rd ELBO term
    df_sum_E["elbo3_term"] = (1-df_sum_E["gamma_"])*(np.log(1-pi)) + df_sum_E["gamma_"]*(np.log(pi))
    elbo3 = df_sum_E["elbo3_term"].sum()

    # Calculate 4th ELBO term - more precisely, it's not the entire term. Omitting part that cancels out with 2nd ELBO term.
    df_sum_E["elbo4_term_2"] = (-1/2)*(df_sum_E["gamma_"])*(math.log(tao_beta))
    elbo4 = df_sum_E["elbo4_term_2"].sum()
    
    # Calculate 5th ELBO term
    df_sum_E["elbo5_term"] = df_sum_E["gamma_"] * np.log(df_sum_E["gamma_"])+ (1-df_sum_E["gamma_"])*(np.log(1-df_sum_E["gamma_"])) 
    elbo5 = df_sum_E["elbo5_term"].sum()

    
    # Sum up the ELBO terms
    elbo = elbo1 + elbo2 + elbo3 - elbo4 - elbo5

    #print("elbo1: ", elbo1)
    #print("elbo2: ", elbo2)
    #print("elbo3: ", elbo3)
    #print("elbo4: ", elbo4)
    #print("elbo5: ", elbo5)
    
    print("ELBO: ", elbo)
    
    result_lst.append(elbo)
print(result_lst)


plt.plot([i for i in range(1,11)], result_lst, 'go')

# axis labeling
plt.xlabel('Iteration')
plt.ylabel('ELBO')

# figure name
plt.title("Evidence lower bound as a function of EM iteration")
plt.show()



# Part 4: Evaluating PRS prediction

f_X_train = "X_train.csv"
f_y_train = "y_train.csv"
f_X_test = "X_test.csv"
f_y_test = "y_test.csv"

# Read in the data into pandas dataframes
df_beta_marginal = pd.read_csv(fIn_beta_marginal)
df_LD = pd.read_csv(fIn_LD)

df_X_train = pd.read_csv(f_X_train)
df_y_train = pd.read_csv(f_y_train)
df_X_test = pd.read_csv(f_X_test)
df_y_test = pd.read_csv(f_y_test)

df_X_train.rename(columns={'Unnamed: 0':'name'}, inplace=True)
df_y_train.rename(columns={'Unnamed: 0':'name'}, inplace=True)
df_X_test.rename(columns={'Unnamed: 0':'name'}, inplace=True)
df_y_test.rename(columns={'Unnamed: 0':'name'}, inplace=True)

# For train cases
gam_myu_elmwise.shape
X_train = df_X_train.to_numpy()[:, 1:]
y_hat_train = np.matmul(X_train, gam_myu_elmwise)
y_train = df_y_train["V1"].to_numpy()
y_hat_train = y_hat_train.astype('float64')
y_train = y_train.astype('float64')

# For test cases
X_test = df_X_test.to_numpy()[:, 1:]
y_hat_test = np.matmul(X_test, gam_myu_elmwise)
y_test = df_y_test["V1"].to_numpy()
y_hat_test = y_hat_test.astype('float64')
y_test = y_test.astype('float64')


# Plotting next to each other
plt.subplot(1, 2, 1)
model = np.polyfit(y_hat_train, y_train, 1)
predict = np.poly1d(model)
y_lin_reg = predict(y_hat_train)
r2 = r2_score(y_train, predict(y_hat_train))
print(math.sqrt(r2))
plt.title("Train: r = " + str(math.sqrt(r2)))
plt.ylabel("True phenotype")
plt.xlabel("Predicted phenotype")
plt.scatter(y_hat_train, y_train, c = 'k', s=4)
plt.plot(y_hat_train, y_lin_reg, c = 'b')
plt.ylim([-3, 3.2])

plt.subplot(1, 2, 2)
model2 = np.polyfit(y_hat_test, y_test, 1)
predict2 = np.poly1d(model2)
y_lin_reg2 = predict2(y_hat_test)
r2 = r2_score(y_test, predict(y_hat_test))
print(math.sqrt(r2))
plt.scatter(y_hat_test, y_test, c = 'k', s=4)
plt.plot(y_hat_test, y_lin_reg2, c = 'b')
plt.ylim([-3, 3.2])
plt.title("Test: r = " + str(math.sqrt(r2)))
plt.xlabel("Predicted phenotype")
plt.tight_layout()

plt.show()


# Plotting individual for increased clarity of dots
model = np.polyfit(y_hat_train, y_train, 1)
predict = np.poly1d(model)
y_lin_reg = predict(y_hat_train)
r2 = r2_score(y_train, predict(y_hat_train))
print(math.sqrt(r2))
plt.title("Train: r = " + str(math.sqrt(r2)))
plt.ylabel("True phenotype")
plt.xlabel("Predicted phenotype")
plt.scatter(y_hat_train, y_train, c = 'k', s=4)
plt.plot(y_hat_train, y_lin_reg, c = 'b')
plt.ylim([-3, 3.2])
plt.show()

model2 = np.polyfit(y_hat_test, y_test, 1)
predict2 = np.poly1d(model2)
y_lin_reg2 = predict2(y_hat_test)
r2 = r2_score(y_test, predict(y_hat_test))
print(math.sqrt(r2))
plt.scatter(y_hat_test, y_test, c = 'k', s=4)
plt.plot(y_hat_test, y_lin_reg2, c = 'b')
plt.ylim([-3, 3.2])
plt.title("Test: r = " + str(math.sqrt(r2)))
plt.ylabel("True phenotype")
plt.xlabel("Predicted phenotype")
plt.tight_layout()
plt.show()


# Part 5: Evaluating fine-mapping
colors = np.where(df_sum_E["gamma_"]>0.05,'r','b')
#df.plot.scatter(x="year",y="length",c=colors)
df_sum_E.plot.scatter(x="SNP_name", y="gamma_", alpha=0.5, c=colors)
plt.title("Inferred PIP. Causal SNPs coloured in red.")
plt.show()