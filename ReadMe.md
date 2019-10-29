# 1. CNN training and logits generation (folder: "scripts/"):

```python train_nn.py -h  # Get more information about that```

- Example:

```python train_nn.py -s 15 -m densenet40 -d CIFAR-10 # Seed, model, dataset```

- Logits from pretrained dataset (folder: "pretrained_models")

```python -u get_logits.py -d cifar10 -o c10```
```python -u get_logits.py -d cifar100 -o c100```
```python -u get_logits.py -d svhn -o svhn```


# 2. Calibration model's tuning, training and evaluation (folder: "scripts/"):

- Temperature Scaling (TempS):

```python tune_cal_guo.py -c TemperatureScaling```

- Vector Scaling (VecS):

```python tune_cal_guo.py -c VectorScaling```

- Dirichlet with Off-diagonal and Intercept regularisation (Dir-L2):

```python -u tune_dirichlet_nn.py -i 0 -kf 5 -d --no_mus  # File number, number of cross-folds, double learning, no intercept tuning separately```

- Matrix Scaling with Off-diagonal and Intercept regularisation (MS-ODIR):

```python -u tune_dirichlet_nn.py -i 0 -kf 5 -d --comp_l2 --use_logits  # File number, nr of cross-folds, double learning, complementary l2, use_logits```

- Dirichlet with Off-diagonal and Intercept regularisation (Dir-ODIR):

```python -u tune_cal_odir.py -i 0 -kf 5 -d --comp_l2  # File number, number of cross-folds, double learning, complementary l2 (i.e ODIR).```


# 3. Notebooks (folder "scripts/notebooks")

- Final Results (Table 3 & 4 and Supp. Table 13_18 and Supp. Figure 11)
- Reliability Diagrams of Dirichlet (Figure 1 and Supp. Figure 12)
- MS-ODIR vs VecS (Table 21)


# 4. p-classwise-ECE and p-confidence-ECE generation (folder "scripts/pECE_generation")
<b>(NB! make sure you have generated file "all_scores_val_test_ens_*.p", as it is used for generate_pECE.py)</b>

- Generate p-ECE for Uncalibrated results:

```python generate_uncal_pECE.py -ece_f ECE```
```python generate_uncal_pECE.py -ece_f classwise_ECE```

- Generate p-ECE for Temperature and Vector Scaling results:

```python generate_temp_vec_pECE.py -ece_f ECE```
```python generate_temp_vec_pECE.py -ece_f classwise_ECE```


- Generate p-ECE for Dir-L2, Dir-ODIR and MS-ODIR

```python generate_pECE.py -ece_f ECE -m dir_l2```
```python generate_pECE.py -ece_f classwise_ECE -m dir_l2```

```python generate_pECE.py -ece_f ECE -m dir_l2_mu_off```
```python generate_pECE.py -ece_f classwise_ECE -m dir_l2_mu_off```

```python generate_pECE.py -ece_f ECE -m mat_scale_l2_mu_off --use_logits```
```python generate_pECE.py -ece_f classwise_ECE -m mat_scale_l2_mu_off --use_logits```


# 5. Notebook for p-ECE results (scripts/notebooks)

- pECE results (Supp. Table 19_20 and Supp. Figure 11)
