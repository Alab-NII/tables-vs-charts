# Format Matters: Tables vs. Charts

This is the repository for the paper: [Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts]
(AAAI 2026) 

## Reproduction of Results
- download [data.zip](https://www.dropbox.com/scl/fi/j4dhzkei8o7ycv56g5nsk/data.zip?rlkey=9fa3x1jz2nnsrggpc2j50v89d&st=y0xxlvgj&dl=0)
- download [outputs.zip](https://www.dropbox.com/scl/fi/m1lo0fikhgtypp92jht0j/outputs.zip?rlkey=krg6k1cyvd1tnmjd19d9yzh86&st=qcbitx9n&dl=0) 


### Table 2: 
Edit ```base_path``` in Line 49 to obtain all results in Table 2. 

```bash
python3 run_eval.py
```



## Running process

### Run the Claim Label Prediction Task for Tables
```bash
python3 run_claim_table.py
```

### Run the Claim Label Prediction Task for Charts
```bash
python3 run_claim_img.py
```


### Combination
```bash
python3 run_claim_combine.py
```


### Evaluation
```bash
python3 run_eval.py
```




The structure of the code in this repository is based on: https://github.com/Alab-NII/SciTabAlign