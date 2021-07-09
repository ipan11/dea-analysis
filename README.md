## Data Envelopment Analysis 

- Code to reproduce the following paper
  
  ```Ford, M. J., & Abdulla, A. (2021). New Methods for Evaluating Energy Infrastructure Development Risks. Risk Analysis.```

### Install instructions

- Tested on a Linux machine (debian variant) with anaconda and R installed.
- Clone the repository in a folder and then type the following in your shell.
  
```bash
conda create -n DEA python=3.8
conda activate DEA
pip install -r requirements.txt
```

### Run instructions

- Following command runs the `main.py` script to produce outputs files and plots in the folders `results`, `artifacts` and `plots`. 

```bash
make run
```

- To remove all the output files type

```bash
make clean
```