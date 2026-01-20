## Initial imports 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from doptools.chem.solvents import SolventVectorizer
from chython import smiles
```

# Chemical features

## CircuS and ChyLine descriptors

CircuS (*Circu*lar *S*ubstructures) are fragment descriptors that count circular substructures (atoms and their connected environments) within certain radius. ChyLine (*Chy*thon *Line*ar) are linear fragments, that account for linear substructures of different sizes. The fragment structure are kept as columns in the DatFrame in SMILES format. The following example show how they work.

First, the data set of molecules will be loaded. Here we use a data set of photoswitches from (https://doi.org/10.1039/D2SC04306H). Only data with recorded $\lambda(E_{\pi-\pi*})$ values will be used.


```python
photoswitches = pd.read_table("../examples/photoswitches.csv", sep=",", index_col=0)
data_lambda = photoswitches[pd.notnull(photoswitches['E isomer pi-pi* wavelength in nm'])]
data_lambda
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>rate of thermal isomerisation from Z-E in s-1</th>
      <th>logRate</th>
      <th>Solvent used for thermal isomerisation rates</th>
      <th>Z PhotoStationaryState</th>
      <th>E PhotoStationaryState</th>
      <th>E isomer pi-pi* wavelength in nm</th>
      <th>Extinction</th>
      <th>E isomer n-pi* wavelength in nm</th>
      <th>Extinction coefficient in M-1 cm-1</th>
      <th>...</th>
      <th>TPSSh/6-31G** DFT Z isomer pi-pi* wavelength in nm</th>
      <th>TPSSh/6-31G** DFT Z isomer n-pi* wavelength in nm</th>
      <th>CAM-B3LYP/6-31G** DFT E isomer pi-pi* wavelength in nm</th>
      <th>CAM-B3LYP/6-31G** DFT E isomer n-pi* wavelength in nm</th>
      <th>CAM-B3LYP/6-31G** DFT Z isomer pi-pi* wavelength in nm</th>
      <th>CAM-B3LYP/6-31G** DFT Z isomer n-pi* wavelength in nm</th>
      <th>BHLYP/6-31G* DFT E isomer pi-pi* wavelength in nm</th>
      <th>BHLYP/6-31G* DFT E isomer n-pi* wavelength in nm</th>
      <th>BHLYP/6-31G* Z isomer pi-pi* wavelength in nm</th>
      <th>BHLYP/6-31G* DFT Z isomer n-pi* wavelength in nm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C[N]1N=NC(=N1)N=NC2=CC=CC=C2</td>
      <td>2.100000e-07</td>
      <td>-6.68</td>
      <td>MeCN</td>
      <td>76.0</td>
      <td>72.0</td>
      <td>310.0</td>
      <td>1.67</td>
      <td>442.0</td>
      <td>0.0373</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C[N]1C=NC(=N1)N=NC2=CC=CC=C2</td>
      <td>3.800000e-07</td>
      <td>-6.42</td>
      <td>MeCN</td>
      <td>90.0</td>
      <td>84.0</td>
      <td>310.0</td>
      <td>1.87</td>
      <td>438.0</td>
      <td>0.0505</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C[N]1C=CC(=N1)N=NC2=CC=CC=C2</td>
      <td>1.100000e-07</td>
      <td>-6.96</td>
      <td>MeCN</td>
      <td>98.0</td>
      <td>97.0</td>
      <td>320.0</td>
      <td>1.46</td>
      <td>425.0</td>
      <td>0.0778</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C[N]1C=C(C)C(=N1)N=NC2=CC=CC=C2</td>
      <td>1.500000e-06</td>
      <td>-5.82</td>
      <td>MeCN</td>
      <td>96.0</td>
      <td>87.0</td>
      <td>325.0</td>
      <td>1.74</td>
      <td>428.0</td>
      <td>0.0612</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C[N]1C=C(C=N1)N=NC2=CC=CC=C2</td>
      <td>7.600000e-09</td>
      <td>-8.12</td>
      <td>MeCN</td>
      <td>98.0</td>
      <td>70.0</td>
      <td>328.0</td>
      <td>1.66</td>
      <td>417.0</td>
      <td>0.0640</td>
      <td>...</td>
      <td>295.0</td>
      <td>410.0</td>
      <td>305.0</td>
      <td>427.0</td>
      <td>256.0</td>
      <td>401.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>400</th>
      <td>OC%38=C%39N=CC=CC%39=C(/N=N/C%40=NC%41=CC(C)=C...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>456.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>401</th>
      <td>OC%42=C%43N=CC=CC%43=C(/N=N/C%44=NC%45=CC=CC=C...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>437.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>402</th>
      <td>N#CC1C(SC(/N=N/C2=NC(C=CC([N+]([O-])=O)=C3)=C3...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>545.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>403</th>
      <td>N#Cc5c(c6ccc(Cl)cc6)c(/N=N/C7=NC(C=CC([N+]([O-...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>535.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>404</th>
      <td>N#CC9C(SC(/N=N/C%10=NC(C=CC([N+]([O-])=O)=C%11...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>550.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 34 columns</p>
</div>



To transform the SMILES into descriptors, the molecules need to be first transformed into Chython Molecule objects, with the *smiles* function. This also allows us to visualize the molecules.


```python
lambda_mols = [smiles(s) for s in data_lambda.SMILES]
[m.canonicalize() for m in lambda_mols]
[m.clean2d() for m in lambda_mols]
lambda_mols[0]
```




    
![svg](tutorial_01_files/tutorial_01_6_0.svg)
    




```python
from doptools import ChythonCircus

circus_fragmentor = ChythonCircus(0, # minimum radius
                                  3) # maximum radius
# using fit function of sklearn Transformer
circus_fragmentor.fit(lambda_mols)
# using transform function of sklearn Transformer
circus_descriptors = circus_fragmentor.transform(lambda_mols) 
circus_descriptors
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>N</th>
      <th>CN</th>
      <th>nn(n)C</th>
      <th>nnn</th>
      <th>n(c)n</th>
      <th>nc(n)N</th>
      <th>N(=N)C</th>
      <th>cc(c)N</th>
      <th>ccc</th>
      <th>...</th>
      <th>s1ccc(C)c1N=Nc(s)n</th>
      <th>c1(sccn1)N=Nc(s)c</th>
      <th>c1(c(C)c(N)sc1N=N)C</th>
      <th>C(#N)c1c(N)sc(N)c1C</th>
      <th>C(CCC)(O)C</th>
      <th>CCC(CC)OC</th>
      <th>CC(C)OC</th>
      <th>C1CC(CCC1O)C(C)C</th>
      <th>C1CC(CCC1C)OC</th>
      <th>C1CC(CCC1)OC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>18</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>16</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>18</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>18</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>19</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 1129 columns</p>
</div>



The calculator can also use SMILES directly, although an additional parameter *fmt* needs to be used.


```python
circus_fragmentor_smi = ChythonCircus(0, # minimum radius
                                      3, # maximum radius
                                     fmt="smiles") # indicating that SMILES should be used directly

circus_fragmentor_smi.fit(data_lambda.SMILES) # using fit function of sklearn Transformer
circus_descriptors = circus_fragmentor_smi.transform(data_lambda.SMILES) # using transform function of sklearn Transformer
circus_descriptors
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>N</th>
      <th>CN</th>
      <th>NN(N)C</th>
      <th>NN=N</th>
      <th>N(=N)C</th>
      <th>NC(N)=N</th>
      <th>N(N)=C</th>
      <th>CC(N)=C</th>
      <th>C(=C)C</th>
      <th>...</th>
      <th>c1(c)ccsc1N=NC(=N)S</th>
      <th>C=1N=C(N=Nc(s)c)SC=1</th>
      <th>c1(c(c(c)c(N=N)s1)C)N</th>
      <th>c1(c(c(c(N)s1)C#N)c)N</th>
      <th>C(CCC)(O)C</th>
      <th>CCC(CC)OC</th>
      <th>CC(C)OC</th>
      <th>C1CC(CCC1O)C(C)C</th>
      <th>C1CC(CCC1C)OC</th>
      <th>C1CC(CCC1)OC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>18</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>16</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>18</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>18</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>19</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 1469 columns</p>
</div>




```python
from doptools import ChythonLinear

chyline_fragmentor = ChythonLinear(2, # minimum length
                                   8) # maximum length
# using fit function of sklearn Transformer
chyline_fragmentor.fit(lambda_mols) 
# using transform function of sklearn Transformer
chyline_descriptors = chyline_fragmentor.transform(lambda_mols) 
chyline_descriptors
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c:c:c:c</th>
      <th>c:c:c:c:c</th>
      <th>c:c:c:c:cN=Nc</th>
      <th>c:c</th>
      <th>n:nC</th>
      <th>n:cN=Nc:c:c</th>
      <th>Nc</th>
      <th>Nc:c</th>
      <th>n:n:n:cN=N</th>
      <th>n:n:cN=Nc:c</th>
      <th>...</th>
      <th>N#Cc:c:c:s:c</th>
      <th>c:n:cN=Nc:cc</th>
      <th>N#Cc:c:s:c</th>
      <th>N#Cc:c:s:cN=N</th>
      <th>N#Cc:c:s:c:c</th>
      <th>Cc:cc</th>
      <th>Nc:s:cN=Nc</th>
      <th>n:cN=Nc:s:cN</th>
      <th>OCCCCCCC</th>
      <th>NCCCCCCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>17</td>
      <td>18</td>
      <td>3</td>
      <td>15</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>17</td>
      <td>18</td>
      <td>3</td>
      <td>15</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>13</td>
      <td>12</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 2445 columns</p>
</div>



## Fingeprint calculation

Fingerprint calculation in DOPtools is presented by Fingerprinter class, an umbrella class that can calculate several different types of fingerpints. The type itself is defined in *fp_type* parameter. The size of vectors is defined in *nBits*, with 1024 by default. *radius* defined the radius, length or otherwise size of Morgan and RDkit fingerprints and is not used for any other type. Any additional parameters should be given as dictionary into *params* argument. The functionality is otherwise the same and CircuS or ChyLine.


```python
from doptools import Fingerprinter

# Avalon fingperints
avalon_calc = Fingerprinter(fp_type="avalon")
avalon_fp = avalon_calc.fit_transform(lambda_mols)
print("Avalon size:", avalon_fp.shape)

# Torsion fingerprints
torsion_calc = Fingerprinter(fp_type="torsion")
torsion_fp = torsion_calc.fit_transform(lambda_mols)
print("Torsion size:", torsion_fp.shape)

# Morgan fingerprints
morgan_calc_2 = Fingerprinter(fp_type="morgan",
                             radius=2)
morgan_fp_2 = morgan_calc_2.fit_transform(lambda_mols)
print("Morgan, radius 2, size:", morgan_fp_2.shape)

morgan_calc_3 = Fingerprinter(fp_type="morgan",
                             radius=3)
morgan_fp_3 = morgan_calc_3.fit_transform(lambda_mols)
print("Morgan, radius 3, size:", morgan_fp_3.shape)

# Adding Morgan features to fingerprints
morganf_calc_3 = Fingerprinter(fp_type="morgan",
                              radius=3, 
                              params={"useFeatures":True}) # additional parameters should be given in a dictionary here
morganf_fp_3 = morganf_calc_3.fit_transform(lambda_mols)
print("Morgan with features, radius 3, size:", morganf_fp_3.shape)
```

    Avalon size: (392, 1024)
    Torsion size: (392, 1024)
    Morgan, radius 2, size: (392, 1024)
    Morgan, radius 3, size: (392, 1024)
    Morgan with features, radius 3, size: (392, 1024)
    

# Concatenation of features

*ComplexFragmentor* class allows to concatenate several feature types into the same table for one dataset, including the possibility of using different structure columns. Here we show the examples for concatenaition of structural and physico-chemical descriptors. For this, we will use an extract of data from (https://doi.org/10.1002/anie.202218659) related to the catalyst modeling.


```python
cat_data = pd.read_excel("../examples/THP_extract.xls")
cat_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>SMILES</th>
      <th>Ar</th>
      <th>R</th>
      <th>solvent</th>
      <th>concentration</th>
      <th>T(K)</th>
      <th>e.r.</th>
      <th>ratio(R)</th>
      <th>ddG</th>
      <th>ddG calib (C=0.05)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccccc1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>51.6:48.4</td>
      <td>0.515861</td>
      <td>0.1864</td>
      <td>0.2957</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)(=O)N=P1...</td>
      <td>[C:2][c:1]1ccccc1</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>25.3:74.7</td>
      <td>0.252518</td>
      <td>-3.0060</td>
      <td>-4.7684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Cc1ccc(cc1)-c1cc2ccccc2c-2c1OP(NP1(Oc3c(cc4ccc...</td>
      <td>[C:2][c:1]1ccc(C)cc1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>47.1:52.9</td>
      <td>0.470613</td>
      <td>-0.3260</td>
      <td>-0.5171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>Cc1ccc(cc1)-c1cc2ccccc2c-2c1OP(NP1(Oc3c(cc4ccc...</td>
      <td>[C:2][c:1]1ccc(C)cc1</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>33.4:66.6</td>
      <td>0.333583</td>
      <td>-1.9169</td>
      <td>-3.0408</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>COc1ccc(cc1)-c1cc2ccccc2c-2c1OP(NP1(Oc3c(cc4cc...</td>
      <td>COc1cc[c:1]([C:2])cc1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>61.6:38.4</td>
      <td>0.616426</td>
      <td>1.3141</td>
      <td>2.0846</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccc2ccccc2c1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>41.2:58.8</td>
      <td>0.412240</td>
      <td>-1.0415</td>
      <td>-1.6521</td>
    </tr>
    <tr>
      <th>6</th>
      <td>38</td>
      <td>Cc1cc(C)cc(c1)-c1cc2ccccc2c-2c1OP(NP1(Oc3c(cc4...</td>
      <td>[C:2][c:1]1cc(C)cc(C)c1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>56.4:43.6</td>
      <td>0.563633</td>
      <td>0.7514</td>
      <td>1.1919</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1cccc2ccccc12</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.125</td>
      <td>333.15</td>
      <td>70.2:29.8</td>
      <td>0.702047</td>
      <td>2.3740</td>
      <td>2.8960</td>
    </tr>
    <tr>
      <th>8</th>
      <td>39</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)(=O)N=P1(NP2(Oc3c(cc...</td>
      <td>[C:2][c:1]1ccc2ccccc2c1</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>36.4:63.6</td>
      <td>0.364384</td>
      <td>-1.5412</td>
      <td>-2.4448</td>
    </tr>
    <tr>
      <th>9</th>
      <td>40</td>
      <td>Cc1cc(C)cc(c1)-c1cc2ccccc2c-2c1OP(NP1(Oc3c(cc4...</td>
      <td>[C:2][c:1]1cc(C)cc(C)c1</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>48.1:51.9</td>
      <td>0.480504</td>
      <td>-0.2161</td>
      <td>-0.3428</td>
    </tr>
    <tr>
      <th>10</th>
      <td>41</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)(=O)N=P1...</td>
      <td>[C:2][c:1]1ccc2ccccc2c1</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>27.3:72.7</td>
      <td>0.272535</td>
      <td>-2.7195</td>
      <td>-4.3139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>CC(C)(C)c1ccc(cc1)-c1cc2ccccc2c-2c1OP(NP1(Oc3c...</td>
      <td>[C:2][c:1]1ccc(cc1)C(C)(C)C</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.125</td>
      <td>333.15</td>
      <td>17.6:82.4</td>
      <td>0.175871</td>
      <td>-4.2784</td>
      <td>-5.2191</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10</td>
      <td>CC(C)(C)c1cccc(c1)-c1cc2ccccc2c-2c1OP(NP1(Oc3c...</td>
      <td>[C:2][c:1]1cccc(c1)C(C)(C)C</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>66.0:34.0</td>
      <td>0.659907</td>
      <td>1.8361</td>
      <td>2.9126</td>
    </tr>
    <tr>
      <th>13</th>
      <td>42</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccc(cc1)-c1ccccc1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>53.0:47.0</td>
      <td>0.530361</td>
      <td>0.3570</td>
      <td>0.5663</td>
    </tr>
    <tr>
      <th>14</th>
      <td>43</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1cccc(c1)-c1ccccc1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.125</td>
      <td>333.15</td>
      <td>55.3:44.7</td>
      <td>0.553035</td>
      <td>0.5898</td>
      <td>0.7195</td>
    </tr>
    <tr>
      <th>15</th>
      <td>44</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)(=O)N=P1(NP2(Oc3c(cc...</td>
      <td>[C:2][c:1]1cccc(c1)-c1ccccc1</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>46.2:53.8</td>
      <td>0.461988</td>
      <td>-0.4220</td>
      <td>-0.6694</td>
    </tr>
    <tr>
      <th>16</th>
      <td>45</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)(=O)N=P1(NP2(Oc3c(cc...</td>
      <td>[C:2][c:1]1ccc(cc1)-c1ccccc1</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>28.1:71.9</td>
      <td>0.280771</td>
      <td>-2.6055</td>
      <td>-4.1331</td>
    </tr>
    <tr>
      <th>17</th>
      <td>46</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)(=O)N=P1...</td>
      <td>[C:2][c:1]1ccc(cc1)-c1ccccc1</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>23.8:76.2</td>
      <td>0.238138</td>
      <td>-3.2212</td>
      <td>-5.1098</td>
    </tr>
    <tr>
      <th>18</th>
      <td>47</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccc(cc1)S(F)(F)(F)(F)F</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>49.7:50.3</td>
      <td>0.496668</td>
      <td>-0.0391</td>
      <td>-0.0620</td>
    </tr>
    <tr>
      <th>19</th>
      <td>48</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1cccc(c1)S(F)(F)(F)(F)F</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>49.4:50.6</td>
      <td>0.494355</td>
      <td>-0.0625</td>
      <td>-0.0991</td>
    </tr>
    <tr>
      <th>20</th>
      <td>49</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)(=O)N=P1(NP2(Oc3c(cc...</td>
      <td>[C:2][c:1]1ccc(cc1)S(F)(F)(F)(F)F</td>
      <td>Fc1c(F)c(F)c(c(F)c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>27.7:72.3</td>
      <td>0.277160</td>
      <td>-2.6553</td>
      <td>-4.2121</td>
    </tr>
    <tr>
      <th>21</th>
      <td>50</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)(=O)N=P1...</td>
      <td>[C:2][c:1]1ccc(cc1)S(F)(F)(F)(F)F</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>21.7:78.3</td>
      <td>0.216818</td>
      <td>-3.5575</td>
      <td>-5.6433</td>
    </tr>
    <tr>
      <th>22</th>
      <td>51</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)(=O)N=P1...</td>
      <td>[C:2][c:1]1cccc(c1)S(F)(F)(F)(F)F</td>
      <td>Fc1c(F)c(F)c2c(F)c(c(F)c(F)c2c1F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>46.1:53.9</td>
      <td>0.461276</td>
      <td>-0.4299</td>
      <td>-0.6820</td>
    </tr>
    <tr>
      <th>23</th>
      <td>52</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccc(cc1)-c1ccc2ccccc2c1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>49.4:50.6</td>
      <td>0.493758</td>
      <td>-0.0692</td>
      <td>-0.1098</td>
    </tr>
    <tr>
      <th>24</th>
      <td>23</td>
      <td>CC1(C)c2ccccc2-c2ccc(cc12)-c1cc2ccccc2c-2c1OP(...</td>
      <td>[C:2][c:1]1ccc2-c3ccccc3C(C)(C)c2c1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>60.8:39.2</td>
      <td>0.607745</td>
      <td>1.2128</td>
      <td>1.9239</td>
    </tr>
    <tr>
      <th>25</th>
      <td>53</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccc2c3ccccc3c3ccccc3c2c1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>52.9:47.1</td>
      <td>0.529012</td>
      <td>0.3218</td>
      <td>0.5105</td>
    </tr>
    <tr>
      <th>26</th>
      <td>54</td>
      <td>FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)N=P1(NP2...</td>
      <td>[C:2][c:1]1ccc2-c3ccccc3C3(CCC3)c2c1</td>
      <td>FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>74.7:25.3</td>
      <td>0.747444</td>
      <td>3.0055</td>
      <td>4.7676</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>FC(F)(F)c1cc(cc(c1)C(F)(F)F)-c1cc2ccccc2c-2c1O...</td>
      <td>[C:2][c:1]1cc(cc(c1)C(F)(F)F)C(F)(F)F</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>51.0:49.0</td>
      <td>0.509999</td>
      <td>0.1175</td>
      <td>0.1864</td>
    </tr>
    <tr>
      <th>28</th>
      <td>30</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccc2-c3ccccc3C3(CCCC3)c2c1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.125</td>
      <td>333.15</td>
      <td>80.8:19.2</td>
      <td>0.808299</td>
      <td>3.9860</td>
      <td>4.8624</td>
    </tr>
    <tr>
      <th>29</th>
      <td>55</td>
      <td>CC(C)(C)c1ccc(cc1)-c1ccc(cc1)-c1cc2ccccc2c-2c1...</td>
      <td>[C:2][c:1]1ccc(cc1)-c1ccc(cc1)C(C)(C)C</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>54.8:45.2</td>
      <td>0.548180</td>
      <td>0.5676</td>
      <td>0.9004</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>FC(F)(F)S(=O)(=O)N=P1(NP2(Oc3c(cc4ccccc4c3-c3c...</td>
      <td>[C:2][c:1]1ccc2-c3ccccc3C3(CCCCC3)c2c1</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.125</td>
      <td>333.15</td>
      <td>79.9:20.1</td>
      <td>0.798531</td>
      <td>3.8146</td>
      <td>4.6533</td>
    </tr>
    <tr>
      <th>31</th>
      <td>56</td>
      <td>Cc1ccc2-c3ccc(cc3C3(CCCC3)c2c1)-c1cc2ccccc2c-2...</td>
      <td>[C:2][c:1]1ccc2-c3ccc(C)cc3C3(CCCC3)c2c1</td>
      <td>FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F...</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>77.8:22.2</td>
      <td>0.778370</td>
      <td>3.4796</td>
      <td>5.5197</td>
    </tr>
    <tr>
      <th>32</th>
      <td>57</td>
      <td>CCCCC1(CCCC)c2ccccc2-c2ccc(cc12)-c1cc2ccccc2c-...</td>
      <td>CCCCC1(CCCC)c2ccccc2-c2cc[c:1]([C:2])cc12</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>76.5:23.5</td>
      <td>0.765416</td>
      <td>3.2758</td>
      <td>5.1964</td>
    </tr>
    <tr>
      <th>33</th>
      <td>58</td>
      <td>CC1(C)c2ccc(cc2-c2c1ccc1ccccc21)-c1cc2ccccc2c-...</td>
      <td>[C:2][c:1]1ccc-2c(c1)C(C)(C)c1ccc3ccccc3c-21</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>methylcyclohexane</td>
      <td>0.250</td>
      <td>353.15</td>
      <td>62.1:37.9</td>
      <td>0.620823</td>
      <td>1.4477</td>
      <td>2.2965</td>
    </tr>
    <tr>
      <th>34</th>
      <td>59</td>
      <td>CC(C)(C)c1ccc2-c3ccc(cc3C3(CCCC3)c2c1)-c1cc2cc...</td>
      <td>[C:2][c:1]1ccc2-c3ccc(cc3C3(CCCC3)c2c1)C(C)(C)C</td>
      <td>FC(F)(F)S(=O)=O</td>
      <td>cyclohexane</td>
      <td>0.250</td>
      <td>333.15</td>
      <td>76.3:23.7</td>
      <td>0.762888</td>
      <td>3.2369</td>
      <td>5.1347</td>
    </tr>
  </tbody>
</table>
</div>



The catalyst is represented here as a combination of Ar and R substituents. Due to the size differences in these substituents, we want to use different sized CircuS descriptors for them. To do that, we need to add columns with actual molecular structures into the table, as the ComplexFragmentor takes a DataFrame as the input.


```python
ars = [smiles(s) for s in cat_data["Ar"]]
[c.clean2d() for c in ars]

rs = [smiles(s) for s in cat_data["R"]]
[c.clean2d() for c in rs]

cat_data["ar_mol"] = ars
cat_data["r_mol"] = rs
```


```python
from doptools import ComplexFragmentor

cf = ComplexFragmentor(associator=[
    ("ar_mol", ChythonCircus(0,3)), # the associator connects the column name in the table to the descriptor calculator
    ("r_mol", ChythonCircus(0,1))
])
cf.fit(cat_data) # DataFrame needs to be given as argument, as it will use the associator keys to pick the correct columns
cat_desc = cf.transform(cat_data)
cat_desc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ar_mol::C</th>
      <th>ar_mol::CC</th>
      <th>ar_mol::cc(c)C</th>
      <th>ar_mol::ccc</th>
      <th>ar_mol::c(c)c(cc)C</th>
      <th>ar_mol::cc(ccc)C</th>
      <th>ar_mol::c(c)ccc</th>
      <th>ar_mol::c1ccccc1C</th>
      <th>ar_mol::c1ccccc1</th>
      <th>ar_mol::c1cc(ccc1C)C</th>
      <th>...</th>
      <th>r_mol::FC</th>
      <th>r_mol::FC(F)(F)S</th>
      <th>r_mol::O=S(=O)C</th>
      <th>r_mol::S=O</th>
      <th>r_mol::cc(c)F</th>
      <th>r_mol::cc(c)c</th>
      <th>r_mol::cc(c)S</th>
      <th>r_mol::FC(F)(F)C</th>
      <th>r_mol::CC(C)(F)F</th>
      <th>r_mol::FC(F)(S)C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>17</td>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>16</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>19</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>17</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>17</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>19</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>19</td>
      <td>2</td>
      <td>8</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>17</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>22</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>20</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>22</td>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>35 rows × 141 columns</p>
</div>



The columns of the table contain SMILES of the fragments, preceded by the name of the column where the structure comes from, separated by "::". 

SMILES columns can still be used to calculate descriptors.


```python
cf2 = ComplexFragmentor(associator=[
    ("Ar", ChythonCircus(0,3,fmt="smiles")), # the associator connects the column name in the table to the descriptor calculator
    ("R", ChythonCircus(0,1,fmt="smiles"))
])
cf2.fit(cat_data) # DataFrame needs to be given as argument, as it will use the associator keys to pick the correct columns
cat_desc2 = cf2.transform(cat_data)
cat_desc2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ar::C</th>
      <th>Ar::CC</th>
      <th>Ar::cc(c)C</th>
      <th>Ar::ccc</th>
      <th>Ar::c(c)c(cc)C</th>
      <th>Ar::cc(ccc)C</th>
      <th>Ar::c(c)ccc</th>
      <th>Ar::c1ccccc1C</th>
      <th>Ar::c1ccccc1</th>
      <th>Ar::c1cc(ccc1C)C</th>
      <th>...</th>
      <th>R::FC</th>
      <th>R::FC(F)(F)S</th>
      <th>R::O=S(=O)C</th>
      <th>R::S=O</th>
      <th>R::cc(c)F</th>
      <th>R::cc(c)c</th>
      <th>R::cc(c)S</th>
      <th>R::FC(F)(F)C</th>
      <th>R::CC(C)(F)F</th>
      <th>R::FC(F)(S)C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>13</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>17</td>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>16</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>19</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>17</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>17</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>19</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>19</td>
      <td>2</td>
      <td>8</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>17</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>22</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>20</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>22</td>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>35 rows × 141 columns</p>
</div>



Solvent descriptors can also be concatenated into the table. The solvent descriptors are physico-chemical parameters (acidity, basicity, polarizability, dipolarity). These are present as tabular values and require specific names of solvents in the column.


```python
from doptools import ComplexFragmentor
from doptools import SolventVectorizer

# this calculator doesn't fit anything, so it can be instantiated before the calculation
sv = SolventVectorizer() 
# the associator connects the column name in the table to the descriptor calculator
cf = ComplexFragmentor(associator=[
    ("ar_mol", ChythonCircus(0,3)), 
    ("r_mol", ChythonCircus(0,1)),
    ("solvent",sv) # added the calcualtion of solvent descriptors 
])
# DataFrame needs to be given as argument, 
# as it will use the associator keys to pick the correct columns
cf.fit(cat_data) 
cat_desc = cf.transform(cat_data)
# only some columns are shown for demonstration purposes
cat_desc[['ar_mol::C',  'ar_mol::cc(c)C', 
       'ar_mol::c(c)c(cc)C','r_mol::FC(F)(F)C',
       'r_mol::CC(C)(F)F',  'solvent::SP Katalan',
       'solvent::SA Katalan', 'solvent::SB Katalan']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ar_mol::C</th>
      <th>ar_mol::cc(c)C</th>
      <th>ar_mol::c(c)c(cc)C</th>
      <th>r_mol::FC(F)(F)C</th>
      <th>r_mol::CC(C)(F)F</th>
      <th>solvent::SP Katalan</th>
      <th>solvent::SA Katalan</th>
      <th>solvent::SB Katalan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>16</th>
      <td>13</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>17</th>
      <td>13</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>23</th>
      <td>17</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>24</th>
      <td>16</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>25</th>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>26</th>
      <td>17</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>29</th>
      <td>17</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>30</th>
      <td>19</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>31</th>
      <td>19</td>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>32</th>
      <td>22</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>33</th>
      <td>20</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.675</td>
      <td>0</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>34</th>
      <td>22</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.683</td>
      <td>0</td>
      <td>0.073</td>
    </tr>
  </tbody>
</table>
</div>



## Calculation of fragments with dynamic bonds


```python
# reaction must be mapped
# reaction SMILES are cut due to size
r_smiles = "[OH:4][CH2:13][CH2:12][CH2:11][CH2:10][C:2](=[CH2:3])[C:1]1=[CH:5][CH:6]=[CH:7][CH:8]=[CH:9]1>>[CH3:3][C:2]1([CH2:10][CH2:11][CH2:12][CH2:13][O:4]1)[C:1]1=[CH:5][CH:6]=[CH:7][CH:8]=[CH:9]1"
reac = smiles(r_smiles)
circus_fragmentor_r = ChythonCircus(0, # minimum radius
                                  3) # maximum radius
# using fit_transform function of sklearn Transformer
circus_fragmentor_r.fit_transform([reac])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>O</th>
      <th>CC(C)=C</th>
      <th>CC(C)([.&gt;-]O)[=&gt;-]C</th>
      <th>C[=&gt;-]C</th>
      <th>CO[.&gt;-]C</th>
      <th>C=CC</th>
      <th>CCC</th>
      <th>C(O)C</th>
      <th>C(C=C)(C([=&gt;-]C)(C)[.&gt;-]O)=CC</th>
      <th>...</th>
      <th>OCCCC</th>
      <th>C(O[.&gt;-]C)CC</th>
      <th>C1=CC(C([=&gt;-]C)([.&gt;-]OC)CC)=CC=C1</th>
      <th>C[=&gt;-]C1([.&gt;-]OCCCC1)C(C=C)=CC</th>
      <th>C[=&gt;-]C1([.&gt;-]OCCCC1)C(=C)C</th>
      <th>C1=CC(C(C)([=&gt;-]C)[.&gt;-]O)=CC=C1</th>
      <th>C1=CC(=CC=C1)C</th>
      <th>C1=CC=CC=C1</th>
      <th>C[=&gt;-]C1([.&gt;-]OCCCC1)C</th>
      <th>C[.&gt;-]1CCCCO[.&gt;-]1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 27 columns</p>
</div>



# Command line interface to calculate descriptors

Normally these scripts are supposed to be launched from the terminal and serve as beckend for server calculations. However, an example is given here for illustration purposes. It calculates the Morgan fingeprrints of three different radii for the dataset of photoswitches from before. The script skips all the row that don't have the indicated property value, so the descriptors will only be calucated for the molecules that have them. 


```python
import subprocess

result = subprocess.run(['launch_preparer.py', 
                         '-i', '../examples/photoswitches.csv', # the input file
                         '--structure_col', 'SMILES', # the name of the column that contains the structures
                         '--property_col', 'E isomer pi-pi* wavelength in nm', #the name of the column that contains the property
                         '--property_names', 'Epipi', # the alias for the property name to be used in the file names
                         '--morgan',  # indicate that the Morgan fingerprints will be calculated
                         '--morgan_nBits', '1024', # the length of the Morgan FP vector
                         '--morgan_radius', '2', '3', '4', # the raddi for Morgan FP, several different can be given at the same time
                         '-o', 'lambda/morgan', # the name of the output directory
                         ], stdout=subprocess.PIPE)
print(result.stdout.decode())

```

    The output directory lambda/morgan created
    'E isomer pi-pi* wavelength in nm' column warning: only 392 out of 405 instances have the property.
    Molecules that don't have the property will be discarded from the set.
    
    

Full list of parameters can be seen using the help function (*-h*) of the script.


```python
result = subprocess.run(['launch_preparer.py', '-h'], stdout=subprocess.PIPE)
print(result.stdout.decode())
```

    usage: Descriptor calculator [-h] -i INPUT [--structure_col STRUCTURE_COL]
                                 [--concatenate CONCATENATE [CONCATENATE ...]]
                                 --property_col PROPERTY_COL [PROPERTY_COL ...]
                                 [--property_names PROPERTY_NAMES [PROPERTY_NAMES ...]]
                                 [--standardize] -o OUTPUT [-f {svm,csv}]
                                 [-p PARALLEL] [-s] [--separate_folders]
                                 [--load_config LOAD_CONFIG] [--morgan]
                                 [--morgan_nBits MORGAN_NBITS [MORGAN_NBITS ...]]
                                 [--morgan_radius MORGAN_RADIUS [MORGAN_RADIUS ...]]
                                 [--morganfeatures]
                                 [--morganfeatures_nBits MORGANFEATURES_NBITS [MORGANFEATURES_NBITS ...]]
                                 [--morganfeatures_radius MORGANFEATURES_RADIUS [MORGANFEATURES_RADIUS ...]]
                                 [--rdkfp]
                                 [--rdkfp_nBits RDKFP_NBITS [RDKFP_NBITS ...]]
                                 [--rdkfp_length RDKFP_LENGTH [RDKFP_LENGTH ...]]
                                 [--rdkfplinear]
                                 [--rdkfplinear_nBits RDKFPLINEAR_NBITS [RDKFPLINEAR_NBITS ...]]
                                 [--rdkfplinear_length RDKFPLINEAR_LENGTH [RDKFPLINEAR_LENGTH ...]]
                                 [--layered]
                                 [--layered_nBits LAYERED_NBITS [LAYERED_NBITS ...]]
                                 [--layered_length LAYERED_LENGTH [LAYERED_LENGTH ...]]
                                 [--avalon]
                                 [--avalon_nBits AVALON_NBITS [AVALON_NBITS ...]]
                                 [--atompairs]
                                 [--atompairs_nBits ATOMPAIRS_NBITS [ATOMPAIRS_NBITS ...]]
                                 [--torsion]
                                 [--torsion_nBits TORSION_NBITS [TORSION_NBITS ...]]
                                 [--linear]
                                 [--linear_min LINEAR_MIN [LINEAR_MIN ...]]
                                 [--linear_max LINEAR_MAX [LINEAR_MAX ...]]
                                 [--circus]
                                 [--circus_min CIRCUS_MIN [CIRCUS_MIN ...]]
                                 [--circus_max CIRCUS_MAX [CIRCUS_MAX ...]]
                                 [--onbond] [--mordred2d] [--solvent SOLVENT]
    
    Prepares the descriptor files for hyperparameter optimization launch.
    
    options:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Input file, requires csv or Excel format
      --structure_col STRUCTURE_COL
                            Column name with molecular structures representations.
                            Default = SMILES.
      --concatenate CONCATENATE [CONCATENATE ...]
                            Additional column names with molecular structures
                            representations to be concatenated with the primary
                            structure column.
      --property_col PROPERTY_COL [PROPERTY_COL ...]
                            Column with properties to be used. Case sensitive.
      --property_names PROPERTY_NAMES [PROPERTY_NAMES ...]
                            Alternative name for the property columns specified by
                            --property_col.
      --standardize         Standardize the input structures? Default = False.
      -o OUTPUT, --output OUTPUT
                            Output folder where the descriptor files will be
                            saved.
      -f {svm,csv}, --format {svm,csv}
                            Descriptor files format. Default = svm.
      -p PARALLEL, --parallel PARALLEL
                            Number of parallel processes to use. Default = 0
      -s, --save            Save (pickle) the fragmentors for each descriptor
                            type.
      --separate_folders    Save each descriptor type into a separate folders.
      --load_config LOAD_CONFIG
                            Load descriptor configuration from a JSON file. JSON
                            parameters are prioritized! Use "basic" to load
                            default parameters
      --morgan              Option to calculate Morgan fingerprints.
      --morgan_nBits MORGAN_NBITS [MORGAN_NBITS ...]
                            Number of bits for Morgan FP. Allows several numbers,
                            which will be stored separately. Default = 1024.
      --morgan_radius MORGAN_RADIUS [MORGAN_RADIUS ...]
                            Maximum radius of Morgan FP. Allows several numbers,
                            which will be stored separately. Default = 2.
      --morganfeatures      Option to calculate Morgan feature fingerprints.
      --morganfeatures_nBits MORGANFEATURES_NBITS [MORGANFEATURES_NBITS ...]
                            Number of bits for Morgan feature FP. Allows several
                            numbers, which will be stored separately. Default =
                            1024.
      --morganfeatures_radius MORGANFEATURES_RADIUS [MORGANFEATURES_RADIUS ...]
                            Maximum radius of Morgan feature FP. Allows several
                            numbers, which will be stored separately. Default = 2.
      --rdkfp               Option to calculate RDkit fingerprints.
      --rdkfp_nBits RDKFP_NBITS [RDKFP_NBITS ...]
                            Number of bits for RDkit FP. Allows several numbers,
                            which will be stored separately. Default = 1024.
      --rdkfp_length RDKFP_LENGTH [RDKFP_LENGTH ...]
                            Maximum length of RDkit FP. Allows several numbers,
                            which will be stored separately. Default = 3.
      --rdkfplinear         Option to calculate RDkit linear fingerprints.
      --rdkfplinear_nBits RDKFPLINEAR_NBITS [RDKFPLINEAR_NBITS ...]
                            Number of bits for RDkit linear FP. Allows several
                            numbers, which will be stored separately. Default =
                            1024.
      --rdkfplinear_length RDKFPLINEAR_LENGTH [RDKFPLINEAR_LENGTH ...]
                            Maximum length of RDkit linear FP. Allows several
                            numbers, which will be stored separately. Default = 3.
      --layered             Option to calculate RDkit layered fingerprints.
      --layered_nBits LAYERED_NBITS [LAYERED_NBITS ...]
                            Number of bits for RDkit layered FP. Allows several
                            numbers, which will be stored separately. Default =
                            1024.
      --layered_length LAYERED_LENGTH [LAYERED_LENGTH ...]
                            Maximum length of RDkit layered FP. Allows several
                            numbers, which will be stored separately. Default = 3.
      --avalon              Option to calculate Avalon fingerprints.
      --avalon_nBits AVALON_NBITS [AVALON_NBITS ...]
                            Number of bits for Avalon FP. Allows several numbers,
                            which will be stored separately. Default = 1024.
      --atompairs           Option to calculate atom pair fingerprints.
      --atompairs_nBits ATOMPAIRS_NBITS [ATOMPAIRS_NBITS ...]
                            Number of bits for atom pair FP. Allows several
                            numbers, which will be stored separately. Default =
                            1024.
      --torsion             Option to calculate topological torsion fingerprints.
      --torsion_nBits TORSION_NBITS [TORSION_NBITS ...]
                            Number of bits for topological torsion FP. Allows
                            several numbers, which will be stored separately.
                            Default = 1024.
      --linear              Option to calculate ChyLine fragments.
      --linear_min LINEAR_MIN [LINEAR_MIN ...]
                            Minimum length of linear fragments. Allows several
                            numbers, which will be stored separately. Default = 2.
      --linear_max LINEAR_MAX [LINEAR_MAX ...]
                            Maximum length of linear fragments. Allows several
                            numbers, which will be stored separately. Default = 5.
      --circus              Option to calculate CircuS fragments.
      --circus_min CIRCUS_MIN [CIRCUS_MIN ...]
                            Minimum radius of CircuS fragments. Allows several
                            numbers, which will be stored separately. Default = 1.
      --circus_max CIRCUS_MAX [CIRCUS_MAX ...]
                            Maximum radius of CircuS fragments. Allows several
                            numbers, which will be stored separately. Default = 2.
      --onbond              Toggle the calculation of CircuS fragments on bonds.
                            With this option the fragments will be bond-cetered,
                            making a bond the minimal element.
      --mordred2d           Option to calculate Mordred 2D descriptors.
      --solvent SOLVENT     Column that contains the solvents. Check the available
                            solvents in the solvents.py script.
    
    

Example of calculation of CircuS fragments with saving the pickled objects for fragmentors.


```python
result = subprocess.run(['launch_preparer.py', 
                         '-i', '../examples/photoswitches.csv', # the input file
                         '--structure_col', 'SMILES', # the name of the column that contains the structures
                         '--property_col', 'E isomer pi-pi* wavelength in nm', #the name of the column that contains the property
                         '--property_names', 'Epipi', # the alias for the property name to be used in the file names
                         '--circus',  
                         '--circus_min', '0', 
                         '--circus_max', '2', '3', '4', '5', 
                         '-o', 'lambda/circus', # the name of the output directory
                         '--save' # indicates that the pickled objects of fragmentors housld be saved
                         ], stdout=subprocess.PIPE)
print(result.stdout.decode())
```

    The output directory lambda/circus created
    'E isomer pi-pi* wavelength in nm' column warning: only 392 out of 405 instances have the property.
    Molecules that don't have the property will be discarded from the set.
    
    


```python

```
