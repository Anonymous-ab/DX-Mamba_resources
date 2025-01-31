<h1 align="center">DX-Mamba: Exploring State Space Model for Dog X-ray Report Generation</h1>
<p align="center">
  <img src="Figures/DX-Mamba.png" width="800" alt="Workflow of the proposed DX-Mamba model">
</p>



<h2>Repository for codes of the DX-Mamba paper</h2>

<p>This repository provides the coding implementation of the paper "DX-Mamba: Exploring State Space Model for Dog X-ray Report Generation".</p>

<h3>Code implementation</h3>

```python
git clone https://github.com/Anonymous-ab/DX-Mamba_resources
cd DXMamba
conda create -n DXMamba_env python=3.9
conda activate DXMamba_env
pip install -r requirements.txt
```

<h3>Data Preparation and preprocessing</h3>


<h3>Training</h3>

```python
python train.py --network MambaVision-L-1K --train_batchsize 64
```

<h3>Test</h3>

```Python
python test.py --network MambaVision-L-1K --checkpoint ./models_ckpt/transformer_decoderlayers12024-11-08-16-40-56_1627_all/Dog-X-ray_bts_8_MambaVision-L-1K_epo_29_Bleu4_25245_test.pth
```


<h2>Experiments</h2>

<p align="center">
  <img src="Figures/Dog-Xray compare.png" alt=" Table 1: Results comparisons of different methods on the Dog-Xray dataset.">
</p>
<p align="center">Table 1: Results comparisons of different methods on the Dog-Xray dataset.</p>



