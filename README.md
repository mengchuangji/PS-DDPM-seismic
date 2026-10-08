This repo contains the official implementation for the paper [Posterior Sampling with Generative Diffusion Priors for Multi-purpose Seismic Data Recovery]

by Weifeng Geng, Chuangji Meng, Jinghuai Gao, Zongben Xu，Yajun Tian, Hongling Chen, Tieqiang Zhang

## Running Experiments

### Dependencies

Run the following conda line to install all necessary python packages for our code and set up the environment.

```bash
conda env create -f environment.yml
```

The environment includes `cudatoolkit=11.0`. You may change that depending on your hardware.


Configuration files are in `configs/`. You don't need to include the prefix `configs/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main_*.py
├── image_samples # contains generated samples
│   └── <i>
│       ├── stochastic_variation.png # samples generated from checkpoint_x.pth, including original, degraded, mean, and std   
│       ├── results.mat # the pytorch tensor corresponding to stochastic_variation.png
│       └── y_0.mat # the pytorch tensor containing the input y of SNIPS
```


### Running 

The general command to sample from the model is as follows:
```
python main.py --ni --config {CONFIG}.yml --doc {DATASET} --timesteps {STEPS} --eta {ETA} --etaB {ETA_B} --deg {DEGRADATION} --sigma_0 {SIGMA_0} 
```
where the following are options
- `ETA` is the eta hyperparameter in the paper. (default: `0.80`)
- `ETA_B` is the eta_b hyperparameter in the paper. (default: `1`)
- `STEPS` controls how many timesteps used in the process.
- `DEGREDATION` is the type of degredation allowed. (One of: `cs2`, `cs4`, `int_r`, `int_ir`, `int_c`,`int_u`, `deno`)
- `SIGMA_0` is the noise observed in y.
- `CONFIG` is the name of the config file (see `configs/` for a list), including hyperparameters such as batch size and network architectures.

For example, for Interpolation of consecutive missing data, sampling posterior solution using pretrained  model with with added noise of standard deviation 0.1, and obtain 3 variations, 20 steps:
we can run the following
```bash
python main-mcj-seis-GT.py -i images --config seismic.yml --doc marmousi_v2_nm -num_variations 3 --deg deno int_c --sigma_0 0.1
```
Samples will be saved in `<exp>/image_samples/marmousi_v2_nm`.

The available degradations are:`cs2`, `cs4`, `int_r`, `int_ir`, `int_c`,`int_u`, `deno`. The sigma_0 (noise level of observation) can be set manually or estimated automatically.

If you don't need GT to evaluate the results, use main_mcj_sample_noGT.py for synthetic data (e.g., mat foramt file) and main_mcj_seis_noGT_field.py for real data (SEGY/SGY file format).

### test data preparation
You can test on Marmousi (mat file format)/ [Opensegy]("http://s3.amazonaws.com/open.source.geoscience/open_data) (SEGY/SGY file format) / field data (SEGY/SGY file format)

### Pretrained models
We provide some trained models and log files in files `<exp>/logs/MmsSegyopenf_lin` (linear noise schedules), `<exp>/logs/MmsSegyopenf_cos` (cosine noise schedules) and `<exp>/logs/MmsSegyopenf_pow4` (power noise schedules,a=4).  see [pretrained model](Shared via Baidu Netdisk: DDPM-seismic)
Link: https://pan.baidu.com/s/13dqNc7UgtbVbDgcM-zMEVw?pwd=1111, Extraction Code: 1111. 
Alternatively, these code, data and checkpoint files are provided as-is from our open source project [DDPM-seismic](https://github.com/mengchuangji/DDPM-seismic)， you can also train a new model from scratch by yourself.
