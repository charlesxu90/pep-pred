# Prediction of cell penetrating peptides

## Prepare environment

```shell
conda env create -f environment.yml -p ./env
conda activate ./env
```

Install apex
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

```