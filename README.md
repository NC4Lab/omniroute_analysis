# omniroute_analysis
Shared code base for Omniroute experiment analysis

## Environment Setup

Use within a conda environment `omniroute_analysis` that you create using:
```
conda env create -f omniroute_analysis_env.yml
```

Depending on your terminal run:
```
conda init powershell
```

Run the following to activate the environment:
```
conda activate omniroute_analysis_env
```

Create a `.env` file in the repo root with the following variables:
```
TRODES_PATH=
NC4_DATA_PATH=
```

For example (using double backslashes on Windows paths):
```
TRODES_PATH=C:\\Users\\lester\\MeDocuments\\Research\\MadhavLab\\Projects\\SpikeGadgets\\Trodes_2-3-4_Windows64\\Trodes.exe
NC4_DATA_PATH=C:\\Users\\lester\\UBC\\Madhav, Manu - lesterkaur2024gate\\analysis\\gate_ephys_test\\data
```

## SpikeInterface Setup

Install `spikeinterface` and related tools using pip:
```
pip install spikeinterface[full] python-dotenv
```

## TO DO
- Update .yml to remove unused
- Update .yml to include SpikeInterface