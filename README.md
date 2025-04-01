# omniroute_analysis
Shared code base for Omniroute experiment analysis

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

## SpikeInterface Setup

Install `spikeinterface` and related tools using pip:
```
pip install spikeinterface[full] python-dotenv
```

Create a `omniroute_analysis.env` file in the repo root with the following variables:
```
TRODES_PATH=
NC4_DATA_PATH=
```

For example:
```
TRODES_PATH=C:\\Users\\lester\\MeDocuments\\Research\\MadhavLab\\Projects\\SpikeGadgets\\Trodes_2-3-4_Windows64\\Trodes.exe
NC4_DATA_PATH=C:\\Users\\lester\\UBC\\Madhav, Manu - lesterkaur2024gate\\analysis\\gate_ephys_test\\data
```
