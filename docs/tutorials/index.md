## Tutorials

This section contains step-by-step tutorials demonstrating how to use the ParaDigMa toolbox to extract digital biomarkers from wrist-worn sensor data.

---

**Quick Start**

- [Pipeline Orchestrator](pipeline_orchestrator) - End-to-end analysis example

**Data Loading and Preparation**

- [Device-specific data loading](device_specific_data_loading) - Load data from Empatica, Axivity, Verily
- [Data preparation](data_preparation) - Clean, format, and prepare sensor data

**Pipeline-Specific Tutorials**

- [Gait analysis](gait_analysis) - Extract arm swing measures during walking
- [Tremor analysis](tremor_analysis) - Extract tremor measures
- [Pulse rate analysis](pulse_rate_analysis) - Extract pulse rate measures

---

**New to ParaDigMa?** Start with [Pipeline Orchestrator](pipeline_orchestrator) for a complete end-to-end example, or begin with [Data Preparation](data_preparation) if you need help loading your own sensor data.

### Running Tutorials in Jupyter

If you want to run the tutorial notebooks in Jupyter Notebook or JupyterLab:

**Option 1: Activate Poetry environment first (Recommended)**
```bash
poetry env activate
jupyter lab
```

**Option 2: Register Poetry kernel with Jupyter (for VSC, PyCharm, etc.)**
```bash
# Register the Poetry environment as a Jupyter kernel
poetry run python -m ipykernel install --user --name paradigma --display-name "Python (paradigma)"
```

After registering the kernel, select "Python (paradigma)" as the kernel when opening notebooks in your IDE or Jupyter interface.

**Option 3: Use Poetry to run Jupyter directly**
```bash
poetry run jupyter lab
```
