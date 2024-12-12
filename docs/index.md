```{include} 
../README.md
```

```{toctree}
:maxdepth: 2
:caption: Project structure
project_structure.md
```

```{toctree}
:maxdepth: 2
:caption: Learn
Prepare your data <data_preparation.md>
Data preprocessing <notebooks/preprocessing/preprocessing.ipynb>
    ```{toctree}
    :maxdepth: 1
    Preprocess IMU data <notebooks/preprocessing/preprocessing_imu.ipynb>
    Preprocess PPG data <notebooks/preprocessing/preprocessing_ppg.ipynb>
Gait & Arm swing <notebooks/gait/arm_swing_pipeline.ipynb>
    ```{toctree}
    :maxdepth: 1
    Extract gait features <notebooks/gait/extract_gait_features.ipynb>
    Detect gait <notebooks/gait/detect_gait.ipynb>
    Extract arm activity features <notebooks/gait/extract_arm_activity_features.ipynb>
    Detect gait without other arm activities <notebooks/gait/detect_gait_without_other_arm_activities.ipynb>
    Quantify arm swing <notebooks/gait/quantify_arm_swing.ipynb>
Tremor <notebooks/tremor/tremor_pipeline.ipynb>
Heart Rate <notebooks/heart_rate/heart_rate_pipeline.ipynb>
```

```{toctree}
:maxdepth: 2
:caption: API
autoapi/index
```

```{toctree}
:maxdepth: 2
:caption: TSDF schema
tsdf_paradigma_schemas.md
tsdf_paradigma_channels_and_units.md
```

```{toctree}
:maxdepth: 2
:caption: Development
changelog.md
contributing.md
conduct.md
```
