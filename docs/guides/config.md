# Config
Throughout the ParaDigMa toolbox we use configuration objects `Config` to specificy parameters used as input of processes. All configuration classes care defined in `config.py`, and can be imported using `from paradigma.config import X`.  The configuration classes frequently use static column names defined in `constants.py` to ensure robustness and consistency.

## Config classes
The config classes are defined either for sensors (IMU, PPG) or for domains (gait, heart rate, tremor).

### Sensor configs
There are two sensor config classes: `IMUConfig` and `PPGConfig`.

### Domain configs
For the latter category, config classes can further be distinguished by the processing steps displayed [here](https://github.com/biomarkersParkinson/paradigma).
