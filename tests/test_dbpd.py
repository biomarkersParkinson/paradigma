import tsdf
import numpy as np
import pandas as pd
from dbpd.constants import DataColumns

from dbpd import dbpd

def test_transform_time_array(shared_datadir):
    metadata_dict = tsdf.load_metadata_from_path(shared_datadir / '1.sensor_data/ppg/PPG_meta.json')

    # Retrieve the metadata object we want, using the name of the binary as key
    metadata_samples = metadata_dict["PPG_time.bin"]
    data = tsdf.load_ndarray_from_binary(metadata_samples)
    assert data.shape == (64775,)