import tsdf

def test_transform_time_array(shared_datadir):
    """
    The initial test to check if the preprocessing function works as expected. It checks the output dimensions and the type of the output.
    """
    metadata_dict = tsdf.load_metadata_from_path(shared_datadir / '1.sensor_data/ppg/PPG_meta.json')

    # Retrieve the metadata object we want, using the name of the binary as key
    metadata_values = metadata_dict["PPG_time.bin"]
    data = tsdf.load_ndarray_from_binary(metadata_values)
    assert data.shape == (64775,)
