function tsdf = tsdf_scan_meta(tsdf_data_full_path)
% K.I. Veldkamp, PhD student AI4P, 29-02-24
% For each given TSDB directory, transcribe TSDF metadata contents to SQL
% table --> function specific for toolbox data structure
tsdf = [];
irow = 1;

meta_list = dir(fullfile(tsdf_data_full_path, '*_meta.json'));
meta_filenames = {meta_list.name};

jsonobj = {};
for n = 1:length(meta_filenames)
    tsdb_meta_fullpath = fullfile(tsdf_data_full_path, meta_filenames{n});
    jsonstr = fileread(tsdb_meta_fullpath);
    jsonobj{n} = loadjson(jsonstr);
    tsdf(irow).tsdf_meta_fullpath = tsdb_meta_fullpath;
    tsdf(irow).subject_id = jsonobj{n}.subject_id;
    tsdf(irow).start_iso8601 = jsonobj{n}.start_iso8601;
    tsdf(irow).end_iso8601 = jsonobj{n}.end_iso8601;
    irow = irow + 1;
end

