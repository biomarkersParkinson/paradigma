%% Code for writing TSDF output for the tremor pipeline on a test subject from PPP 
% Diogo C. Soriano and Nienke Timmermans
% 20/02/2024

clear all; 
close all;

%% Test subject
ppp_pep_userid = '0A0B82C94960D6DCABC1F597EC0BA657F4B0EDC320702BCEE3B6955CE924DE05'; % test subject for eScience
week_id = 104;

% Set here data location
% Nienke:
% location = "D:\ppp_data\TSDF output\Test"; % location for storing tsdf output
% Diogo:
location = 'Z:\Diogo\Snellius Pipeline\Vedran - CheckPoint\Rest-Tremor---PPP-library\Test';

%% Loading the data
addpath(genpath('tsdf4matlab')); % Add tsdf package
addpath(genpath('functions_eScience')); % Add all functions in the subfolder
addpath(genpath('Model')); % Add subfolder with the classifier structures
addpath(genpath('jsonlab')); % Add subfolder with functions to load json file

% Uncomment for obtaining arm activity classifier based on MFCCs trained
% with PD@Home labels

% addpath(genpath('Arm activity classifier')); % Add subfolder with the classifier structures
% load('ArmClass.mat'); % Load Arm Activity Classifier

unix_ticks_ms    = 1000.0;

% Set your path here
% ppp_data_path      = ['D:\ppp_data\New Conversion\WatchData.IMU.Week',num2str(week_id)]; % IMU data path
ppp_data_path      = ['Z:\Diogo\PPP Sample Data\data\SampleData - eScienceTest\WatchData.IMU.Week',num2str(week_id)]; % IMU data path
meta_segments_list = dir(fullfile(ppp_data_path, ppp_pep_userid, ['WatchData.IMU.Week',num2str(week_id),'.raw_segment*_meta.json'])); % create segment list
meta_filenames = {meta_segments_list.name}; % get names
Nfiles         = length(meta_filenames); % get number of files

bad_quality_threshold_imu = 0.3;

shift_treshold = 300;       % Set shift threshold for determining a segment also as bad quality, threshold is in seconds

[imu_indices_bq, ~] = bad_quality(strcat(ppp_data_path,'\',ppp_pep_userid), Nfiles, bad_quality_threshold_imu, shift_treshold);

meta_filenames(imu_indices_bq) = [];      % remove bad quality segments from further analysis by removing them in the metafile list which is needed for further analyses

if ~isempty(meta_filenames) % check if there are good quality segments left

    Nfiles         = length(meta_filenames); % get number of files

    t_imu = {};
    v_imu = {};
    scale_factors = [];

    for n = 1:Nfiles
        meta_fullpath = fullfile(ppp_data_path, ppp_pep_userid, '/', meta_filenames{n});
        [metadata_list, data_list] = load_tsdf_metadata_from_path(meta_fullpath);
        time_idx = tsdf_values_idx(metadata_list, 'time', week_id);
        values_idx = tsdf_values_idx(metadata_list, 'samples', week_id);

        t_iso = metadata_list{time_idx}.start_iso8601;
        date_time_obj= datetime(t_iso, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'UTC');
        t_diff_imu{n} = data_list{time_idx};
        ts_imu(n) = posixtime(date_time_obj) * unix_ticks_ms;      % calculate the unix timestamp in ms
        t_imu{n} = cumsum(double(data_list{time_idx})) + ts_imu(n);
        tr_imu{n} = (t_imu{n}-ts_imu(n))/unix_ticks_ms;
        fs_est(n) = 1/(mean(diff(tr_imu{n})));      % Check the sampling frequency for each file - inverse of the mean of sampling interval

        v_imu{n} = data_list{values_idx};             % store data values for every seperate tsdf file in cell
        scale_factors(n,:) = metadata_list{values_idx}.scale_factors';
        clear data_list;
    end

    %% Preprocessing and feature extraction
    Fs_aprox = round(mean(fs_est));                    % Approximate sampling frequency considering all files
    fsvector = [50 100];                               % reference sampling rate
    [~,ind] = min(abs(Fs_aprox-fsvector));             % Check which sampling rate is closer to the approximate sampling rate obtained
    Fs = fsvector(ind);                                % Stablish a sampling rate from the reference which is closest to the approximate sampling rate

    % Allocating memory
    Derivative   = []; % Mean absolute derivative values [x,y,z]
    PowerAxis    = []; % Power within the range [0.5 - 25] Hz 
    DomTremorPowerAxis = []; % Dominant Power in the tremor range for each axis 
    PSDTremorFeatures  = [];
    PSDHighTremorFeatures = [];
    MelCepsCoeff    = [];
    FreqPeak        = [];
    WinDateTime     = [];
    PowerArmActv  = []; 
    PowerHighFreq = []; 

    TremorProb = [];
    TremorHat  = [];
    RestTremorHat     = [];
  
    v_imu_proc = {};
    t_imu_proc = {};

    for n = 1:Nfiles
        
        y_curr_gyro  = scale_factors(n,4:6).*double(v_imu{n}(:,4:6)); % Get the gyroscope signals scaled by the respective scaling factors
        
        % Interpolate the data - sampling frequency adjustment
        [t_imu_proc{n}, v_imu_proc{n}] = InterpData(tr_imu{n},y_curr_gyro,Fs,unix_ticks_ms,ts_imu(n)); % pre-processing: interpolate data

        % Save pre-processed data in TSDF
        procdata_tsdf{1} = int64(t_imu_proc{n});
        procdata_tsdf{2} = v_imu_proc{n};

        metafile_template = metadata_list{1};

        start_time_iso = datetime(t_imu_proc{n}(1)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'UTC');
        end_time_iso   = datetime(t_imu_proc{n}(end)/unix_ticks_ms+1, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'UTC');

        metafile_template.start_iso8601 = datestr(start_time_iso);
        metafile_template.end_iso8601 = datestr(end_time_iso);
        metafile_template.freq_sampling_original = Fs_aprox;
        metafile_template.freq_sampling_adjusted = Fs;

        metafile_proctime = metafile_template;
        metafile_procgyro = metafile_template;

        metafile_proctime.channels = {'time'};
        metafile_proctime.units = {'time_absolute_unix_ms'};
        metafile_proctime.file_name = 'Tremor_preprocessed_time.bin';

        metafile_procgyro.channels = {'rotation_x','rotation_y','rotation_z'};
        metafile_procgyro.units = {'deg/s','deg/s','deg/s'};
        metafile_procgyro.file_name = 'Tremor_preprocessed_gyro.bin';

        procmeta_tsdf{1} = metafile_proctime;
        procmeta_tsdf{2} = metafile_procgyro;

        mat_metadata_file_name = "Tremor_preprocessed_data.json";
        save_tsdf_data(procmeta_tsdf, procdata_tsdf, location, mat_metadata_file_name);

        %% Extracting the features
        % General parameter setting for Configuration structure for feature
        % extraction. Other "fixed" feature extraction parameters are set
        % within TremorFeature function

        Conf.WindowSizeTime = 4; % Window size of 4 seconds
        Conf.Fs = Fs;            % Sampling frequency
        TypeOfSignal     = 'Gy'; % Gyroscope signals

        tic
        Features_per_segment = TremorFeaturesAndClassification(Conf,v_imu_proc{n},t_imu_proc{n},TypeOfSignal); % Calculate features
        elapsed = toc
        % Concatenate features
        if istable(Features_per_segment.PSDTremorFeatures)
            
            WinDateTime          = vertcat(WinDateTime, Features_per_segment.WindowIniTime);

            Derivative  = vertcat(Derivative, Features_per_segment.Derivatives);
            PowerAxis   = vertcat(PowerAxis,Features_per_segment.PowerAxis);
            DomTremorPowerAxis = vertcat(DomTremorPowerAxis,Features_per_segment.DomTremorPowerAxis);
            FreqPeak           = vertcat(FreqPeak, Features_per_segment.FreqPeak);
            PSDTremorFeatures  = vertcat(PSDTremorFeatures, Features_per_segment.PSDTremorFeatures);
            PSDHighTremorFeatures = vertcat(PSDHighTremorFeatures, Features_per_segment.PSDHighTremorFeatures);
            PowerArmActv  = vertcat(PowerArmActv,Features_per_segment.PowerArmActv);
            PowerHighFreq = vertcat(PowerHighFreq,Features_per_segment.PowerHighFreq);
            MelCepsCoeff      = vertcat(MelCepsCoeff, Features_per_segment.MelCepsCoeff);
                     
            % Get Tremor Classification Variables for tsdf writting
            TremorProb = vertcat(TremorProb,Features_per_segment.TremorProb);
            TremorHat  = vertcat(TremorHat,Features_per_segment.TremorHat);
            RestTremorHat     = vertcat(RestTremorHat,Features_per_segment.RestTremorHat);
           
        end
    end

    % Features that need to be saved for further analysis

    FeaturesTremor = [Derivative PowerAxis DomTremorPowerAxis FreqPeak ...
        PSDTremorFeatures PSDHighTremorFeatures PowerArmActv PowerHighFreq MelCepsCoeff];            

    %% Write TSDF output
    
    % Save data
    data_tsdf{1} = int64(WinDateTime);
    data_tsdf{2} = TremorProb;
    data_tsdf{3} = [int64(TremorHat) int64(RestTremorHat)];
    data_tsdf{4} = FeaturesTremor.Variables;

    % Save metadata
    metafile_template = metadata_list{1};

    start_time_iso = datetime(WinDateTime(1)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'UTC');
    end_time_iso   = datetime(WinDateTime(end)/unix_ticks_ms+1, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'UTC');

    metafile_template.start_iso8601 = datestr(start_time_iso);
    metafile_template.end_iso8601 = datestr(end_time_iso);
    metafile_template.freq_sampling_original = Fs_aprox;
    metafile_template.freq_sampling_adjusted = Fs;

    metafile_time  = metafile_template;
    metafile_prob  = metafile_template;
    metafile_label = metafile_template;
    metafile_features = metafile_template;
    metafile_imu_proc = metafile_template;

    metafile_time.channels = {'time'};
    metafile_time.units = {'time_absolute_unix_ms'};
    metafile_time.file_name = 'Tremor_time.bin';

    metafile_prob.channels = {'tremor probability'};
    metafile_prob.units = {'probability'};
    metafile_prob.file_name = 'Tremor_prob.bin';

    metafile_label.channels = {'tremor label', 'rest tremor label'};
    metafile_label.units = {'boolean','boolean'};
    metafile_label.file_name = 'Tremor_label.bin';
 
    metafile_features.channels = FeaturesTremor.Properties.VariableNames';
    metafile_features.units = {'deg/s^2','deg/s^2','deg/s^2','(deg/s)^2','(deg/s)^2','(deg/s)^2','(deg/s)^2','(deg/s)^2','(deg/s)^2',...
    'Hz','(deg/s)^2','Hz','(deg/s)^2','(deg/s)^2','Hz','(deg/s)^2','','','','','','','','','','','','','(deg/s)^2','(deg/s)^2'};
    metafile_features.file_name = 'Tremor_features.bin';

    % Test plot
    TimeVector = datetime(WinDateTime,'ConvertFrom','epochtime',...
        'TicksPerSecond',1e3,'Format','dd-MMM-yyyy HH:mm:ss');

    figure;
    plot(TimeVector,RestTremorHat);
    xlabel('Date Time');
    ylabel('Rest Tremor Label');
    title('Rest tremor time-course');
    set(gca,'fontsize',18,'fontweight','bold','linewidth',2);

    meta_tsdf{1} = metafile_time;
    meta_tsdf{2} = metafile_prob;
    meta_tsdf{3} = metafile_label;
    meta_tsdf{4} = metafile_features;

    mat_metadata_file_name = "Tremor_features_meta.json";
    save_tsdf_data({meta_tsdf{1,[1,4]}}, {data_tsdf{1,[1,4]}}, location, mat_metadata_file_name);

    mat_metadata_file_name = "Tremor_predictions_meta.json";
    save_tsdf_data({meta_tsdf{1,[1,2,3]}}, {data_tsdf{1,[1,2,3]}}, location, mat_metadata_file_name);

    % Saving the matlab struct as a check
    PPP.ID = ppp_pep_userid;
    PPP.Week = week_id;
    PPP.Features = FeaturesTremor;
    PPP.WinDateTime = WinDateTime;
    PPP.ArmActvPower = FeaturesTremor.GyArmActvPower;
    PPP.TremorProb = TremorProb;
    PPP.TremorHat  = TremorHat;
    PPP.RestTremorHat = RestTremorHat;
    
    % Set here your location
    % save('Z:\Diogo\Snellius Pipeline\Vedran - CheckPoint\Rest-Tremor---PPP-library\Test\FeatureVector','PPP');

end
