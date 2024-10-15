function WeeklyAggregates(ppp_pep_userid,week_vector)

% #########################################################################
% Input: 
% ppp_pep_userid    - id of the subject
% week_vector       - array of week numbers you want to calculate weekly
% aggregates for

% Output: Saves two binary files (Tremor_weeks.bin and Tremor_aggregates.bin) and a metadata file in TSDF format
% Saved in Tremor_weeks.bin per week:
%   - Week number
%   - Data availability flag
%   - Number of valid days
%   - Total number of windows available
%   - Number of windows available during daytime (08:00-22:00)
% Saved in Tremor_aggreagtes.bin per week:
%   Tremor time/ arm activity time measures:
%         - tremor proportion overall (24h)
%         - tremor proportion during daytime (08:00-22:00)
%         - tremor proportion overall and in rest (without arm activity)
%         - tremor proportion during daytime and in rest
%         - arm activity proportion overall
%         - arm activity proportion during daytime
%         - tremor proportion during daytime and non-rest (with arm
%         activity)
%         - tremor proportion during nighttime
%   Tremor amplitude and frequency measures (during daytime, based on dominant power 1.25
%   Hz around the tremor peak):
%         - median tremor amplitude 
%         - modus of tremor amplitude
%         - 90th percentile of tremor amplitude
%         - IQR of tremor amplitude
%         - median tremor amplitude in rest (without arm activity)
%         - modus of tremor amplitude in rest
%         - 90th percentile of tremor amplitude in rest
%         - IQR of tremor amplitude in rest
%         - median tremor frequency
%         - IQR of tremor frequency
%         - median tremor frequency in rest
%         - IQR of tremor frequency in rest                
% #########################################################################

NWeeks = length(week_vector);
unix_ticks_ms = 1000;
arm_activity_threshold = 10^0.88; % threshold for arm activity detection (based on 0.5-3 Hz band power)
tremor_amplitude_threshold = 0.02; % treshold for estimating the tremor amplitude (based on tremor time)
valid_day_threshold = 10; % threshold for valid day (based on number of hours/day)  

% Initialize
start_time_iso = [];
DataFlag = [];
total_number_windows = [];
number_windows_daytime = [];
tremor_proportion_24h = [];
tremor_proportion_daytime = [];
tremor_proportion_24h_rest = [];
tremor_proportion_daytime_rest = [];
arm_activity_proportion_24h = [];
arm_activity_proportion_daytime = [];
tremor_proportion_daytime_nonrest = [];
tremor_proportion_nighttime = [];
tremor_amplitude_median = [];
tremor_amplitude_modus = [];
tremor_amplitude_90th = [];
tremor_amplitude_IQR = [];
tremor_amplitude_median_rest = [];
tremor_amplitude_modus_rest  = [];
tremor_amplitude_90th_rest  = [];
tremor_amplitude_IQR_rest  = [];
tremor_frequency_median = [];
tremor_frequency_IQR = [];
tremor_frequency_median_rest = [];
tremor_frequency_IQR_rest = [];

%% In case you want to remove the day of first visit (because of OFF state)
% load('Visit1_day.mat')
% idnum = char(ppp_pep_userid);
% idnum = string(idnum(:,5:end));
% Visit1_date = Visit1_day.Date(ismember(Visit1_day.ID,idnum));
% Visit1_day_num = day(Visit1_date);

for i = 1:NWeeks
    
    subjectfolder = strcat('C:\Users\z835211\Documents\Data\TSDF output\WatchData.IMU.Week',num2str(week_vector(i)),'\',ppp_pep_userid);
    
    if isfolder(subjectfolder)

        DataFlag(i) = 1; % 1 if there is data for this week, 0 if there is no data (of enough good quality)

        % load tremor predictions and features for specific subject and week:
        [metadata, data] = load_tsdf_metadata_from_path(strcat(subjectfolder,'\Tremor_predictions_meta.json'));
        [~, data2] = load_tsdf_metadata_from_path(strcat(subjectfolder,'\Tremor_features_meta.json'));

        metafile_template = metadata{1};

        % extract the dates to determine the dates of valid days
        tremor_time = datetime(data{1,1}/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'Europe/Amsterdam');

        days = unique(day(tremor_time));

        Valid_days = [];

        for k = 1:length(days)

            NumWindows_daytime = length(find(day(tremor_time)==days(k) & ismember(hour(tremor_time),[8,9,10,11,12,13,14,15,16,17,18,19,20,21]))); % Find number of windows during daytime

            if NumWindows_daytime*4/3600 >= valid_day_threshold % Determine which days are valid days

                Valid_days = [Valid_days; days(k)];

            end
        end

        if ~isempty(Valid_days)
            
            % if i == 1
            %     Valid_days(Valid_days==Visit1_day_num) = []; % remove because of OFF-state around visit 1
            % end

            number_valid_days(i) = length(Valid_days);

            % Create several flags to indicate for every window whether
            % specific conditions are met:
            valid = ismember(day(tremor_time),Valid_days); % window belongs to a valid day
            tremor_valid = valid & data{1,2}(:,2)==1; % window belongs to a valid day and is classified as tremor
            daytime = ismember(hour(tremor_time),[8,9,10,11,12,13,14,15,16,17,18,19,20,21]); % window belongs to daytime
            daytime_valid = valid & daytime; % window belongs to daytime and to a valid day
            tremor_daytime_valid = daytime_valid & data{1,2}(:,2)==1; % window belongs to daytime and a valid day and is classified as tremor
            nighttime_valid = valid & ~daytime; % window belongs to nighttime and to a valid day
            tremor_nighttime_valid = nighttime_valid & data{1,2}(:,2)==1; % window belongs to nighttime and to a valid day and is classified as tremor

            rest_valid = data2{1,2}(:,17)<arm_activity_threshold & valid; % window belongs to a valid day and rest (no arm activity)
            tremor_rest_valid = rest_valid & data{1,2}(:,2)==1; % window belongs to a valid day and rest and is classified as tremor
            nonrest_valid = data2{1,2}(:,17)>=arm_activity_threshold & valid; % window belongs to a valid day and non-rest (with arm activity)
            daytime_rest_valid = daytime & rest_valid; % window belongs to daytime, a valid day and rest
            tremor_daytime_rest_valid = daytime_rest_valid & data{1,2}(:,2)==1; % window belongs to daytime, a valid day and rest and is classified as tremor
            daytime_nonrest_valid= daytime & nonrest_valid; % window belongs to daytime, a valid day and non-rest
            tremor_daytime_nonrest_valid = daytime_nonrest_valid & data{1,2}(:,2)==1; % window belongs to daytime, a valid day and non-rest and is classified as tremor

            total_number_windows(i) = length(find(valid==1)); % total number of windows available during valid days
            number_windows_daytime(i) = length(find(daytime_valid==1)); % total number of windows available during daytime on valid days
            
            % Calculate the weekly aggregated measures for tremor time, arm
            % activity and tremor frequency:
            tremor_proportion_24h(i) = length(find(tremor_valid==1))/length(find(valid==1));
            tremor_proportion_24h_rest(i) = length(find(tremor_rest_valid==1))/length(find(rest_valid==1));
            arm_activity_proportion_24h(i) = length(find(nonrest_valid==1))/length(find(valid==1));

            tremor_proportion_daytime(i) = length(find(tremor_daytime_valid==1))/length(find(daytime_valid==1));
            tremor_proportion_daytime_rest(i) = length(find(tremor_daytime_rest_valid==1))/length(find(daytime_rest_valid==1));
            arm_activity_proportion_daytime(i) = length(find(daytime_nonrest_valid==1))/length(find(daytime_valid==1));

            tremor_proportion_daytime_nonrest(i) = length(find(tremor_daytime_nonrest_valid==1))/length(find(daytime_nonrest_valid==1));
            tremor_proportion_nighttime(i) = length(find(tremor_nighttime_valid==1))/length(find(nighttime_valid==1));

            tremor_frequency_median(i) = median(data2{1,2}(tremor_daytime_valid,10));
            tremor_frequency_IQR(i) = iqr(data2{1,2}(tremor_daytime_valid,10));
            tremor_frequency_median_rest(i) = median(data2{1,2}(tremor_daytime_rest_valid,10));
            tremor_frequency_IQR_rest(i) = iqr(data2{1,2}(tremor_daytime_rest_valid,10));

            if tremor_proportion_daytime(i)>=tremor_amplitude_threshold % determine whether there is enough tremor detected to calculate the amplitude
                tremor_amplitude_median(i) = median(log10(1+data2{1,2}(tremor_daytime_valid,13)));
                tremor_amplitude_90th(i) = prctile(log10(1+data2{1,2}(tremor_daytime_valid,13)),90);
                tremor_amplitude_IQR(i) = iqr(log10(1+data2{1,2}(tremor_daytime_valid,13)));
                % determine the modus:
                binEdges = linspace(0,8,41); % Define the edges of the bins
                Histogram = histcounts(log10(1+data2{1,2}(tremor_daytime_valid,13)), binEdges); 
                [~, maxIndex] = max(Histogram);
                tremor_amplitude_modus(i) = mean([binEdges(maxIndex), binEdges(maxIndex + 1)]);

                tremor_amplitude_median_rest(i) = median(log10(1+data2{1,2}(tremor_daytime_rest_valid,13)));
                tremor_amplitude_90th_rest(i) = prctile(log10(1+data2{1,2}(tremor_daytime_rest_valid,13)),90);
                tremor_amplitude_IQR_rest(i) = iqr(log10(1+data2{1,2}(tremor_daytime_rest_valid,13)));
                % determine the modus:
                binEdges = linspace(0,8,41); % Define the edges of the bins
                Histogram = histcounts(log10(1+data2{1,2}(tremor_daytime_rest_valid,13)), binEdges); 
                [~, maxIndex] = max(Histogram);
                tremor_amplitude_modus_rest(i) = mean([binEdges(maxIndex), binEdges(maxIndex + 1)]);
            else % tremor amplitude is not calculated
                tremor_amplitude_median(i) = NaN;
                tremor_amplitude_modus(i) = NaN;
                tremor_amplitude_90th(i) = NaN;
                tremor_amplitude_IQR(i) = NaN;
                tremor_amplitude_median_rest(i) = NaN;
                tremor_amplitude_modus_rest(i) = NaN;
                tremor_amplitude_90th_rest(i) = NaN;
                tremor_amplitude_IQR_rest(i) = NaN;
            end
        else % there are no valid days for this week
            number_valid_days(i) = 0;
            total_number_windows(i) = 0;
            number_windows_daytime(i) = 0;
            tremor_proportion_24h(i) = NaN;
            tremor_proportion_24h_rest(i) = NaN;
            arm_activity_proportion_24h(i) = NaN;
            tremor_proportion_daytime(i) = NaN;
            tremor_proportion_daytime_rest(i) = NaN;
            arm_activity_proportion_daytime(i) = NaN;
            tremor_proportion_daytime_nonrest(i) = NaN;
            tremor_proportion_nighttime(i) = NaN;
            tremor_amplitude_median(i) = NaN;
            tremor_amplitude_modus(i) = NaN;
            tremor_amplitude_90th(i) = NaN;
            tremor_amplitude_IQR(i) = NaN;
            tremor_amplitude_median_rest(i) = NaN;
            tremor_amplitude_modus_rest(i) = NaN;
            tremor_amplitude_90th_rest(i) = NaN;
            tremor_amplitude_IQR_rest(i) = NaN;
            tremor_frequency_median(i) = NaN;
            tremor_frequency_IQR(i) = NaN;
            tremor_frequency_median_rest(i) = NaN;
            tremor_frequency_IQR_rest(i) = NaN;
        end

        % Determine the start time of the first week
        if isempty(start_time_iso) 
            start_time_iso = datetime(data{1,1}(1)/unix_ticks_ms,"ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy Z', 'TimeZone', 'Europe/Amsterdam');
        end

    else % theres is no data available for this week
        DataFlag(i) = 0;
        number_valid_days(i) = NaN;
        total_number_windows(i) = NaN;
        number_windows_daytime(i) = NaN;
        tremor_proportion_24h(i) = NaN;
        tremor_proportion_24h_rest(i) = NaN;
        arm_activity_proportion_24h(i) = NaN;
        tremor_proportion_daytime(i) = NaN;
        tremor_proportion_daytime_rest(i) = NaN;
        arm_activity_proportion_daytime(i) = NaN;
        tremor_proportion_daytime_nonrest(i) = NaN;
        tremor_proportion_nighttime(i) = NaN;
        tremor_amplitude_median(i) = NaN;
        tremor_amplitude_modus(i) = NaN;
        tremor_amplitude_90th(i) = NaN;
        tremor_amplitude_IQR(i) = NaN;
        tremor_amplitude_median_rest(i) = NaN;
        tremor_amplitude_modus_rest(i) = NaN;
        tremor_amplitude_90th_rest(i) = NaN;
        tremor_amplitude_IQR_rest(i) = NaN;
        tremor_frequency_median(i) = NaN;
        tremor_frequency_IQR(i) = NaN;
        tremor_frequency_median_rest(i) = NaN;
        tremor_frequency_IQR_rest(i) = NaN;
    end
end

% Determine the endtime of the last week:
end_time_iso = datetime(data{1,1}(end)/unix_ticks_ms,"ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy Z', 'TimeZone', 'Europe/Amsterdam');

% Save output in TSDF:

metafile_template.ppp_source_protobuf = [];
metafile_template.freq_sampling_original = [];
metafile_template.freq_sampling_adjusted = [];

metafile_template.start_iso8601 = start_time_iso;
metafile_template.end_iso8601 = end_time_iso;

metafile_template.arm_activity_threshold = arm_activity_threshold;
metafile_template.amplitude_estimation_threshold = tremor_amplitude_threshold;

data_tsdf{1} = [int64(week_vector)' int64(DataFlag)' int64(number_valid_days)' int64(total_number_windows)' int64(number_windows_daytime)'];
data_tsdf{2} = [tremor_proportion_24h' tremor_proportion_24h_rest' arm_activity_proportion_24h' tremor_proportion_daytime' tremor_proportion_daytime_rest'...
    arm_activity_proportion_daytime' tremor_proportion_daytime_nonrest' tremor_proportion_nighttime' tremor_amplitude_median' tremor_amplitude_modus' tremor_amplitude_90th' tremor_amplitude_IQR' tremor_amplitude_median_rest'...
    tremor_amplitude_modus_rest' tremor_amplitude_90th_rest' tremor_amplitude_IQR_rest' tremor_frequency_median' tremor_frequency_IQR' tremor_frequency_median_rest'...
    tremor_frequency_IQR_rest'];

metafile_weeks  = metafile_template;
metafile_aggregates  = metafile_template;

metafile_weeks.channels = {'week number','data availability','number of valid days','total number of windows','number of windows daytime'};
metafile_weeks.units = {'','boolean_num','','',''};
metafile_weeks.file_name = 'Tremor_weeks.bin';

metafile_aggregates.channels = {'tremor proportion','tremor proportion in rest','arm activity proportion','daytime tremor proportion',...
    'daytime tremor proportion in rest','daytime arm activity proportion', 'daytime tremor proportion during arm activity','nighttime tremor proportion','median tremor amplitude','modus tremor amplitude',...
    'tremor amplitude 90th cent','tremor amplitude IQR','median tremor amplitude in rest','modus tremor amplitude in rest',...
    'tremor amplitude 90th cent in rest','tremor amplitude IQR in rest','median tremor frequency','tremor frequency IQR',...
    'median tremor frequency in rest','tremor frequency IQR in rest'};
metafile_aggregates.units = {'proportion','proportion','proportion','proportion','proportion','proportion','proportion','proportion','','','','','','','','','Hz','Hz','Hz','Hz'};
metafile_aggregates.file_name = 'Tremor_aggregates.bin';

meta_tsdf{1} = metafile_weeks;
meta_tsdf{2} = metafile_aggregates;

metadata_file_name = "Tremor_weekly_aggregates_meta.json";
location = ['C:\Users\z835211\Documents\Data\TSDF output\Weekly aggregates\',ppp_pep_userid, '\'];
mkdir(['C:\Users\z835211\Documents\Data\TSDF output\Weekly aggregates\',ppp_pep_userid]);

save_tsdf_data(meta_tsdf, data_tsdf, location, metadata_file_name);

clear all

end
