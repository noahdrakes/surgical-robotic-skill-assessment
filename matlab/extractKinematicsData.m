%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract Kinetics and Kinematics Data from ROS Bag Files
%
% Description:
%   Extracts kinetics and kinematics data streams from
%   ROS bag recordings for downstream analysis.
%
% Inputs:
%   - rosbag file(s) containing recorded sensor and robotic data
%
% Features:
%   - Handles multiple data streams and sensor types, including:
%       * geometry_msgs/AccelStamped
%       * geometry_msgs/WrenchStamped
%       * geometry_msgs/PoseStamped
%       * geometry_msgs/TwistStamped
%       * sensor_msgs/Joy
%       * sensor_msgs/JointState
%       * std_msgs/Empty
%
% Author: Sergio Machaca
% Contact: smachac2@jh.edu
% Created: 04 June 2025
% Last Updated: 27 June 2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setup

clear; clc; clear shouldWrite;

% Select directory
bagDir = uigetdir(pwd, 'Select folder containing bag files');
if bagDir == 0
    error('No folder selected. Exiting...');
end
cd(bagDir);

% Create output directory one level up from bagDir
parquetRootDir = fullfile(fileparts(bagDir), 'parquet');
if ~exist(parquetRootDir, 'dir')
    mkdir(parquetRootDir);
    fprintf('Created output root folder: %s\n', parquetRootDir);
else
    fprintf('Saving output to: %s\n', parquetRootDir);
end

% Get list of bag files
bagFiles = dir('*.bag');

% Initialize skip and overwrite settings
overwriteAll = false;
skipAll = false;

%% Main loop

% Loop through bag files
for i = 1:length(bagFiles)
    bagFileName = bagFiles(i).name;
    fprintf('Processing bag file: %s\n', bagFileName);

    bag = rosbag(bagFileName);
    folderName = sprintf('T%02d', i);

    if isprop(bag, 'AvailableTopics') && istable(bag.AvailableTopics)
        topicInfo = bag.AvailableTopics;
        topicList = topicInfo.Properties.RowNames;
    else
        warning('No available topics in %s\n', bagFileName);
        continue;
    end

    for t = 1:length(topicList)
        topicName = topicList{t};
        shortName = erase(topicName, '/');

        % Skip video/image-related topics
        if contains(topicName, 'decklink', 'IgnoreCase', true) || ...
           contains(topicName, 'image_raw', 'IgnoreCase', true) || ...
           contains(topicName, 'compressed', 'IgnoreCase', true) || ...
           contains(topicName, 'theora', 'IgnoreCase', true) || ...
           contains(topicName, 'camera_info', 'IgnoreCase', true)
            continue;
        end

        outDir = fullfile(parquetRootDir, folderName);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end

        outFile = fullfile(outDir, [shortName '.parquet']);
        if ~shouldWrite(outFile)
            continue;
        end

        % Get message type
        msgType = char(topicInfo.MessageType(t));

        % Extract based on message type
        switch msgType
            case 'geometry_msgs/AccelStamped'
                extractAndWriteAccel(bag, topicName, outFile, bagFileName);
            case 'geometry_msgs/WrenchStamped'
                extractAndWriteWrench(bag, topicName, outFile, bagFileName);
            case 'geometry_msgs/PoseStamped'
                extractAndWritePose(bag, topicName, outFile, bagFileName);
            case 'geometry_msgs/TwistStamped'
                extractAndWriteTwist(bag, topicName, outFile, bagFileName);
            case 'sensor_msgs/Joy'
                extractAndWriteJoy(bag, topicName, outFile, bagFileName);
            case 'sensor_msgs/JointState'
                extractAndWriteJointState(bag, topicName, outFile, bagFileName);
            case 'std_msgs/Empty'
                extractAndWriteEmpty(bag, topicName, outFile, bagFileName);
            otherwise
                % Unknown message type
        end
    end
end


%% Helper functions

function tf = shouldWrite(outFile)
    persistent overwriteAllLocal skipAllLocal
    tf = true;

    if isempty(overwriteAllLocal), overwriteAllLocal = false; end
    if isempty(skipAllLocal), skipAllLocal = false; end

    if exist(outFile, 'file')
        if skipAllLocal
            tf = false;
            return;
        elseif ~overwriteAllLocal
            fprintf('File %s already exists.\n', outFile);
            answer = input('Overwrite? (y)es / (Y)es to all / (n)o / (N)o to all: ', 's');
            switch answer
                case 'y'
                    tf = true;
                case 'Y'
                    overwriteAllLocal = true;
                    tf = true;
                case 'n'
                    tf = false;
                case 'N'
                    skipAllLocal = true;
                    tf = false;
                otherwise
                    fprintf('Invalid input. Skipping.\n');
                    tf = false;
            end
        end
    end
end


%% Extraction functions

function extractAndWriteAccel(bag, topicName, outFile, bagFileName)
    msgs = readMessages(select(bag, 'Topic', topicName), 'DataFormat', 'struct');
    n = length(msgs);

    header_seq = nan(n,1);
    header_stamp_sec = nan(n,1);
    header_stamp_nsec = nan(n,1);
    header_frame_id = strings(n,1);
    linear_accel_x = nan(n,1);
    linear_accel_y = nan(n,1);
    linear_accel_z = nan(n,1);
    angular_accel_x = nan(n,1);
    angular_accel_y = nan(n,1);
    angular_accel_z = nan(n,1);
    bag_file_col = repmat(string(bagFileName), n, 1);

    for j = 1:n
        header_seq(j) = msgs{j}.Header.Seq;
        header_stamp_sec(j) = msgs{j}.Header.Stamp.Sec;
        header_stamp_nsec(j) = msgs{j}.Header.Stamp.Nsec;
        header_frame_id(j) = string(msgs{j}.Header.FrameId);
        linear_accel_x(j) = msgs{j}.Accel.Linear.X;
        linear_accel_y(j) = msgs{j}.Accel.Linear.Y;
        linear_accel_z(j) = msgs{j}.Accel.Linear.Z;
        angular_accel_x(j) = msgs{j}.Accel.Angular.X;
        angular_accel_y(j) = msgs{j}.Accel.Angular.Y;
        angular_accel_z(j) = msgs{j}.Accel.Angular.Z;
    end

    % Compute timestamps in seconds
    timeSec = double(header_stamp_sec) + double(header_stamp_nsec) * 1e-9;

    % Estimate sample rate (Hz)
    if length(timeSec) > 1
        dt = diff(timeSec);
        mean_dt = mean(dt, 'omitnan');
        sample_rate = 1 / mean_dt;
    else
        sample_rate = NaN;
    end

    % Add a constant column for sample rate
    sample_rate_col = repmat(sample_rate, n, 1);

    % Construct table
    T = table(header_seq, header_stamp_sec, header_stamp_nsec, header_frame_id, ...
        linear_accel_x, linear_accel_y, linear_accel_z, ...
        angular_accel_x, angular_accel_y, angular_accel_z, ...
        bag_file_col, sample_rate_col);

    T.Properties.VariableNames{end-1} = 'BagFileName';
    T.Properties.VariableNames{end} = 'SampleRateHz';

    % Write to Parquet
    parquetwrite(outFile, T);
    fprintf('Saved %s data to %s (Sample Rate: %.2f Hz)\n', topicName, outFile, sample_rate);
end

function extractAndWriteWrench(bag, topicName, outFile, bagFileName)
    msgs = readMessages(select(bag, 'Topic', topicName), 'DataFormat', 'struct');
    n = length(msgs);

    header_seq = nan(n,1);
    header_stamp_sec = nan(n,1);
    header_stamp_nsec = nan(n,1);
    header_frame_id = strings(n,1);
    force_x = nan(n,1);
    force_y = nan(n,1);
    force_z = nan(n,1);
    torque_x = nan(n,1);
    torque_y = nan(n,1);
    torque_z = nan(n,1);
    bag_file_col = repmat(string(bagFileName), n, 1);

    for j = 1:n
        header_seq(j) = msgs{j}.Header.Seq;
        header_stamp_sec(j) = msgs{j}.Header.Stamp.Sec;
        header_stamp_nsec(j) = msgs{j}.Header.Stamp.Nsec;
        header_frame_id(j) = string(msgs{j}.Header.FrameId);
        force_x(j) = msgs{j}.Wrench.Force.X;
        force_y(j) = msgs{j}.Wrench.Force.Y;
        force_z(j) = msgs{j}.Wrench.Force.Z;
        torque_x(j) = msgs{j}.Wrench.Torque.X;
        torque_y(j) = msgs{j}.Wrench.Torque.Y;
        torque_z(j) = msgs{j}.Wrench.Torque.Z;
    end

    % Compute timestamps in seconds
    timeSec = double(header_stamp_sec) + double(header_stamp_nsec) * 1e-9;

    % Estimate sample rate (Hz)
    if length(timeSec) > 1
        dt = diff(timeSec);
        mean_dt = mean(dt, 'omitnan');
        sample_rate = 1 / mean_dt;
    else
        sample_rate = NaN;
    end

    % Add a constant column for sample rate
    sample_rate_col = repmat(sample_rate, n, 1);

    % Construct output table
    T = table(header_seq, header_stamp_sec, header_stamp_nsec, header_frame_id, ...
        force_x, force_y, force_z, torque_x, torque_y, torque_z, ...
        bag_file_col, sample_rate_col);

    T.Properties.VariableNames{end-1} = 'BagFileName';
    T.Properties.VariableNames{end} = 'SampleRateHz';

    parquetwrite(outFile, T);
    fprintf('Saved %s data to %s (Sample Rate: %.2f Hz)\n', topicName, outFile, sample_rate);
end

function extractAndWritePose(bag, topicName, outFile, bagFileName)
    msgs = readMessages(select(bag, 'Topic', topicName), 'DataFormat', 'struct');
    n = length(msgs);

    header_seq = nan(n,1);
    header_stamp_sec = nan(n,1);
    header_stamp_nsec = nan(n,1);
    header_frame_id = strings(n,1);
    pos_x = nan(n,1);
    pos_y = nan(n,1);
    pos_z = nan(n,1);
    ori_x = nan(n,1);
    ori_y = nan(n,1);
    ori_z = nan(n,1);
    ori_w = nan(n,1);
    bag_file_col = repmat(string(bagFileName), n, 1);

    for j = 1:n
        header_seq(j) = msgs{j}.Header.Seq;
        header_stamp_sec(j) = msgs{j}.Header.Stamp.Sec;
        header_stamp_nsec(j) = msgs{j}.Header.Stamp.Nsec;
        header_frame_id(j) = string(msgs{j}.Header.FrameId);
        pos_x(j) = msgs{j}.Pose.Position.X;
        pos_y(j) = msgs{j}.Pose.Position.Y;
        pos_z(j) = msgs{j}.Pose.Position.Z;
        ori_x(j) = msgs{j}.Pose.Orientation.X;
        ori_y(j) = msgs{j}.Pose.Orientation.Y;
        ori_z(j) = msgs{j}.Pose.Orientation.Z;
        ori_w(j) = msgs{j}.Pose.Orientation.W;
    end

    % Compute timestamps in seconds
    timeSec = double(header_stamp_sec) + double(header_stamp_nsec) * 1e-9;

    % Estimate sample rate (Hz)
    if length(timeSec) > 1
        dt = diff(timeSec);
        mean_dt = mean(dt, 'omitnan');
        sample_rate = 1 / mean_dt;
    else
        sample_rate = NaN;
    end

    % Add a constant column for sample rate
    sample_rate_col = repmat(sample_rate, n, 1);

    % Construct output table
    T = table(header_seq, header_stamp_sec, header_stamp_nsec, header_frame_id, ...
        pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w, ...
        bag_file_col, sample_rate_col);

    T.Properties.VariableNames{end-1} = 'BagFileName';
    T.Properties.VariableNames{end} = 'SampleRateHz';

    parquetwrite(outFile, T);
    fprintf('Saved %s data to %s (Sample Rate: %.2f Hz)\n', topicName, outFile, sample_rate);
end

function extractAndWriteTwist(bag, topicName, outFile, bagFileName)
    msgs = readMessages(select(bag, 'Topic', topicName), 'DataFormat', 'struct');
    n = length(msgs);

    header_seq = nan(n,1);
    header_stamp_sec = nan(n,1);
    header_stamp_nsec = nan(n,1);
    header_frame_id = strings(n,1);
    linear_x = nan(n,1);
    linear_y = nan(n,1);
    linear_z = nan(n,1);
    angular_x = nan(n,1);
    angular_y = nan(n,1);
    angular_z = nan(n,1);
    bag_file_col = repmat(string(bagFileName), n, 1);

    for j = 1:n
        header_seq(j) = msgs{j}.Header.Seq;
        header_stamp_sec(j) = msgs{j}.Header.Stamp.Sec;
        header_stamp_nsec(j) = msgs{j}.Header.Stamp.Nsec;
        header_frame_id(j) = string(msgs{j}.Header.FrameId);
        linear_x(j) = msgs{j}.Twist.Linear.X;
        linear_y(j) = msgs{j}.Twist.Linear.Y;
        linear_z(j) = msgs{j}.Twist.Linear.Z;
        angular_x(j) = msgs{j}.Twist.Angular.X;
        angular_y(j) = msgs{j}.Twist.Angular.Y;
        angular_z(j) = msgs{j}.Twist.Angular.Z;
    end

    % Compute timestamps in seconds
    timeSec = double(header_stamp_sec) + double(header_stamp_nsec) * 1e-9;

    % Estimate sample rate (Hz)
    if n > 1
        dt = diff(timeSec);
        mean_dt = mean(dt, 'omitnan');
        sample_rate = 1 / mean_dt;
    else
        sample_rate = NaN;
    end

    % Add a constant column for sample rate
    sample_rate_col = repmat(sample_rate, n, 1);

    % Construct output table
    T = table(header_seq, header_stamp_sec, header_stamp_nsec, header_frame_id, ...
        linear_x, linear_y, linear_z, angular_x, angular_y, angular_z, ...
        bag_file_col, sample_rate_col);

    T.Properties.VariableNames{end-1} = 'BagFileName';
    T.Properties.VariableNames{end} = 'SampleRateHz';

    parquetwrite(outFile, T);
    fprintf('Saved %s data to %s (Sample Rate: %.2f Hz)\n', topicName, outFile, sample_rate);
end

function extractAndWriteJoy(bag, topicName, outFile, bagFileName)
    msgs = readMessages(select(bag, 'Topic', topicName), 'DataFormat', 'struct');
    n = length(msgs);

    header_seq = nan(n,1);
    header_stamp_sec = nan(n,1);
    header_stamp_nsec = nan(n,1);
    header_frame_id = strings(n,1);

    axes_data = cell(n,1);
    buttons_data = cell(n,1);
    bag_file_col = repmat(string(bagFileName), n, 1);

    for j = 1:n
        header_seq(j) = msgs{j}.Header.Seq;
        header_stamp_sec(j) = msgs{j}.Header.Stamp.Sec;
        header_stamp_nsec(j) = msgs{j}.Header.Stamp.Nsec;
        header_frame_id(j) = string(msgs{j}.Header.FrameId);
        axes_data{j} = msgs{j}.Axes;
        buttons_data{j} = msgs{j}.Buttons;
    end

    % Compute timestamps in seconds
    timeSec = double(header_stamp_sec) + double(header_stamp_nsec) * 1e-9;

    % Estimate sample rate (Hz)
    if n > 1
        dt = diff(timeSec);
        mean_dt = mean(dt, 'omitnan');
        sample_rate = 1 / mean_dt;
    else
        sample_rate = NaN;
    end

    % Add a constant column for sample rate
    sample_rate_col = repmat(sample_rate, n, 1);

    % Create output table
    T = table(header_seq, header_stamp_sec, header_stamp_nsec, header_frame_id, ...
        axes_data, buttons_data, bag_file_col, sample_rate_col);

    T.Properties.VariableNames{end-1} = 'BagFileName';
    T.Properties.VariableNames{end} = 'SampleRateHz';

    parquetwrite(outFile, T);
    fprintf('Saved %s data to %s (Sample Rate: %.2f Hz)\n', topicName, outFile, sample_rate);
end

function extractAndWriteJointState(bag, topicName, outFile, bagFileName)
    msgs = readMessages(select(bag, 'Topic', topicName), 'DataFormat', 'struct');
    n = length(msgs);

    header_seq = nan(n,1);
    header_stamp_sec = nan(n,1);
    header_stamp_nsec = nan(n,1);
    header_frame_id = strings(n,1);
    name_data = cell(n,1);
    position_data = cell(n,1);
    velocity_data = cell(n,1);
    effort_data = cell(n,1);
    bag_file_col = repmat(string(bagFileName), n, 1);

    for j = 1:n
        msg = msgs{j};
        header_seq(j) = msg.Header.Seq;
        header_stamp_sec(j) = msg.Header.Stamp.Sec;
        header_stamp_nsec(j) = msg.Header.Stamp.Nsec;
        header_frame_id(j) = string(msg.Header.FrameId);
        name_data{j} = string(msg.Name);
        position_data{j} = msg.Position;
        velocity_data{j} = msg.Velocity;
        effort_data{j} = msg.Effort;
    end

    % Compute timestamps in seconds
    timeSec = double(header_stamp_sec) + double(header_stamp_nsec) * 1e-9;

    % Estimate sample rate (Hz)
    if n > 1
        dt = diff(timeSec);
        mean_dt = mean(dt, 'omitnan');
        sample_rate = 1 / mean_dt;
    else
        sample_rate = NaN;
    end

    % Add sample rate column
    sample_rate_col = repmat(sample_rate, n, 1);

    % Create table and write
    T = table(header_seq, header_stamp_sec, header_stamp_nsec, header_frame_id, ...
        name_data, position_data, velocity_data, effort_data, ...
        bag_file_col, sample_rate_col);

    T.Properties.VariableNames{end-1} = 'BagFileName';
    T.Properties.VariableNames{end} = 'SampleRateHz';

    parquetwrite(outFile, T);
    fprintf('Saved %s data to %s (Sample Rate: %.2f Hz)\n', topicName, outFile, sample_rate);
end

function extractAndWriteEmpty(bag, topicName, outFile, bagFileName)
    bagSel = select(bag, 'Topic', topicName);
    msgs = readMessages(bagSel, 'DataFormat', 'struct');
    times = bagSel.MessageList.Time;

    n = length(msgs);
    stamp_sec = nan(n, 1);
    stamp_nsec = nan(n, 1);
    bag_file_col = repmat(string(bagFileName), n, 1);

    for j = 1:n
        ros_time = times(j);
        stamp_sec(j) = floor(ros_time);
        stamp_nsec(j) = round((ros_time - stamp_sec(j)) * 1e9);
    end

    % Estimate sample rate (Hz)
    if n > 1
        dt = diff(times);
        mean_dt = mean(dt, 'omitnan');
        sample_rate = 1 / mean_dt;
    else
        sample_rate = NaN;
    end

    % Add sample rate column
    sample_rate_col = repmat(sample_rate, n, 1);

    T = table(stamp_sec, stamp_nsec, bag_file_col, sample_rate_col);
    T.Properties.VariableNames{end-1} = 'BagFileName';
    T.Properties.VariableNames{end} = 'SampleRateHz';

    parquetwrite(outFile, T);
    fprintf('Saved %s events to %s (Sample Rate: %.2f Hz)\n', topicName, outFile, sample_rate);
end