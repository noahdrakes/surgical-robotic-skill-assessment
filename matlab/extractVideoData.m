%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract Video Streams from ROS Bag Files
%
% Description:
%   Extracts compressed video streams (MP4) from ROS bag files,
%   saves frame timestamps as CSV files, and logs extraction details.
%
% Inputs:
%   - Folder containing ROS bag files named with pattern 'trial_*.bag'
%
% Features:
%   - Extracts left and right video streams from predefined topics
%   - Saves video files and corresponding frame timestamp CSVs
%   - Logs extraction metadata including frame count and FPS
%
% Author: Sergio Machaca
% Contact: smachac2@jh.edu
% Created: 04 June 2025
% Last Updated: 27 June 2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

%% Setup

% Select directory
bagDir = uigetdir(pwd, 'Select folder containing bag files');
if bagDir == 0
    error('No folder selected. Exiting...');
end
cd(bagDir);

% Define output directory one level up from bagDir
videoDir = fullfile(fileparts(bagDir), 'video');
if ~exist(videoDir, 'dir')
    mkdir(videoDir);
    fprintf('Created output folder: %s\n', videoDir);
else
    fprintf('Saving videos to: %s\n', videoDir);
end

% Choose video topics
leftTopic = '/left/decklink/ECM/image_raw/compressed';
rightTopic = '/right/decklink/ECM/image_raw/compressed';

% Get list of bag files
bagFiles = dir('trial_*.bag');
nBags = length(bagFiles);
logEntries = {};  % Collect log entries
fprintf('Found %d bag files\n', nBags);

for i = 1:nBags
    bagFile = bagFiles(i).name;
    trialIdx = sprintf('T%02d', i);  % T01, T02, ...
    fprintf('\nProcessing %s...\n', bagFile);

    % Load the bag
    bag = rosbag(bagFile);

    % Create subfolder for this trial
    trialFolder = fullfile(videoDir, trialIdx);
    if ~exist(trialFolder, 'dir')
        mkdir(trialFolder);
    end

    % LEFT
    try
        leftVideo = fullfile(trialFolder, sprintf('%s_left.mp4', trialIdx));
        leftCSV = fullfile(trialFolder, sprintf('%s_left_timestamps.csv', trialIdx));
        [nFrames, fps, timestamps] = extractVideo(bag, leftTopic, leftVideo, leftCSV);
        logEntries{end+1,1} = trialIdx;
        logEntries{end,2} = bagFile;
        logEntries{end,3} = leftTopic;
        logEntries{end,4} = leftVideo;
        logEntries{end,5} = nFrames;
        logEntries{end,6} = fps;
    catch ME
        warning('Could not extract LEFT from %s: %s', bagFile, ME.message);
    end

    % RIGHT
    try
        rightVideo = fullfile(trialFolder, sprintf('%s_right.mp4', trialIdx));
        rightCSV = fullfile(trialFolder, sprintf('%s_right_timestamps.csv', trialIdx));
        [nFrames, fps, timestamps] = extractVideo(bag, rightTopic, rightVideo, rightCSV);
        logEntries{end+1,1} = trialIdx;
        logEntries{end,2} = bagFile;
        logEntries{end,3} = rightTopic;
        logEntries{end,4} = rightVideo;
        logEntries{end,5} = nFrames;
        logEntries{end,6} = fps;
    catch ME
        warning('Could not extract RIGHT from %s: %s', bagFile, ME.message);
    end
end

% Save extraction log
if ~isempty(logEntries)
    logTable = cell2table(logEntries, 'VariableNames', ...
        {'Trial', 'BagFile', 'Topic', 'VideoFile', 'NumFrames', 'FPS'});
    logPath = fullfile(videoDir, 'video_extraction_log.csv');
    writetable(logTable, logPath);
    fprintf('\nSaved extraction log to %s\n', logPath);
end

sprintf('Extracted videos to %s.', videoDir);

%% Helper function

% Extract video and save timestamps
function [nFrames, fps, times] = extractVideo(bag, imageTopic, videoOut, timestampCSV)
    imgSel = select(bag, 'Topic', imageTopic);
    msgs = readMessages(imgSel);
    nFrames = length(msgs);

    if nFrames == 0
        error('No messages on topic %s', imageTopic);
    end

    times = zeros(nFrames, 1);
    for k = 1:nFrames
        t = msgs{k}.Header.Stamp;
        times(k) = double(t.Sec) + double(t.Nsec) * 1e-9;
    end

    % Estimate FPS
    if nFrames > 1
        deltas = diff(times);
        fps = 1 / mean(deltas);
    else
        fps = 30;
    end
    fprintf('  [%s] %d frames at %.2f fps\n', imageTopic, nFrames, fps);

    % Write video
    firstImg = readImage(msgs{1});
    v = VideoWriter(videoOut, 'MPEG-4');
    v.FrameRate = fps;
    open(v);
    for k = 1:nFrames
        img = readImage(msgs{k});
        writeVideo(v, img);
    end
    close(v);
    fprintf('  Wrote %s\n', videoOut);

    % Save timestamps to CSV
    timestampTable = table((1:nFrames)', times, 'VariableNames', {'Frame', 'Timestamp'});
    writetable(timestampTable, timestampCSV);
    fprintf('  Wrote %s\n', timestampCSV);
end