function compressVideo()

    maxSizeMB = 16;
    scaleWidth = 1280;  % Set to [] to skip resizing

    % Select files
    [files, path] = uigetfile('*.mp4', 'Select videos to compress', 'MultiSelect', 'on');
    if isequal(files, 0)
        disp('No files selected.');
        return;
    end

    % Handle single vs multiple file selection
    if ischar(files)
        files = {files};
    end

    for i = 1:length(files)
        inputFile = fullfile(path, files{i});
        [~, name, ~] = fileparts(files{i});
        outputFile = fullfile(path, [name, '_compressed.mp4']);
        applyFFMPEG(inputFile, outputFile, maxSizeMB, scaleWidth);
    end
end

function applyFFMPEG(inputPath, outputPath, maxSizeMB, scaleWidth)
    duration = get_duration(inputPath);

    % maxBits is 15.4 MB in bits
    maxBits = 15400 * 8 * 1024;
    targetBitrateKbps = floor((maxBits / duration) / 1000);

    fprintf('\nCompressing: %s\n', inputPath);
    fprintf('Duration: %.1fs | Target bitrate: %d kbps\n', duration, targetBitrateKbps);

    % Construct ffmpeg command
    if isempty(scaleWidth)
        scaleCmd = '';
    else
        scaleCmd = sprintf('-vf scale=%d:-2', scaleWidth);
    end

    cmd = sprintf(['ffmpeg -y -i "%s" %s -c:v libx264 -b:v %dk -pix_fmt yuv420p ' ...
                   '-movflags +faststart "%s"'], ...
                   inputPath, scaleCmd, targetBitrateKbps, outputPath);

    status = system(cmd);
    if status == 0
        fileInfo = dir(outputPath);
        sizeMB = fileInfo.bytes / (1024 * 1024);
        fprintf('Compressed to %.2f MB → %s\n', sizeMB, outputPath);
    else
        fprintf('ffmpeg failed for %s\n', inputPath);
    end
end

function duration = get_duration(videoPath)
    cmd = sprintf(['ffprobe -v error -show_entries format=duration ' ...
                   '-of default=noprint_wrappers=1:nokey=1 "%s"'], videoPath);
    [~, out] = system(cmd);
    duration = str2double(strtrim(out));
end