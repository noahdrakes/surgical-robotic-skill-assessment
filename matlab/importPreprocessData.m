function [cleanData, validData] = importPreprocessData(args)

    % EDIT by noah drakes
    cropTimeMatrix = {};
    %%%%%%%%%%%%%%%%%%%%%%%

    % Unpack data struct
    dataDir = args.dataDir;
    inclTable = args.inclTable;
    fbTable = args.fbTable;
    metricSource = args.metricSource;
    metricType = args.metricType;
    plotCrop = args.plotCrop;

    % Initialize output
    validData = table();
    figButton = [];
    figEnd = [];
    idxButton = 1;
    idxEnd = 1;
    
    % Define data directories
    subjectFolders = dir(fullfile(dataDir, 'S*'));
    subjectFolders = subjectFolders([subjectFolders.isdir]);
    
    % Loop through subjects
    for i = 1:length(subjectFolders)
        subjID = subjectFolders(i).name;
        subjNum = str2double(extractAfter(subjID, 'S'));
        if isnan(subjNum)
            continue;
        end
    
        % Get trial inclusion row
        subjInclRow = inclTable(inclTable.Participant == subjNum, :);
        if isempty(subjInclRow)
            continue;
        end
    
        % Determine passed trials
        trialVars = startsWith(subjInclRow.Properties.VariableNames, 'T');
        trialNames = subjInclRow.Properties.VariableNames(trialVars);
        passedTrials = []; 
        
        for t = 1:length(trialNames)
            trialLabel = trialNames{t};
            if string(subjInclRow.(trialLabel)) == "P"
                trialNum = str2double(extractAfter(trialLabel, 'T'));
                if ~isnan(trialNum)
                    passedTrials(end+1) = trialNum;
                end
            end
        end
    
        if length(passedTrials) < 6
            warning('Skipping %s: fewer than 6 passed trials', subjID);
            continue;
        end
        passedTrials = passedTrials(1:6);
    
        % Get feedback labels
        fbRow = fbTable(fbTable.Participant == subjNum, :);
        if isempty(fbRow)
            warning('No feedback row for %s', subjID);
            continue;
        else
            fbLabels = string(table2array(fbRow(1, 2:7)));
            fbLabels = categorical(fbLabels, {'N', 'H'}, 'Ordinal', true);
        end
    
        % Locate parquet directory
        pqDir = fullfile(dataDir, subjID, 'parquet');
        if ~isfolder(pqDir)
            warning('Missing parquet folder for %s', subjID);
            continue;
        end
    
        % Process each trial
        for j = 1:length(passedTrials)
            trialNum = passedTrials(j);

            % Initialize variables
            isValidTrial = true;
            isValidATImini40Preprocess = false;
            isValidForceNLeftPreprocess = false; isValidForceNRightPreprocess = false;
            isValidAccelLeftPreprocess = false;  isValidAccelRightPreprocess = false;
            isValidPSM1Preprocess = false; isValidPSM2Preprocess = false;
            leftLen = 0; rightLen = 0;
            
           % Initialize tables
            data = table(); dataRight = table(); dataLeft = table();
            
            % Predefine debug folder for debug plots
            plotCropDir = fullfile(fileparts(dataDir), 'figures', 'debug_crop');
            
            % Identify crop time to remove button presses
            atiFilePath = fullfile(pqDir, sprintf('T%02d', trialNum), 'ATImini40.parquet');
            cropStartTime = NaN;
            cropEndTime = NaN;
            
            if isfile(atiFilePath)
                try
                    rawATI = parquetread(atiFilePath);
            
                    % Only compute cropTime if required columns exist
                    if all(ismember({'header_stamp_sec','header_stamp_nsec','force_z'}, rawATI.Properties.VariableNames))
                        timeVec = double(rawATI.header_stamp_sec) + double(rawATI.header_stamp_nsec) * 1e-9;
                        forceMag = sqrt(rawATI.force_x.^2 + rawATI.force_y.^2 + rawATI.force_z.^2);
            
                        % Identify start and end times for cropping trials
                        cropStartTime = cropButtonPress(forceMag, timeVec, plotCrop, subjID, trialNum, plotCropDir);
                        cropEndTime = cropTrialEnd(forceMag, timeVec, plotCrop, subjID, trialNum, plotCropDir);
                        

                        %%% EDIT by noah drakes %%
                        % writecell({subjID, trialNum, cropStartTime, cropEndTime}, "cropped.csv")
                        cropTimeMatrix = [cropTimeMatrix; {subjID, trialNum, cropStartTime, cropEndTime}];
                        %%%%%%%%%%%%%%%%%%%%%%%%%%
                    else
                        warning('Required columns missing in ATImini40 for %s T%02d', subjID, trialNum);
                    end
                catch ME
                    warning('Error reading ATImini40 data for %s T%02d: %s', subjID, trialNum, ME.message);
                end
            else
                warning('ATImini40 file not found for %s T%02d', subjID, trialNum);
            end


            switch metricSource
                case 'platform'
                    switch metricType
                        case 'force'
                            % Define file path
                            filePath = fullfile(pqDir, sprintf('T%02d', trialNum), 'ATImini40.parquet');
                        
                            % Skip if missing
                            if ~isfile(filePath)
                                warning('Missing file: %s', filePath);
                                continue;
                            end
                        
                            % Read and preprocess
                            try
                                % Read data
                                rawData = parquetread(filePath);

                                % Crop data using start and end times
                                if all(ismember({'header_stamp_sec','header_stamp_nsec'}, rawData.Properties.VariableNames))
                                    timeVec = double(rawData.header_stamp_sec) + double(rawData.header_stamp_nsec) * 1e-9;
                                
                                    % Default crop times if not defined
                                    if isnan(cropStartTime), cropStartTime = timeVec(1); end
                                    if isnan(cropEndTime), cropEndTime = timeVec(end); end
                                
                                    % Ensure crop range is valid
                                    if cropEndTime > cropStartTime
                                        keepIdx = timeVec >= cropStartTime & timeVec <= cropEndTime;
                                        rawData = rawData(keepIdx, :);
                                    else
                                        warning('Invalid crop range [%f to %f] for %s T%02d', ...
                                            cropStartTime, cropEndTime, subjID, trialNum);
                                    end
                                else
                                    warning('Missing time columns in rawData for %s T%02d', subjID, trialNum);
                                end
                        
                                % Tare
                                [data, isValidATImini40Preprocess] = dataFilterTare(rawData);

                            catch
                                warning('Could not read or preprocess %s', filePath);
                                continue;
                            end
                        
                            % Validate force columns after renaming and preprocessing
                            if ~isValidATImini40Preprocess || ~all(ismember({'force_x','force_y','force_z'}, data.Properties.VariableNames))
                                warning('Force columns missing or invalid in %s', filePath);
                                continue;
                            end
                        
                            % Extract to variables
                            fx = data.force_x;
                            fy = data.force_y;
                            fz = data.force_z;
        
                            % Calculate task completion time
                            taskTime = computeTaskTime(data);
        
                            % Calculate magnitude
                            data.Magnitude = sqrt(fx.^2 + fy.^2 + fz.^2);
                    end
 
                case 'tool'
                    switch metricType
                        case 'force'
                            % Define file paths
                            leftFilePath = fullfile(pqDir, sprintf('T%02d', trialNum),...
                                'forcen_left.parquet');
                            rightFilePath = fullfile(pqDir, sprintf('T%02d', trialNum),...
                                'forcen_right.parquet');
                        
                            % Skip if both missing
                            if ~isfile(leftFilePath) && ~isfile(rightFilePath)
                                warning('Missing ForceN files for trial %d of %s',...
                                    trialNum, subjID);
                                continue;
                            end
                        
                            % Read and preprocess
                            if isfile(leftFilePath)
                                try
                                    % Read data
                                    dataLeftRaw = parquetread(leftFilePath);

                                    % Crop data using cropTime
                                    if all(ismember({'header_stamp_sec','header_stamp_nsec'}, dataLeftRaw.Properties.VariableNames))
                                        timeVec = double(dataLeftRaw.header_stamp_sec) + double(dataLeftRaw.header_stamp_nsec) * 1e-9;
                                        if ~isnan(cropStartTime)
                                            dataLeftRaw = dataLeftRaw(timeVec >= cropStartTime, :);
                                        end
                                    else
                                        warning('Missing time columns in dataLeftRaw for T%02d', trialNum);
                                    end

                                    % Tare
                                    [dataLeft, isValidForceNLeftPreprocess] =...
                                        dataFilterTare(dataLeftRaw);
                                catch
                                    warning('Error reading or preprocessing %s', leftFilePath);
                                end
                            end
                        
                            if isfile(rightFilePath)
                                try
                                    % Read data
                                    dataRightRaw = parquetread(rightFilePath);

                                    % Crop data using cropTime
                                    if all(ismember({'header_stamp_sec','header_stamp_nsec'}, dataRightRaw.Properties.VariableNames))
                                        timeVec = double(dataRightRaw.header_stamp_sec) + double(dataRightRaw.header_stamp_nsec) * 1e-9;
                                        if ~isnan(cropStartTime)
                                            dataRightRaw = dataRightRaw(timeVec >= cropStartTime, :);
                                        end
                                    else
                                        warning('Missing time columns in dataRightRaw for T%02d', trialNum);
                                    end

                                    % Tare
                                    [dataRight, isValidForceNRightPreprocess] =...
                                        dataFilterTare(dataRightRaw);
                                catch
                                    warning('Error reading or preprocessing %s', rightFilePath);
                                end
                            end
                        
                            if ~isValidForceNLeftPreprocess && ~isValidForceNRightPreprocess
                                warning('ForceN data invalid or missing for trial %d of %s',...
                                    trialNum, subjID);
                                continue;
                            end
                        
                            % Combine and compute
                            if isValidForceNLeftPreprocess && isValidForceNRightPreprocess
                                minLen = min(height(dataLeft), height(dataRight));
                                fx = dataLeft.force_x(1:minLen) + dataRight.force_x(1:minLen);
                                fy = dataLeft.force_y(1:minLen) + dataRight.force_y(1:minLen);
                            elseif isValidForceNLeftPreprocess
                                fx = dataLeft.force_x;
                                fy = dataLeft.force_y;
                            elseif isValidForceNRightPreprocess
                                fx = dataRight.force_x;
                                fy = dataRight.force_y;
                            end
        
                            % Calculate task completion time
                            taskTime = computeTaskTime(dataRight, dataLeft);

                            % Convert from mN to N
                            fx = fx / 1000;
                            fy = fy / 1000;
                                                    
                            % Calculate magnitude
                            data.Magnitude = sqrt(fx.^2 + fy.^2);

                        case 'accel'
                            % Define file paths
                            leftFilePath = fullfile(pqDir, sprintf('T%02d', trialNum), 'accel_left.parquet');
                            rightFilePath = fullfile(pqDir, sprintf('T%02d', trialNum), 'accel_right.parquet');
                        
                            % Skip if both missing
                            if ~isfile(leftFilePath) && ~isfile(rightFilePath)
                                warning('Missing accel files for trial %d of %s', trialNum, subjID);
                                continue;
                            end
                        
                            % Read and preprocess left accel data
                            if isfile(leftFilePath)
                                try
                                    % Read data
                                    dataLeftRaw = parquetread(leftFilePath);

                                    % Crop data using cropTime
                                    if all(ismember({'header_stamp_sec','header_stamp_nsec'}, dataLeftRaw.Properties.VariableNames))
                                        timeVec = double(dataLeftRaw.header_stamp_sec) + double(dataLeftRaw.header_stamp_nsec) * 1e-9;
                                        if ~isnan(cropStartTime)
                                            dataLeftRaw = dataLeftRaw(timeVec >= cropStartTime, :);
                                        end
                                    else
                                        warning('Missing time columns in dataLeftRaw for T%02d', trialNum);
                                    end

                                    % Tare and filter
                                    [dataLeft, isValidAccelLeftPreprocess] =...
                                            dataFilterTare(dataLeftRaw);

                                    % Gravity correction for left data
                                    % REMOVE IN FURTHER DATA ANALYSIS;
                                    % GRAVITY COMPENSATION WAS REMOVED FROM
                                    % PYTHON CODE, BUT ONLY AFTER THE
                                    % FIRST DATASET WAS COLLECTED.
                                    % (Arduino library outputs m/s^2, no
                                    % need to convert in vb_LIS3DH.py)
                                    if isValidAccelLeftPreprocess
                                        dataLeft.linear_accel_x = dataLeft.linear_accel_x / 9.81;
                                        dataLeft.linear_accel_y = dataLeft.linear_accel_y / 9.81;
                                        dataLeft.linear_accel_z = dataLeft.linear_accel_z / 9.81;
                                    end

                                catch
                                    warning('Error reading or preprocessing %s', leftFilePath);
                                    dataLeft = [];
                                end
                            end
                            
                            % Read and preprocess right accel data
                            if isfile(rightFilePath)
                                try
                                    % Read data
                                    dataRightRaw = parquetread(rightFilePath);

                                    % Crop data using cropTime
                                    if all(ismember({'header_stamp_sec','header_stamp_nsec'}, dataRightRaw.Properties.VariableNames))
                                        timeVec = double(dataRightRaw.header_stamp_sec) + double(dataRightRaw.header_stamp_nsec) * 1e-9;
                                        if ~isnan(cropStartTime)
                                            dataRightRaw = dataRightRaw(timeVec >= cropStartTime, :);
                                        end
                                    else
                                        warning('Missing time columns in dataRightRaw for T%02d', trialNum);
                                    end

                                    % Tare and filter
                                    [dataRight, isValidAccelRightPreprocess] =...
                                            dataFilterTare(dataRightRaw);

                                    % Gravity correction for left data
                                    % REMOVE IN FURTHER DATA ANALYSIS;
                                    % GRAVITY COMPENSATION WAS REMOVED FROM
                                    % PYTHON CODE, BUT ONLY AFTER THE
                                    % FIRST DATASET WAS COLLECTED.
                                    % (Arduino library outputs m/s^2, no
                                    % need to convert in vb_LIS3DH.py)
                                    if isValidAccelLeftPreprocess
                                        dataLeft.linear_accel_x = dataLeft.linear_accel_x / 9.81;
                                        dataLeft.linear_accel_y = dataLeft.linear_accel_y / 9.81;
                                        dataLeft.linear_accel_z = dataLeft.linear_accel_z / 9.81;
                                    end

                                catch
                                    warning('Error reading or preprocessing %s', rightFilePath);
                                    dataRight = [];
                                end
                            end
                        
                            % Check validity
                            if ~isValidAccelLeftPreprocess && ~isValidForceNRightPreprocess
                                warning('Accel data invalid or missing for trial %d of %s', trialNum, subjID);
                                continue;
                            end
                        
                            % Combine and compute
                            if isValidAccelLeftPreprocess && isValidAccelRightPreprocess
                                minLen = min(height(dataLeft), height(dataRight));
                                ax = dataLeft.linear_accel_x(1:minLen) + dataRight.linear_accel_x(1:minLen);
                                ay = dataLeft.linear_accel_y(1:minLen) + dataRight.linear_accel_y(1:minLen);
                                az = dataLeft.linear_accel_z(1:minLen) + dataRight.linear_accel_z(1:minLen);
                            elseif isValidAccelLeftPreprocess
                                ax = dataLeft.linear_accel_x;
                                ay = dataLeft.linear_accel_y;
                                az = dataLeft.linear_accel_z;
                            elseif isValidAccelRightPreprocess
                                ax = dataRight.linear_accel_x;
                                ay = dataRight.linear_accel_y;
                                az = dataRight.linear_accel_z;
                            end
         
                            % Calculate task completion time
                            taskTime = computeTaskTime(dataRight, dataLeft);
        
                            % Calculate magnitude
                            % data.Magnitude = sqrt(ax.^2 + ay.^2 + az.^2);
                            data.Magnitude = DFT321f([ax, ay, az]);  % DFT321 ALGORITHM
                    end

                case 'psm'
                    % Define file paths
                    PSM1Path = fullfile(pqDir, sprintf('T%02d', trialNum), 'PSM1measured_cp.parquet');
                    PSM2Path = fullfile(pqDir, sprintf('T%02d', trialNum), 'PSM2measured_cp.parquet');
                    
                    % Skip if both missing
                    if ~isfile(PSM1Path) && ~isfile(PSM2Path)
                        warning('Missing PSM files for trial %d of %s', trialNum, subjID);
                        continue;
                    end

                    % Read PSM1 data
                    if isfile(PSM1Path)
                        try
                            % Read data
                            dataPSM1Raw = parquetread(PSM1Path);

                            % Crop data using cropTime
                            if all(ismember({'header_stamp_sec','header_stamp_nsec'}, dataPSM1Raw.Properties.VariableNames))
                                timeVec = double(dataPSM1Raw.header_stamp_sec) + double(dataPSM1Raw.header_stamp_nsec) * 1e-9;
                                if ~isnan(cropStartTime)
                                    dataPSM1 = dataPSM1Raw(timeVec >= cropStartTime, :);
                                end
                            else
                                warning('Missing time columns in dataPSM1Raw for T%02d', trialNum);
                                dataPSM1 = dataPSM1Raw;
                            end

                        catch
                            warning('Error reading or preprocessing %s', PSM1Path);
                            dataPSM1 = [];
                        end
                    end

                    % Read PSM2 data
                    if isfile(PSM2Path)
                        try
                            % Read data
                            dataPSM2Raw = parquetread(PSM2Path);

                            % Crop data using cropTime
                            if all(ismember({'header_stamp_sec','header_stamp_nsec'}, dataPSM2Raw.Properties.VariableNames))
                                timeVec = double(dataPSM2Raw.header_stamp_sec) + double(dataPSM2Raw.header_stamp_nsec) * 1e-9;
                                if ~isnan(cropStartTime)
                                    dataPSM2 = dataPSM2Raw(timeVec >= cropStartTime, :);
                                end
                            else
                                warning('Missing time columns in dataPSM2Raw for T%02d', trialNum);
                                dataPSM2 = dataPSM2Raw;
                            end

                        catch
                            warning('Error reading or preprocessing %s', PSM2Path);
                            dataPSM2 = [];
                        end
                    end

                    % Compute path length for PSM1
                    if ~isempty(dataPSM1) &&...
                            all(ismember({'pos_x','pos_y','pos_z'},...
                            dataPSM1.Properties.VariableNames))
                        p = [dataPSM1.pos_x, dataPSM1.pos_y, dataPSM1.pos_z];
                        dp = diff(p);
                        PSM1Len = sum(vecnorm(dp, 2, 2));
                    end
                    
                    % Compute path length for PSM2
                    if ~isempty(dataPSM2) &&...
                            all(ismember({'pos_x','pos_y','pos_z'},...
                            dataPSM2.Properties.VariableNames))
                        p = [dataPSM2.pos_x, dataPSM2.pos_y, dataPSM2.pos_z];
                        dp = diff(p);
                        PSM2Len = sum(vecnorm(dp, 2, 2));
                    end
                    
                    % Calculate task completion time
                    taskTime = computeTaskTime(dataPSM2, dataPSM1);

                    % Calculate mean path length
                    data.PSMPathLen = (PSM1Len + PSM2Len) / 2;

                    % Calculate speed (mm/s)
                    data.PSMSpeed = (PSM1Len + PSM2Len) / taskTime;

                case 'ecm'
                    % Define file path
                    ECM1Path = fullfile(pqDir, sprintf('T%02d', trialNum), 'ECM1measured_cp.parquet');
                    
                    % Skip if missing
                    if ~isfile(ECM1Path)
                        warning('Missing ECM1 file for trial %d of %s', trialNum, subjID);
                        continue;
                    end

                    % Read ECM1 data
                    if isfile(ECM1Path)
                        try
                            % Read data
                            dataECM1Raw = parquetread(ECM1Path);

                            % Crop data using cropTime
                            if all(ismember({'header_stamp_sec','header_stamp_nsec'}, dataECM1Raw.Properties.VariableNames))
                                timeVec = double(dataECM1Raw.header_stamp_sec) + double(dataECM1Raw.header_stamp_nsec) * 1e-9;
                                if ~isnan(cropStartTime)
                                    dataECM1 = dataECM1Raw(timeVec >= cropStartTime, :);
                                end
                            else
                                warning('Missing time columns in dataECM1Raw for T%02d', trialNum);
                                dataECM1 = dataECM1Raw;
                            end

                        catch
                            warning('Error reading or preprocessing %s', ECM1Path);
                            dataECM1 = [];
                        end
                    end

                    % Compute path length for ECM1
                    if ~isempty(dataECM1) &&...
                            all(ismember({'pos_x','pos_y','pos_z'},...
                            dataECM1.Properties.VariableNames))
                        p = [dataECM1.pos_x, dataECM1.pos_y, dataECM1.pos_z];
                        dp = diff(p);
                        ECM1Len = sum(vecnorm(dp, 2, 2));
                    end
                    
                    % Calculate task completion time
                    taskTime = computeTaskTime(dataECM1);

                    % Get ECM path length
                    data.ECMPathLen = ECM1Len;

                    % Calculate speed (mm/s)
                    data.ECMSpeed = ECM1Len / taskTime;
            end
            
    
            % Annotate
            data.Participant = repmat(subjNum, height(data), 1);
            data.Trial       = repmat(trialNum, height(data), 1);
            data.Feedback    = repmat(fbLabels(j), height(data), 1);
    
            % Summarize
            switch metricSource
                case 'tool'
                    switch metricType
                        case 'force'
                            summary = groupsummary(data, {'Participant','Trial','Feedback'}, @(x) sqrt(mean(x.^2)), 'Magnitude');
                            metricLabel = 'ToolForce';
                        case 'accel'
                            summary = groupsummary(data, {'Participant','Trial','Feedback'}, 'mean', 'Magnitude');
                            metricLabel = 'ToolAccel';
                    end

                case 'platform'
                    switch metricType
                        case 'force'
                            summary = groupsummary(data, {'Participant','Trial','Feedback'}, @(x) sqrt(mean(x.^2)), 'Magnitude');
                            metricLabel = 'PlatformForce';
                    end
                
                case 'psm'
                    switch metricType
                        case 'speed'
                            summary = groupsummary(data, {'Participant','Trial','Feedback'}, 'mean', 'PSMSpeed');
                            metricLabel = 'PSMSpeed';
                        case 'pathlen'
                            summary = groupsummary(data, {'Participant','Trial','Feedback'}, 'mean', 'PSMPathLen');
                            metricLabel = 'PSMPathLen';
                    end

                case 'ecm'
                    switch metricType
                        case 'speed'
                            summary = groupsummary(data, {'Participant','Trial','Feedback'}, 'mean', 'ECMSpeed');
                            metricLabel = 'ECMSpeed';
                        case 'pathlen'
                            summary = groupsummary(data, {'Participant','Trial','Feedback'}, 'mean', 'ECMPathLen');
                            metricLabel = 'ECMPathLen';
                    end
                    
                otherwise
                    warning('Check metricSource. No matching results.')
            end

            summary.Properties.VariableNames{end} = metricLabel;
            summary.TaskTime = repmat(taskTime, height(summary), 1);
            summary.ValidTrial = repmat(isValidTrial, height(summary), 1);
    
            % Append
            validData = [validData; summary];
        end
    end
    
    % Filter for participants with 6 valid trials
    [G, pid] = findgroups(validData.Participant);
    validCounts = splitapply(@(x) sum(x), validData.ValidTrial, G);
    incompletePIDs = pid(validCounts < 6);
    for i = 1:numel(incompletePIDs)
        warning('Participant %d has fewer than 6 valid trials and will be removed.', incompletePIDs(i));
    end
    validData.ValidTrial(ismember(validData.Participant, incompletePIDs)) = false;
    cleanData = validData(validData.ValidTrial, :);

    % Create normalized trial number column and append
    cleanData.NormalizedTrial = repmat((1:6)', height(cleanData)/6, 1);

    writecell(cropTimeMatrix, "croppedTimes.csv")

end