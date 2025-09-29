function cropTime = cropButtonPress(force, time, doPlot, subjID, trialID, debugDir)

    if nargin < 3, doPlot = false; end
    if nargin < 4, subjID = ''; end
    if nargin < 5, trialID = ''; end
    if nargin < 6
        debugDir = fullfile('MISTIC_suturing_study', 'figures', 'DebugCrop');
    end

    cropTime = NaN;

    % Ensure subjID and trialID are strings
    if ~ischar(subjID) && ~isstring(subjID)
        subjID = char(subjID);
    end
    if isnumeric(trialID)
        trialID = sprintf('T%02d', trialID);
    else
        trialID = char(trialID);
    end

    if length(force) < 10 || length(force) ~= length(time)
        warning('Invalid input to cropButtonPress.');
        return;
    end

    % Parameters
    maxWindow = 20;
    maxPressGap = 2.6;
    minPressCount = 5;
    maxClusterDuration = 40;
    cropBuffer = 0.2;
    minPeakWidthSec = 0.1;
    minPromBase = 0.15;

    manualWindows = struct( ...
        'S08_T01', [0 30], ...
        'S10_T01', [0 12], ...
        'S12_T01', [0 35], ...
        'S13_T04', [0 10], ...
        'S14_T06', [0 15], ...
        'S16_T05', [0 9], ...
        'S16_T07', [0 10]);

    lookupKey = sprintf('%s_%s', subjID, trialID);
    if isfield(manualWindows, lookupKey)
        maxWindow = manualWindows.(lookupKey)(2);
    end

    % Restrict to analysis window
    t0 = time(1);
    tRel = time - t0;
    keepIdx = tRel <= maxWindow;
    fzShort = force(keepIdx);
    timeShort = time(keepIdx);
    tRelShort = tRel(keepIdx);

    fs = 1 / median(diff(timeShort));
    fzSmooth = sgolayfilt(fzShort, 3, 71);
    noiseFloor = median(abs(fzSmooth));
    minProm = max(minPromBase, 2 * noiseFloor);

    minDistPts = round(0.2 * fs);
    minWidthPts = round(minPeakWidthSec * fs);

    [~, locs, ~] = findpeaks(fzSmooth, ...
        'MinPeakProminence', minProm, ...
        'MinPeakDistance', minDistPts, ...
        'MinPeakWidth', minWidthPts);

    peakTimes = timeShort(locs);

    % Cluster logic
    validClusters = [];
    if ~isempty(peakTimes)
        timeDiffs = diff(peakTimes);
        isGap = timeDiffs > maxPressGap;
        groupEdges = [0; find(isGap); numel(peakTimes)];

        for k = 1:length(groupEdges)-1
            idx1 = groupEdges(k) + 1;
            idx2 = groupEdges(k+1);
            clusterTimes = peakTimes(idx1:idx2);
            groupSize = length(clusterTimes);
            duration = clusterTimes(end) - clusterTimes(1);
            maxInternalGap = max(diff(clusterTimes));

            if groupSize >= minPressCount && duration <= maxClusterDuration && maxInternalGap <= maxPressGap
                validClusters(end+1) = k;
            end
        end
    end

    % Determine cropTime and plot title
    if ~isempty(validClusters)
        k = validClusters(end);
        idx1 = groupEdges(k) + 1;
        idx2 = groupEdges(k+1);
        clusterLocs = locs(idx1:idx2);
        lastPeakTime = timeShort(clusterLocs(end));
        cropTime = lastPeakTime + cropBuffer;
        plotTitle = sprintf('%s %s | Crop = %.2fs', subjID, trialID, cropTime - t0);
    elseif isempty(locs)
        plotTitle = sprintf('%s %s | No peaks detected', subjID, trialID);
    else
        plotTitle = sprintf('%s %s | No valid cluster found', subjID, trialID);
    end

    % Plot and save
    if doPlot
        if ~exist(debugDir, 'dir')
            mkdir(debugDir);
        end
        figHandle = figure('Visible', 'off', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        hold on;
        plot(tRelShort, fzShort, 'b-', 'DisplayName', 'Raw Force');
        plot(tRelShort, fzSmooth, 'r-', 'DisplayName', 'Smoothed Force');
        if ~isempty(locs)
            plot(tRelShort(locs), fzSmooth(locs), 'ko', 'MarkerSize', 5, 'DisplayName', 'Detected Peaks');
        end
        if ~isnan(cropTime)
            xline(cropTime - t0, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Crop Time');
            fill([cropTime - t0, tRelShort(end), tRelShort(end), cropTime - t0], ...
                 [min(fzShort), min(fzShort), max(fzShort), max(fzShort)], ...
                 [0.9 0.9 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
        xlabel('Time (s)');
        ylabel('Force (N)');
        xlim([max(tRelShort(end) - 20, 0), tRelShort(end)]);
        title(plotTitle, 'Interpreter', 'none');
        legend('Location', 'best');
        hold off;
    
        outName = sprintf('%s_%s_cropButtonPress.png', subjID, trialID);
        pngDir = fullfile(debugDir, 'PNG');
        if ~isfolder(pngDir)
            mkdir(pngDir);
        end
        outPath = fullfile(pngDir, outName);
        fprintf('Saving %s\n', outPath);
        % exportgraphics(figHandle, outPath, 'ContentType', 'vector'); % SVG
        exportgraphics(figHandle, outPath, 'Resolution', 150); % PNG
        close(figHandle);
    end
end