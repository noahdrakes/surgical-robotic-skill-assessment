function endCropTime = cropTrialEnd(force, time, doPlot, subjID, trialID, debugDir)

    if nargin < 3, doPlot = false; end
    if nargin < 4, subjID = ''; end
    if nargin < 5, trialID = ''; end
    if nargin < 6
        debugDir = fullfile('MISTIC_suturing_study', 'figures', 'DebugCrop');
    end

    endCropTime = NaN;

    % Convert trialID to string
    if isnumeric(trialID)
        trialID = sprintf('T%02d', trialID);
    else
        trialID = char(trialID);
    end

    % Validate input lengths
    if length(force) < 10 || length(force) ~= length(time)
        warning('Invalid input to cropTrialEnd.');
        if doPlot
            % Plot empty figure with warning message
            tRelTail = zeros(1,1);
            forceTail = zeros(1,1);
            forceSmooth = zeros(1,1);
            locs = [];
            plotTitle = sprintf('%s %s | Invalid input data', subjID, trialID);
            plotSingleDebug(tRelTail, forceTail, forceSmooth, locs, NaN, 0, plotTitle, debugDir, subjID, trialID, 'cropTrialEnd');
        end
        return;
    end

    % Parameters
    tailWindow = 20;         % seconds
    cropBuffer = 1;          % seconds after last spike
    minProm = 0.15;
    minDistSec = 0.2;
    minWidthSec = 0.1;

    % Isolate tail window
    tEnd = time(end);
    tailMask = time >= (tEnd - tailWindow);
    timeTail = time(tailMask);
    forceTail = force(tailMask);
    tRelTail = timeTail - timeTail(1);

    if numel(timeTail) < 5
        endCropTime = tEnd;
        if doPlot
            plotTitle = sprintf('%s %s | Too short tail window', subjID, trialID);
            plotSingleDebug(tRelTail, forceTail, zeros(size(forceTail)), [], NaN, timeTail(1), plotTitle, debugDir, subjID, trialID, 'cropTrialEnd');
        end
        return;
    end

    % Smooth signal
    fs = 1 / median(diff(timeTail));
    forceSmooth = sgolayfilt(forceTail, 3, 71);

    % Detect peaks
    [~, locs] = findpeaks(forceSmooth, ...
        'MinPeakProminence', minProm, ...
        'MinPeakDistance', round(minDistSec * fs), ...
        'MinPeakWidth', round(minWidthSec * fs));

    if isempty(locs)
        endCropTime = tEnd;  % fallback
        plotTitle = sprintf('%s %s | No peaks detected', subjID, trialID);
    else
        lastPeakTime = timeTail(locs(end));
        postIdx = find(timeTail > lastPeakTime);

        if ~isempty(postIdx)
            postForce = forceTail(postIdx);
            postTime  = timeTail(postIdx);
            baselineIdx = find(postForce <= 1.0, 1, 'first');

            if ~isempty(baselineIdx)
                baselineTime = postTime(baselineIdx);
                endCropTime = min(baselineTime + cropBuffer, tEnd);
            else
                endCropTime = min(lastPeakTime + cropBuffer, tEnd);
            end
        else
            endCropTime = min(lastPeakTime + cropBuffer, tEnd);
        end
        plotTitle = sprintf('%s %s | End Crop = %.2fs', subjID, trialID, endCropTime - timeTail(1));
    end

    if doPlot
        plotSingleDebug(tRelTail, forceTail, forceSmooth, locs, endCropTime, timeTail(1), plotTitle, debugDir, subjID, trialID, 'cropTrialEnd');
    end
end


% Plot and save
function plotSingleDebug(tRel, fzRaw, fzSmooth, locs, cropTime, t0, plotTitle, debugDir, subjID, trialID, prefix)
    if ~exist(debugDir, 'dir')
        mkdir(debugDir);
    end
    figHandle = figure('Visible', 'off', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
    plot(tRel, fzRaw, 'b-', 'DisplayName', 'Raw Force'); hold on;
    plot(tRel, fzSmooth, 'r-', 'DisplayName', 'Smoothed');

    if ~isempty(locs)
        plot(tRel(locs), fzSmooth(locs), 'ko', 'MarkerSize', 5, 'DisplayName', 'Detected Peaks');
    end

    if ~isnan(cropTime)
        xline(cropTime - t0, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Crop Time');
        fill([cropTime - t0, tRel(end), tRel(end), cropTime - t0], ...
             [min(fzRaw), min(fzRaw), max(fzRaw), max(fzRaw)], ...
             [0.9 0.9 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end

    xlabel('Time (s)');
    ylabel('Force (N)');
    xlim([max(tRel(end) - 20, 0), tRel(end)]);
    title(plotTitle, 'Interpreter', 'none');
    legend('Location', 'best');

    outName = sprintf('%s_%s_%s.png', subjID, trialID, prefix);
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