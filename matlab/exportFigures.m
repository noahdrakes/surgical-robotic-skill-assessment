function exportFigures(data, figHandle, doSave, fileType, overwrite)

    % Defaults
    if nargin < 2 || isempty(figHandle), figHandle = gcf; end
    if nargin < 3 || isempty(doSave), doSave = true; end
    if nargin < 4 || isempty(fileType), fileType = 'fig'; end
    if nargin < 5 || isempty(overwrite), overwrite = false; end

    if ~doSave
        fprintf('[Skipping save] %s/%s/%s.%s\n', data.studySet, data.metricSource, data.metricType, fileType);
        return;
    end

    % Build path
    baseDir = fileparts(mfilename('fullpath'));
    projectRoot = fullfile(baseDir, '..');
    figFolder = fullfile(projectRoot, 'figures', data.studySet, data.metricSource);

    if ~exist(figFolder, 'dir')
        mkdir(figFolder);
    end

    % Build consistent base filename
    if strcmpi(data.metricSource, data.metricType)
        baseFileName = sprintf('%s_%s', data.studySet, data.metricType);
    else
        baseFileName = sprintf('%s_%s_%s', data.studySet, data.metricSource, data.metricType);
    end

    % Prepare file paths
    mainFileName = [baseFileName '.' fileType];
    mainFullPath = fullfile(figFolder, mainFileName);

    % Disable axes toolbar
    allAxes = findall(figHandle, 'type', 'axes');
    for ax = allAxes'
        axtoolbar(ax, 'default');
        axtoolbar(ax, 'Visible', 'off');
    end

    % Save main figure
    if exist(mainFullPath, 'file') && ~overwrite
        fprintf('[Skipped existing figure] %s\n', mainFullPath);
    else
        saveFigureFile(figHandle, mainFullPath, fileType);
        fprintf('[Saved] %s\n', mainFullPath);
    end

    % Save assumption figure if provided
    if isfield(data, 'hFigAssumptions') &&...
            isa(data.hFigAssumptions, 'matlab.ui.Figure') &&...
            isvalid(data.hFigAssumptions)
        assumptFileName = [baseFileName '_assumptions.' fileType];
        assumptFullPath = fullfile(figFolder, assumptFileName);

        % Disable toolbar
        allAxes = findall(data.hFigAssumptions, 'type', 'axes');
        for ax = allAxes'
            axtoolbar(ax, 'default');
            axtoolbar(ax, 'Visible', 'off');
        end

        if exist(assumptFullPath, 'file') && ~overwrite
            fprintf('[Skipped existing figure] %s\n', assumptFullPath);
        else
            saveFigureFile(data.hFigAssumptions, assumptFullPath, fileType);
            fprintf('[Saved] %s\n', assumptFullPath);
        end
    end
end

function saveFigureFile(figH, fullPath, fileType)
    switch lower(fileType)
        case 'fig'
            savefig(figH, fullPath);
        case {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
            exportgraphics(figH, fullPath, 'Resolution', 300);
        case {'pdf', 'svg'}
            exportgraphics(figH, fullPath, 'ContentType', 'vector');
        otherwise
            error('Unsupported file type: %s', fileType);
    end
end