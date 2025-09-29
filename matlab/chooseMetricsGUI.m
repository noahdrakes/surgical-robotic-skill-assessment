function [studySet, metricSource, metricType, flags] = chooseMetricsGUI()
    % Default values
    defaultSettingsFile = 'guiSettings.mat';

    defaultFlags = struct( ...
        'useThreeTrialSplit', true, ...
        'overwriteFig', true, ...
        'overwriteLME', true, ...
        'exportFig', true, ...
        'exportLME', true, ...
        'plotAssumptions', true, ...
        'plotCrop', false, ...
        'figFormat', 'pdf' ...
    );

    % Label-to-value mappings
    studySetLabels = {'Skill Assessment', 'Multimodality'};
    studySetValues = {'skillassessment', 'multimodality'};

    metricSourceLabels = {'Platform', 'Tool', 'PSM', 'ECM'};
    metricSourceValues = {'platform', 'tool', 'psm', 'ecm'};

    metricTypeLabels = {'Force', 'Acceleration', 'Speed', 'Path Length'};
    metricTypeValues = {'force', 'accel', 'speed', 'pathlen'};

    figFormats = {'pdf', 'png', 'jpg', 'svg'};

    % Try loading previous settings
    if isfile(defaultSettingsFile)
        loaded = load(defaultSettingsFile);
        if isfield(loaded, 'lastFlags')
            flags = loaded.lastFlags;
        else
            flags = defaultFlags;
        end
        
        % Default dropdowns
        lastDropdowns = struct( ...
            'studySet', studySetValues{1}, ...
            'metricSource', metricSourceValues{1}, ...
            'metricType', metricTypeValues{1}, ...
            'figFormat', defaultFlags.figFormat ...
        );
        
        if isfield(loaded, 'lastDropdowns')
            lastDropdowns = loaded.lastDropdowns;
        end
    else
        flags = defaultFlags;
    end

    % Initialize output
    studySet = '';
    metricSource = '';
    metricType = '';
    userCanceled = true;  % Flag to detect window close

    % UI figure
    fig = uifigure('Name', 'Select Analysis Options', ...
        'Position', [800, 500, 900, 450], 'CloseRequestFcn', @closeFig);

    % === Dropdowns (left column) ===
    uilabel(fig, 'Text', 'Study Set:', 'Position', [30, 400, 100, 22]);
    ddStudySet = uidropdown(fig, 'Items', studySetLabels, 'ItemsData', studySetValues, ...
        'Position', [160, 400, 200, 22]);
    
    uilabel(fig, 'Text', 'Metric Source:', 'Position', [30, 360, 100, 22]);
    ddMetricSource = uidropdown(fig, 'Items', metricSourceLabels, 'ItemsData', metricSourceValues, ...
        'Position', [160, 360, 200, 22], 'Value', metricSourceValues{1});
    
    uilabel(fig, 'Text', 'Metric Type:', 'Position', [30, 320, 100, 22]);
    ddMetricType = uidropdown(fig, ...
        'Items', {}, 'ItemsData', {}, 'Position', [160, 320, 200, 22]);

    uilabel(fig, 'Text', 'Figure Format:', 'Position', [30, 280, 100, 22]);
    ddFigFormat = uidropdown(fig, 'Items', figFormats, ...
        'Position', [160, 280, 200, 22], 'Value', flags.figFormat);

    % Map from metric source to valid types
    validMetricTypesMap = containers.Map( ...
        {'platform', 'tool', 'psm', 'ecm'}, ...
        { ...
            {'force'}, ...                                 % platform
            {'force', 'accel'}, ...                        % tool
            {'speed', 'pathlen'}, ...                      % psm
            {'speed', 'pathlen'} ...                       % ecm
        } ...
    );

    % Set initial valid types
    updateMetricTypes(ddMetricSource.Value);

    % Update values
    ddStudySet.Value      = lastDropdowns.studySet;
    ddMetricSource.Value  = lastDropdowns.metricSource;
    ddFigFormat.Value     = lastDropdowns.figFormat;

    updateMetricTypes(lastDropdowns.metricSource);
    ddMetricType.Value = lastDropdowns.metricType;
    
    % Callback for Metric Source dropdown
    ddMetricSource.ValueChangedFcn = @(src, event) updateMetricTypes(src.Value);

    % --- Function to update metric type options based on metric source
    function updateMetricTypes(metricSource)
        validTypes = validMetricTypesMap(metricSource);
        [validLabels, validValues] = filterMetricTypes(validTypes);
        ddMetricType.Items = validLabels;
        ddMetricType.ItemsData = validValues;
        ddMetricType.Value = validValues{1};  % Reset to first valid
    end
    
    % --- Helper to filter labels/values by allowed values
    function [filteredLabels, filteredValues] = filterMetricTypes(allowedValues)
        mask = ismember(metricTypeValues, allowedValues);
        filteredLabels = metricTypeLabels(mask);
        filteredValues = metricTypeValues(mask);
    end

    % === Checkboxes (left column) ===
    cb1 = uicheckbox(fig, 'Text', 'Use Three-Trial Split', ...
        'Position', [30, 230, 250, 22], 'Value', flags.useThreeTrialSplit);
    cb2 = uicheckbox(fig, 'Text', 'Overwrite Figures', ...
        'Position', [30, 200, 250, 22], 'Value', flags.overwriteFig);
    cb3 = uicheckbox(fig, 'Text', 'Overwrite LME', ...
        'Position', [30, 170, 250, 22], 'Value', flags.overwriteLME);
    cb4 = uicheckbox(fig, 'Text', 'Export Figures', ...
        'Position', [30, 140, 250, 22], 'Value', flags.exportFig);
    cb5 = uicheckbox(fig, 'Text', 'Export LME Results', ...
        'Position', [30, 110, 250, 22], 'Value', flags.exportLME);
    cb6 = uicheckbox(fig, 'Text', 'Plot LME Assumptions', ...
        'Position', [30, 80, 250, 22], 'Value', flags.plotAssumptions);
    cb7 = uicheckbox(fig, 'Text', 'Plot Crop Regions', ...
        'Position', [30, 50, 250, 22], 'Value', flags.plotCrop);

    % === OK Button ===
    uibutton(fig, 'Text', 'OK', 'Position', [100, 10, 120, 30], ...
        'ButtonPushedFcn', @(btn, event) onOK());

    % === Informational Key (right column) ===
    keyHTML = [ ...
        "<html><body style='font-family: Helvetica, Arial, sans-serif; font-size: 11px; padding: 4px; margin: 0;'>" + ...
        "<h2 style='text-align: left; text-decoration: underline; margin-bottom: 10px; '>OPTIONS</h2>" + ...
        "<b>STUDY DATASETS</b><br>" + ...
        "<table style='border-collapse: collapse; width: 650px; vertical-align: top;'>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Skill Assessment</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Evaluates performance across experience levels</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Multimodality</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Compares performance with and without haptic feedback</td></tr>" + ...
        "</table><br>" + ...
        "<b>METRIC SOURCES & TYPES</b><br>" + ...
        "<table style='border-collapse: collapse; width: 650px; vertical-align: top;'>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Platform | Force</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Task platform forces (ATI mini40 SI-40-2)</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Tool | Force, Accel</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Surgical tool tip forces and accelerations</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>PSM | Speed, Path Len</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Motion metrics for patient-side manipulators</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>ECM | Speed, Path Len</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Motion metrics for endoscopic camera arm</td></tr>" + ...
        "</table><br>" + ...
        "<b>ANALYSIS OPTIONS</b><br>" + ...
        "<table style='border-collapse: collapse; width: 650px; vertical-align: top;'>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Three-Trial Split</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Separate feedback trials into 3-trial groups</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Overwrite Figures</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Replace existing figure exports</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Overwrite LME Output</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Replace existing linear mixed effects results</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Export Figures</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Save generated plots</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Export LME Results</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Save statistical model results</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Plot LME Assumptions</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Generate LME diagnostic plots</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Plot Crop Regions</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>Show cropped windows for trial task completion time</td></tr>" + ...
        "  <tr><td style='text-align: right; width: 150px; padding-right: 5px; white-space: nowrap; vertical-align: top;'>Figure Format</td><td style='width: 10px; text-align: left; vertical-align: top;'>-</td><td style='vertical-align: top;'>File type for exported figures (pdf, png, jpg, etc.)</td></tr>" + ...
        "</table>" + ...
        "</body></html>"];
        
    uilabel(fig, ...
        'Position', [400, 30, 670, 400], ...
        'Text', keyHTML, ...
        'Interpreter', 'html', ...
        'WordWrap', 'on', ...
        'Tag', 'infoLabel');

    % Pause script until GUI interaction is complete
    uiwait(fig);

    % Check inputs
    if userCanceled
        fprintf('User canceled selection. No options chosen.\n');
    else
        % Reverse lookup to get display labels
        studySetLabel = studySetLabels{strcmp(studySetValues, studySet)};
        metricSourceLabel = metricSourceLabels{strcmp(metricSourceValues, metricSource)};
        metricTypeLabel = metricTypeLabels{strcmp(metricTypeValues, metricType)};
        
        fprintf('Selections:\n  Study Set      : %s\n  Metric Source  : %s\n  Metric Type    : %s\n', ...
            studySetLabel, metricSourceLabel, metricTypeLabel);
    end

    % === Callbacks ===
    function onOK()
        % Capture dropdowns
        studySet = ddStudySet.Value;
        metricSource = ddMetricSource.Value;
        metricType = ddMetricType.Value;

        % Checkboxes and format
        flags.useThreeTrialSplit = cb1.Value;
        flags.overwriteFig = cb2.Value;
        flags.overwriteLME = cb3.Value;
        flags.exportFig = cb4.Value;
        flags.exportLME = cb5.Value;
        flags.plotAssumptions = cb6.Value;
        flags.plotCrop = cb7.Value;
        flags.figFormat = ddFigFormat.Value;

        % Save selections
        lastFlags = flags;
        lastDropdowns = struct( ...
            'studySet', studySet, ...
            'metricSource', metricSource, ...
            'metricType', metricType, ...
            'figFormat', flags.figFormat ...
        );
        save(defaultSettingsFile, 'lastFlags', 'lastDropdowns');

        userCanceled = false;
        uiresume(fig);
        delete(fig);
    end

    function closeFig(~, ~)
        userCanceled = true;
        uiresume(fig);
        delete(fig);
    end
end