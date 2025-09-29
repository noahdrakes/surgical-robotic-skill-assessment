function exportLME(LMEData, doSaveCoeffs, overwrite)

    if nargin < 2 || isempty(doSaveCoeffs), doSaveCoeffs = false; end
    if nargin < 3 || isempty(overwrite), overwrite = false; end

    % Validate required fields
    requiredFields = {'lme', 'metricLabel', 'studySet', 'metricSource'};
    for i = 1:numel(requiredFields)
        if ~isfield(LMEData, requiredFields{i})
            error('LMEData must contain field "%s"', requiredFields{i});
        end
    end

    % Capture model output as string
    modelTitle = sprintf('--- Linear Mixed Effects Model: %s ---', LMEData.metricLabel);
    modelText = evalc('disp(LMEData.lme)');
    fullText = sprintf('%s\n\n%s', modelTitle, modelText);

    % Build output path
    baseDir = fileparts(mfilename('fullpath'));
    projectRoot = fullfile(baseDir, '..');
    outputFolder = fullfile(projectRoot, 'models', LMEData.studySet, LMEData.metricSource);
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    % Create model summary (.txt)
    if isfield(LMEData, 'metricSource') && strcmpi(LMEData.metricLabel, LMEData.metricSource)
        baseName = LMEData.metricLabel;
    else
        baseName = sprintf('%s_%s', LMEData.metricSource, LMEData.metricLabel);
    end
    txtFile = fullfile(outputFolder, sprintf('%s_LME.txt', baseName));
    
    % Check for overwrite and save
    if exist(txtFile, 'file') && ~overwrite
        fprintf('[Skipped existing LME summary] %s\n', txtFile);
    else
        fid = fopen(txtFile, 'w');
        fprintf(fid, '%s', fullText);
        fclose(fid);
        fprintf('[Saved LME summary] %s\n', txtFile);
    end

    % Save coefficients (.csv)
    if doSaveCoeffs
        csvFile = fullfile(outputFolder, sprintf('%s_LME_coeffs.csv', LMEData.metricLabel));
        if exist(csvFile, 'file') && ~overwrite
            fprintf('[Skipped existing LME coefficients] %s\n', csvFile);
        else
            coeffTable = LMEData.lme.Coefficients;
            if ~istable(coeffTable)
                coeffTable = dataset2table(coeffTable); % handle if it's still a dataset
            end
            writetable(coeffTable, csvFile);
            fprintf('[Saved LME coefficients] %s\n', csvFile);
        end
    end
end