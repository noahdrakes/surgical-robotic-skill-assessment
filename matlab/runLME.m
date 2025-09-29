function runLME(params)

    % Get formula string based on study set and metric label
    formula = getFormula(params.studySet, params.metricLabel);

    % Fit model
    lme = fitlme(params.data, formula);

    % Assumption plots
    hFigAssumptions = plotLMEAssumptions(lme, params.plotAssumptions);

    % Package LME result
    LMEData = struct();
    LMEData.lme = lme;
    LMEData.data = params.data;
    LMEData.studySet = params.studySet;
    LMEData.metricSource = params.metricSource;
    LMEData.metricType = params.metricType;
    LMEData.metricLabel = params.metricLabel;
    LMEData.hFigAssumptions = hFigAssumptions;

    % Export LME stats
    exportLME(LMEData, params.exportLME, params.overwriteLME);

    % Plot results
    hFigLME = plotLME(LMEData);

    % Export plots
    exportFigures(LMEData, hFigLME, ...
        params.exportFig, params.figFormat, params.overwriteFig);

    % ========== Nested helper ==========
    function f = getFormula(studySet, metricLabel)
        switch studySet
            case 'multimodality'
                f = sprintf('%s ~ NormalizedTrial + Feedback + (NormalizedTrial|Participant)', metricLabel);
            case 'skillassessment'
                f = sprintf('%s ~ NormalizedTrial * SkillGroup + (1|Participant)', metricLabel);
            otherwise
                error('Unknown study set: %s', studySet);
        end
    end
end