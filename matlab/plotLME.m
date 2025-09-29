function hFigLME = plotLME(argin_struct)

    % Unpack data struct
    lme = argin_struct.lme;
    data = argin_struct.data;
    label = argin_struct.metricLabel;
    studySet = argin_struct.studySet;

    % Get groupVar type, define color maps
    switch studySet
        case 'multimodality'
            groupVar = 'Feedback';
            groupTitle = 'Feedback Group';
            groupLevels = categories(data.Feedback);
            colorMap = containers.Map(...
                {'H', 'N'}, ...
                {hex2rgb("#009E73"), hex2rgb("#E69F00")});  % Green, Orange

        case 'skillassessment'
            groupVar = 'SkillGroup';
            groupTitle = 'Skill Group';
            groupLevels = categories(data.SkillGroup);
            colorMap = containers.Map(...
                {'Novice', 'Intermediate', 'Expert'}, ...
                {hex2rgb("#D55E00"), hex2rgb("#56B4E9"), hex2rgb("#CC79A7")});  % Orange, Blue, Pink

        otherwise
            warning('Not a valid study set.')
    end

    % Unit labels for metrics
    switch label
        case 'PlatformForce'
            plainLabel = 'RMS Force (N)';
            metricLabel = 'RMS Force ($\mathrm{N}$)';
        case 'ToolForce'
            plainLabel = 'RMS Force (N)';
            metricLabel = 'RMS Force ($\mathrm{N}$)';
        case 'ToolAccel'
            plainLabel = 'Mean Acceleration (m/s^2)';
            metricLabel = 'Mean Acceleration ($\mathrm{m/s^2}$)';
        case 'PSMSpeed'
            plainLabel = 'Mean PSM Speed (mm/s)';
            metricLabel = 'Mean PSM Speed ($\mathrm{mm/s}$)';
        case 'PSMPathLen'
            plainLabel = 'Mean PSM Path Length (mm)';
            metricLabel = 'Mean PSM Path Length ($\mathrm{mm}$)';
        case 'ECMSpeed'
            plainLabel = 'Mean ECM Speed (mm/s)';
            metricLabel = 'Mean ECM Speed ($\mathrm{mm/s}$)';
        case 'ECMPathLen'
            plainLabel = 'Mean ECM Path Length (mm)';
            metricLabel = 'Mean ECM Path Length ($\mathrm{mm}$)';
        case 'TaskTime'
            plainLabel = 'Mean Task Completion Time (s)';
            metricLabel = 'Mean Task Completion Time ($\mathrm{s}$)';
        otherwise
            warning('Not a valid metricLabel.')
    end

    % Prepare trial axis
    trialVals = unique(data.NormalizedTrial);
    xq = linspace(min(trialVals), max(trialVals), 100)';

    % Create figure
    hFigLME = figure('NumberTitle', 'off', 'Name', plainLabel);
    hold on;

    for i = 1:length(groupLevels)
        group = groupLevels{i};

        % Prediction input (line of best fit)
        predData = table();
        groupCat = data.(groupVar);
        predData.(groupVar) = categorical(repmat({group}, length(xq), 1), ...
            categories(groupCat), 'Ordinal', isordinal(groupCat));
        predData.NormalizedTrial = xq;
        predData.Participant = repmat(data.Participant(1), length(xq), 1); % dummy participant

        [yhat, yCI] = predict(lme, predData, 'Conditional', false);

        % Actual data points
        groupData = data(data.(groupVar) == group, :);
        scatter(groupData.NormalizedTrial, groupData.(label), 40, ...
            'MarkerFaceColor', colorMap(group), ...
            'MarkerEdgeColor', 'k', ...
            'MarkerFaceAlpha', 0.4, ...
            'HandleVisibility', 'off');

        % Prediction line (line of best fit)
        plot(xq, yhat, 'Color', colorMap(group), ...
            'LineWidth', 2, ...
            'DisplayName', group);

        % Confidence interval
        fill([xq; flipud(xq)], ...
             [yCI(:,1); flipud(yCI(:,2))], ...
             colorMap(group), ...
             'FaceAlpha', 0.2, ...
             'EdgeColor', 'none', ...
             'DisplayName', sprintf('%s 95\\%% CI', group));
    end

    % Labels and formatting
    xlabel('Trial Number');
    ylabel(metricLabel);
    title(sprintf('%s per Trial by %s', metricLabel, groupTitle));
    legend('Location', 'northeast');
    xticks(unique(trialVals));
    grid on; box on;

end