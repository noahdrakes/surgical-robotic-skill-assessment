function hFigAssumptions = plotLMEAssumptions(lme, plotFlag)
    
    if ~plotFlag
        hFigAssumptions = [];  % Return empty if no plot requested
        return;
    end

    % Check if lme arg is actually an LME
    if ~isa(lme, 'LinearMixedModel')
        error('Input must be a LinearMixedModel object.');
    end

    % Extract residuals and fitted values
    res = residuals(lme);
    fittedVals = fitted(lme);

    % Residuals vs. fitted (linearity and homoscedasticity)
    if contains(char(lme.Formula), 'ToolForce')
        metricName = 'Tool RMS Force'; 
    elseif contains(char(lme.Formula), 'ToolAccel')
        metricName = 'Tool Acceleration';
    elseif contains(char(lme.Formula), 'PlatformForce')
        metricName = 'Platform RMS Force';
    elseif contains(char(lme.Formula), 'PSMSpeed')
        metricName = 'PSM Speed';
    elseif contains(char(lme.Formula), 'PSMPathLen')
        metricName = 'PSM Path Length';
    elseif contains(char(lme.Formula), 'ECMSpeed')
        metricName = 'ECM Speed';
    elseif contains(char(lme.Formula), 'ECMPathLen')
        metricName = 'ECM Path Length';
    elseif contains(char(lme.Formula),'TaskTime')
        metricName = 'Task Completion Time';
    end

    hFigAssumptions = figure('Name', ['LME Assumptions: ' metricName],...
        'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);
    
    subplot(1,3,1);
    scatter(fittedVals, res, 25, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('Fitted Values'); ylabel('Residuals');
    title('Residuals vs Fitted');
    grid on; refline(0,0);

    % Q-Q plot to check normality
    subplot(1,3,2);
    qqplot(res);
    title('Q-Q Plot of Residuals');

    % Histogram of residuals
    subplot(1,3,3);
    histogram(res, 'Normalization', 'pdf', 'FaceColor', [0.3 0.3 0.9], 'FaceAlpha', 0.7);
    hold on;
    x = linspace(min(res), max(res), 100);
    mu = mean(res); sigma = std(res);
    plot(x, normpdf(x, mu, sigma), 'r', 'LineWidth', 2);
    title('Histogram of Residuals');
    xlabel('Residual'); ylabel('Density');
    legend('Residuals', 'Normal PDF');
    grid on;
    box off;

    % Normality test
    % fprintf('\nLilliefors test for normality of residuals:\n');
    % [h, p] = lillietest(res);
    % if h
    %     fprintf('Residuals are NOT normally distributed (p = %.4f)\n', p);
    % else
    %     fprintf('Residuals are normally distributed (p = %.4f)\n', p);
    % end

end