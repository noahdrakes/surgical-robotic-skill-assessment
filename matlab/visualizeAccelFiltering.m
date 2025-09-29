function visualizeAccelFiltering(data)

    if ~all(ismember({'linear_accel_x','linear_accel_y','linear_accel_z'}, data.Properties.VariableNames))
        error('Accelerometer columns not found.');
    end

    if ~ismember('SampleRateHz', data.Properties.VariableNames)
        error('SampleRateHz column missing. Cannot compute time axis or apply HPF.');
    end

    fs = mean(data.SampleRateHz, 'omitnan');
    t = (1:height(data)) / fs;

    % Parameters
    sgolayOrder = 3;
    sgolayWindow = 21;
    butterOrder = 2;
    cutoffFreq = 0.3;

    % Raw & tared
    rawX = data.linear_accel_x;
    rawY = data.linear_accel_y;
    rawZ = data.linear_accel_z;

    offset = mean([rawX(1:50), rawY(1:50), rawZ(1:50)], 1);
    tareX = rawX - offset(1);
    tareY = rawY - offset(2);
    tareZ = rawZ - offset(3);

    % SG filter
    sgX = sgolayfilt(tareX, sgolayOrder, sgolayWindow);
    sgY = sgolayfilt(tareY, sgolayOrder, sgolayWindow);
    sgZ = sgolayfilt(tareZ, sgolayOrder, sgolayWindow);

    % Butterworth HPF
    [b, a] = butter(butterOrder, cutoffFreq / (fs / 2), 'high');
    hpX = filtfilt(b, a, sgX);
    hpY = filtfilt(b, a, sgY);
    hpZ = filtfilt(b, a, sgZ);

    % Plot
    figure('Name', 'Accel Filter Visualization', 'NumberTitle', 'off');
    titles = {'X', 'Y', 'Z'};
    signals = {tareX, tareY, tareZ; sgX, sgY, sgZ; hpX, hpY, hpZ};
    stages = {'Tared Only', 'Tared + SG', 'Tared + SG + HPF'};
    colors = {'k--', 'b-', 'r-'};

    for i = 1:3
        subplot(3, 1, i);
        hold on;
        for j = 1:3
            plot(t, signals{j, i}, colors{j}, 'DisplayName', stages{j});
        end
        ylabel(['Accel ' titles{i}]);
        if i == 3, xlabel('Time (s)'); end
        if i == 1, legend('Location', 'best'); end
        title(['Accel ' titles{i} ' Axis']);
        grid on;
    end
end