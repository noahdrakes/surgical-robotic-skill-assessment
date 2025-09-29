function [data, isValid] = dataFilterTare(data)

    % Filter parameters
    sgolayOrder = 3;
    sgolayWindow = 21;
    butterOrder = 2;
    cutoffFreq = 0.3;  % Hz — adjust based on expected gravity variation

    isValid = true;

    % Check for sufficient data
    if height(data) < 50
        isValid = false;
        warning('Too few samples (%d) for taring — need at least 50.', height(data));
        return;
    end

    % Accel (tare + SG + optional Butterworth HPF)
    hasAccel = all(ismember({'linear_accel_x','linear_accel_y','linear_accel_z'},...
        data.Properties.VariableNames));
    if hasAccel
        % Tare
        offset = mean([...
            data.linear_accel_x(1:50), ...
            data.linear_accel_y(1:50), ...
            data.linear_accel_z(1:50)], 1);

        ax = sgolayfilt(data.linear_accel_x - offset(1), sgolayOrder, sgolayWindow);
        ay = sgolayfilt(data.linear_accel_y - offset(2), sgolayOrder, sgolayWindow);
        az = sgolayfilt(data.linear_accel_z - offset(3), sgolayOrder, sgolayWindow);

        % Optional high-pass filter to remove gravity
        if ismember('SampleRateHz', data.Properties.VariableNames)
            fs = mean(data.SampleRateHz, 'omitnan');
            [b, a] = butter(butterOrder, cutoffFreq / (fs / 2), 'high');

            data.linear_accel_x = filtfilt(b, a, ax);
            data.linear_accel_y = filtfilt(b, a, ay);
            data.linear_accel_z = filtfilt(b, a, az);
        else
            % Fall back to SG-smoothed signal if SampleRateHz not present
            data.linear_accel_x = ax;
            data.linear_accel_y = ay;
            data.linear_accel_z = az;
        end
    end

    % Force (tare only)
    hasForceXYZ = all(ismember({'force_x','force_y','force_z'}, data.Properties.VariableNames));
    if hasForceXYZ
        offset = mean([...
            data.force_x(1:50), ...
            data.force_y(1:50), ...
            data.force_z(1:50)], 1);
        data.force_x = data.force_x - offset(1);
        data.force_y = data.force_y - offset(2);
        data.force_z = data.force_z - offset(3);
    end

    hasForceXY = all(ismember({'force_x','force_y'}, data.Properties.VariableNames)) && ...
                 ~ismember('force_z', data.Properties.VariableNames);
    if hasForceXY
        offset = mean([data.force_x(1:50), data.force_y(1:50)], 1);
        data.force_x = data.force_x - offset(1);
        data.force_y = data.force_y - offset(2);
    end

    % If none were valid, mark as invalid
    if ~hasAccel && ~hasForceXYZ && ~hasForceXY
        isValid = false;
    end
end