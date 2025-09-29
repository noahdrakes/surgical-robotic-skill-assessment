function taskTime = computeTaskTime(arg1, arg2, arg)

    if nargin < 1, arg1 = []; end
    if nargin < 2, arg2 = []; end
    if nargin < 3, arg = []; end

    taskTime = NaN;  % default if no timestamps are available
    hasTimeCols = @(tbl) istable(tbl) && height(tbl) > 0 && ...
        all(ismember({'header_stamp_sec', 'header_stamp_nsec'},...
        tbl.Properties.VariableNames));

    try
        if hasTimeCols(arg1)
            startTime = double(arg1.header_stamp_sec(1)) +...
                double(arg1.header_stamp_nsec(1)) * 1e-9;
            endTime   = double(arg1.header_stamp_sec(end)) +...
                double(arg1.header_stamp_nsec(end)) * 1e-9;
            taskTime  = endTime - startTime;

        elseif hasTimeCols(arg2)
            startTime = double(arg2.header_stamp_sec(1)) +...
                double(arg2.header_stamp_nsec(1)) * 1e-9;
            endTime   = double(arg2.header_stamp_sec(end)) +...
                double(arg2.header_stamp_nsec(end)) * 1e-9;
            taskTime  = endTime - startTime;

        elseif hasTimeCols(arg)
            startTime = double(arg.header_stamp_sec(1)) +...
                double(arg.header_stamp_nsec(1)) * 1e-9;
            endTime   = double(arg.header_stamp_sec(end)) +...
                double(arg.header_stamp_nsec(end)) * 1e-9;
            taskTime  = endTime - startTime;
        end
    catch
        % Leave taskTime as NaN if any error occurs
    end
end