function genFeedbackMx()

    % Generate all permutations of 3H and 3N
    base = ['H','H','H','N','N','N'];
    allPerms = unique(perms(base), 'rows');

    % Filter out sequences with 3 consecutive H or N
    validSeqs = {};
    for i = 1:size(allPerms, 1)
        seq = allPerms(i, :);
        if isValid(seq)
            validSeqs{end+1} = seq; %#ok<AGROW>
        end
    end

    fprintf('Found %d valid sequences.\n', numel(validSeqs));

    % Sample 30 participants with replacement
    numParticipants = 30;
    sampledIdx = randi(numel(validSeqs), [numParticipants, 1]);
    sampledSeqs = validSeqs(sampledIdx);
    
    % Save to CSV
    filename = '../protocol/suturing_sequences.csv';
    saveSeqToCSV(sampledSeqs, filename);
end

function tf = isValid(seq)
    % Convert to string if input is char array
    if ischar(seq)
        seq = string(seq);
    end
    tf = true;
    for i = 1:(length(seq) - 2)
        if seq(i) == seq(i+1) && seq(i+1) == seq(i+2)
            tf = false;
            return;
        end
    end
end

function saveSeqToCSV(sequences, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'Participant ID,Trial 1,Trial 2,Trial 3,Trial 4,Trial 5,Trial 6\n');
    
    for i = 1:length(sequences)
        seq = sequences{i};
        fprintf(fid, '%d,%s,%s,%s,%s,%s,%s\n', i, seq(1), seq(2), seq(3), seq(4), seq(5), seq(6));
    end
    
    fclose(fid);
    fprintf('Saved %d sequences to ''%s''\n', length(sequences), filename);
end