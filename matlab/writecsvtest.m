a = {"1", "2", "3"; "4", "5", "6"; "7", "8", 12312}

% a = {
%     "1", "2", "3";
%     "4", "5", "6";
%     "hi", 10, 11
% };

a(4, :) = {2, 3, 4}

% csvwrite("test.csv", a)
% csvwrite("test.csv", a)

writecell(a, "test.csv")
% writecell({3,4,3}, "test.csv")