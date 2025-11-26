function ranges = make_chunks(N, M)
%MAKECHUNKS Split 1:N into consecutive chunks of length <= M.
%
%   ranges = MAKECHUNKS(N, M)
%   returns an K-by-2 array where each row is [startIdx, endIdx]
%   for that chunk. Chunks cover 1:N without overlap.
%
%   Example:
%       >> makeChunks(10, 3)
%       ans =
%            1     3
%            4     6
%            7     9
%           10    10

    if N <= 0 || M <= 0
        ranges = zeros(0,2);
        return;
    end

    % Chunk start indices
    starts = 1:M:N;

    % Chunk end indices (cap at N)
    ends = starts + M - 1;
    ends(ends > N) = N;

    ranges = [starts(:), ends(:)];
end