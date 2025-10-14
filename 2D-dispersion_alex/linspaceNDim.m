function Y = linspaceNDim(d1, d2, n)
%LINSPACENDIM Linearly spaced multidimensional data set.
%   linspaceNDim(d1, d2) generates a multi-dimensional matrix of 100
%   equally spaced points between each element of matrices d1 and d2.
%
%   linspaceNDim(d1, d2, N) generates N points between each element of
%   matrices d1 and d2.
%
%   Example:
%       d1 = [0, 0];
%       d2 = [1, 2];
%       Y = linspaceNDim(d1, d2, 5);
%       % Y will be a 2x5 matrix with points linearly spaced between d1 and d2
%
%   This is based on the linspace_NDim function by Steeve AMBROISE
%   See license in fileshare_licenses/linspace_NDim license.txt

if nargin == 2
    n = 100;
end

d1 = d1(:);
d2 = d2(:);

if length(d1) ~= length(d2)
    error('d1 and d2 must have the same number of elements');
end

Y = zeros(length(d1), n);

for i = 1:length(d1)
    Y(i, :) = linspace(d1(i), d2(i), n);
end

end

