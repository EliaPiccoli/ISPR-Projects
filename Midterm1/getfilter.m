function [F] = getfilter(varargin)
%GETFILTER Return the requested filter
%   Give the name of the filter return the matrix of its representation.
%   This version produces filters of defualt size [3x3].
%   Certain parameters only apply to certain filters.
%Input:
% - FN : string, filter name
% - A : string, for gradient filters specify the axis (X, Y)
% - S : flaot, for gaussian filter specify standard deviation
% - N : integer, specify filter dimension
%Output:
% - F : NxN matrix representing the filter
%
% -------------------------
%
% Intelligent Systems for Pattern Recognition AY 2020/2021
% Midterm 1, Assignment 6
% Elia Piccoli 621332
% getfilter.m

    % input checking
    [FN, A, S, N] = parseparameters(varargin{:});

    % Assign to F the correct filter
    switch FN
        case 'roberts'
            if A == 'X'; F = [1 0 ; 0 -1]; else; F = [0 -1 ; 1 0]; end
        case 'sobel'
            if A == 'X'; F = [1 0 -1; 2 0 -2; 1 0 -1]; else; F = [1 2 1; 0 0 0; -1 -2 -1]; end
        case 'prewitt'
            if A == 'X'; F = [-1 0 1; -1 0 1; -1 0 1]; else; F = [1 1 1; 0 0 0; -1 -1 -1]; end
        case 'average'
            F = ones(N)/(N*N);
        case 'gaussian'
            r = N;c = N;
            [row, col] = meshgrid(-(r-1)/2:(r-1)/2, -(c-1)/2:(c-1)/2);
            F = exp(-(row.^2+col.^2)/(2*S^2));
            F = F./sum(F(:));
        case 'log' % Laplacian of Gaussian
            % gaussian filter
            r = N;c = N;
            [row, col] = meshgrid(-(r-1)/2:(r-1)/2, -(c-1)/2:(c-1)/2);
            F = exp(-(row.^2+col.^2)/(2*S^2));
            F = F./sum(F(:));
            % laplacian filter
            F1 = F.*((row.*row + col.*col - 2*S^2)/(S^4));
            F = F1 - sum(F1(:))/prod(r*c);
    end
end

function [filter, p1, p2, p3] = parseparameters(varargin)
%PARSEPARAMETERS Input checking and default values
    switch nargin
        case 0
            error('You must specify one argument: string  = filter name');
        case 1
            filter = varargin{1};
            p1 = 'X';
            p2 = 0.5;
            p3 = 3;
        case 2
            [filter, p1] = varargin{1:2};
            p2 = 0.5;
            p3 = 3;
        case 3
            [filter, p1, p2] = varargin{1:3};
            p3 = 3;
        case 4
            [filter, p1, p2, p3] = varargin{:};
    end
end