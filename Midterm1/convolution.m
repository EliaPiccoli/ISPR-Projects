function [C] = convolution(I, K, P)
%CONVOLUTION Compute 2D-Convolution between an image I and kernel K
% Computes convoltion betweem the Image and the Kernel. If provided the
% algorithm uses Matlab ParPool to compute the result.
%
%Input:
% - I : matrix [ N x M ], Image 
% - K : matrix [ N x N ], Kernel
% - P : boolean, parallel computation
%       True -> function uses parpool
%       False -> single core execution
%Output:
% - C : Convolution Matrix
%
% -------------------------
%
% Intelligent Systems for Pattern Recognition AY 2020/2021
% Midterm 1, Assignment 6
% Elia Piccoli 621332
% convolution.m

    % input checking
    if nargin < 2
        error('You sould provide 2 arguments: Image, Kernel');
    elseif nargin < 3
        P = false;
    end

    % cast I to int32 to avoid data type problems during computation
    I = cast(I, 'int32');

    % Create result matrix
    [IR,IC] = size(I);
    [KR,KC] = size(K);
    C = zeros(1, IR*IC);

    % Compute padding of original image (replicate)
    Xpad = floor(KR/2);
    Ypad = floor(KC/2);
    upad = repmat(I(1, :), Xpad, 1);
    bpad = repmat(I(end, :), Xpad, 1);
    PI = [upad ; I ; bpad];
    lpad = repmat(PI(:, 1), 1, Ypad);
    rpad = repmat(PI(:, end), 1, Ypad);
    PI = [lpad PI rpad];
    [~, PIC] = size(PI);

    % linearlize the matrixes for better parrallel computation
    LPI = reshape(PI', 1, []);
    LK = reshape(K', 1, []);

    % reverse the kernel
    LK = flip(LK);

    % Compute convolution
    if P
        parfor index=0:IR*IC-1
            C(index+1) = computeconv(index, IC, PIC, LPI, KR, KC, LK, Xpad, Ypad);
        end
    else
        for index=0:IR*IC-1
            C(index+1) = computeconv(index, IC, PIC, LPI, KR, KC, LK, Xpad, Ypad);
        end
    end
    
    % un-linearize C
    C = reshape(C, [IC, IR])';
end

function[sum] = computeconv(index, IC, PIC, LPI, KR, KC, LK, XP, YP)
    x = floor(index/IC);
    y = mod(index, IC);
    sum = 0;
    for i=0:KR-1
        for j=0:KC-1
            ImX = x + i - floor(KR/2);
            ImY = y + j - floor(KC/2);
            sum = sum + LK(i*KR+j+1)*LPI((ImX+XP)*PIC+ImY+YP+1);
        end
    end
end

% %% Test Conv
% p = fspecial('sobel');
% f1 = conv2(T, p);
% f1r = rescale(f1,'InputMin',0,'InputMax',255);
% f2 = imfilter(T, p, 'conv', 'replicate');
% myf = convolution(T, p, true);
% myfr = rescale(myf,'InputMin',0,'InputMax',255);
% figure;subplot(221);imshow(T);title('IMG');
% subplot(222);imshow(f1r);title('MATLAB CONV2');
% subplot(223);imshow(myfr);title('MY CONV');
% subplot(224);imshow(f2);title('MATLAB IMFILTER');
