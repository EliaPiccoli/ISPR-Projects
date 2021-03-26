% Intelligent Systems for Pattern Recognition AY 2020/2021
% Midterm 1, Assignment 6
% Elia Piccoli 621332
% faces.m

clc;clear;
close all;

% Select randomly from the dataset a tree image
% face = imread(strcat('./dataset/6_', int2str(randi([1,30])), '_s.bmp'));
face = imread('./dataset/6_10_s.bmp');

% Convert to gray scale
F = rgb2gray(face);

% -------------- Filtering

% image * average
AF = getfilter('average');
FAF = convolution(F, AF, true);
RFAF = rescale(FAF,'InputMin',0,'InputMax',255);
% image * gaussian
GF = getfilter('gaussian');
FGF = convolution(F, GF, true);
RFGF = rescale(FGF,'InputMin',0,'InputMax',255);

% -------------- Enhancement

% Sobel
SFX = getfilter('sobel', 'X');
SFY = getfilter('sobel', 'Y');
% image * Sobel
FSFX = convolution(F, SFX, true);
FSFY = convolution(F, SFY, true);
RFSFX = rescale(FSFX,'InputMin',0,'InputMax',255);
RFSFY = rescale(FSFY,'InputMin',0,'InputMax',255);
% (image * average) * Sobel
FAFSFX = convolution(FAF, SFX, true);
FAFSFY = convolution(FAF, SFY, true);
RFAFSFX = rescale(FAFSFX,'InputMin',0,'InputMax',255);
RFAFSFY = rescale(FAFSFY,'InputMin',0,'InputMax',255);
% (image * gaussian) * Sobel
FGFSFX = convolution(FGF, SFX, true);
FGFSFY = convolution(FGF, SFY, true);
RFGFSFX = rescale(FGFSFX,'InputMin',0,'InputMax',255);
RFGFSFY = rescale(FGFSFY,'InputMin',0,'InputMax',255);
% compute magnitude
MFSF = uint8(sqrt(double((FSFX.^2)+(FSFY.^2))));
MFAFSF = uint8(sqrt(double((FAFSFX.^2)+(FAFSFY.^2))));
MFGFSF = uint8(sqrt(double((FGFSFX.^2)+(FGFSFY.^2))));
RMFSF = rescale(MFSF,'InputMin',0,'InputMax',255);
RMFAFSF = rescale(MFAFSF,'InputMin',0,'InputMax',255);
RMFGFSF = rescale(MFGFSF,'InputMin',0,'InputMax',255);

%Prewitt
PFX = getfilter('prewitt', 'X');
PFY = getfilter('prewitt', 'Y');
% image * Prewitt
FPFX = convolution(F, PFX, true);
FPFY = convolution(F, PFY, true);
RFPFX = rescale(FPFX,'InputMin',0,'InputMax',255);
RFPFY = rescale(FPFY,'InputMin',0,'InputMax',255);
% (image * average) * Prewitt
FAFPFX = convolution(FAF, PFX, true);
FAFPFY = convolution(FAF, PFY, true);
RFAFPFX = rescale(FAFPFX,'InputMin',0,'InputMax',255);
RFAFPFY = rescale(FAFPFY,'InputMin',0,'InputMax',255);
% (image * gaussian) * Prewitt
FGFPFX = convolution(FGF, PFX, true);
FGFPFY = convolution(FGF, PFY, true);
RFGFPFX = rescale(FGFPFX,'InputMin',0,'InputMax',255);
RFGFPFY = rescale(FGFPFY,'InputMin',0,'InputMax',255);
% compute magnitude
MFPF = uint8(sqrt(double((FPFX.^2)+(FPFY.^2))));
MFAFPF = uint8(sqrt(double((FAFPFX.^2)+(FAFPFY.^2))));
MFGFPF = uint8(sqrt(double((FGFPFX.^2)+(FGFPFY.^2))));
RMFPF = rescale(MFPF,'InputMin',0,'InputMax',255);
RMFAFPF = rescale(MFAFPF,'InputMin',0,'InputMax',255);
RMFGFPF = rescale(MFGFPF,'InputMin',0,'InputMax',255);

% Roberts
RFX = getfilter('roberts', 'X');
RFY = getfilter('roberts', 'Y');
% image * Roberts
FRFX = convolution(F, RFX, true);
FRFY = convolution(F, RFY, true);
RFRFX = rescale(FRFX,'InputMin',0,'InputMax',255);
RFRFY = rescale(FRFY,'InputMin',0,'InputMax',255);
% (image * average) * Roberts
FAFRFX = convolution(FAF, RFX, true);
FAFRFY = convolution(FAF, RFY, true);
RFAFRFX = rescale(FAFRFX,'InputMin',0,'InputMax',255);
RFAFRFY = rescale(FAFRFY,'InputMin',0,'InputMax',255);
% (image * gaussian) * Roberts
FGFRFX = convolution(FGF, RFX, true);
FGFRFY = convolution(FGF, RFY, true);
RFGFRFX = rescale(FGFRFX,'InputMin',0,'InputMax',255);
RFGFRFY = rescale(FGFRFY,'InputMin',0,'InputMax',255);
% compute magnitude
MFRF = abs(FRFX) + abs(FRFY);
MFAFRF = abs(FAFRFX) + abs(FAFRFY);
MFGFRF = abs(FGFRFX) + abs(FGFRFY);
RMTRF = rescale(MFRF,'InputMin',0,'InputMax',255);
RMTAFRF = rescale(MFAFRF,'InputMin',0,'InputMax',255);
RMTGFRF = rescale(MFGFRF,'InputMin',0,'InputMax',255);

% -------------- Detection

% Sobel
threshold = 180/255;
TRMFSF = RMFSF;
TRMFAFSF = RMFAFSF;
TRMFGFSF = RMFGFSF;
TRMFSF(RMFSF < threshold) = 0;
TRMFAFSF(RMFAFSF < threshold) = 0;
TRMFGFSF(RMFGFSF < threshold) = 0;
% Prewitt
threshold = 170/255;
TRMFPF = RMFPF;
TRMFAFPF = RMFAFPF;
TRMFGFPF = RMFGFPF;
TRMFPF(RMFPF < threshold) = 0;
TRMFAFPF(RMFAFPF < threshold) = 0;
TRMFGFPF(RMFGFPF < threshold) = 0;
% Roberts
threshold = 60/255;
TRMFRF = RMTRF;
TRMFAFRF = RMTAFRF;
TRMFGFRF = RMTGFRF;
TRMFRF(RMTRF < threshold) = 0;
TRMFAFRF(RMTAFRF < threshold) = 0;
TRMFGFRF(RMTGFRF < threshold) = 0;

% -------------- Plot
% figure;
% subplot(131);imshow(T);title('Original');
% subplot(132);imshow(RTAF);title('Average');
% subplot(133);imshow(RTGF);title('Gaussian');
% Sobel
figure('NumberTitle', 'off', 'Name', 'SobelFilter Gradients');
subplot(231);imshow(RFSFX);title('OriginalX');
subplot(232);imshow(RFAFSFX);title('AverageX');
subplot(233);imshow(RFGFSFX);title('GaussianX');
subplot(234);imshow(RFSFY);title('OriginalY');
subplot(235);imshow(RFAFSFY);title('AverageY');
subplot(236);imshow(RFGFSFY);title('GaussianY');
figure('NumberTitle', 'off', 'Name', 'SobelFilter Magnitude');
subplot(221);imshow(F);title('Original Image')
subplot(222);imshow(RMFSF);title('OriginalM');
subplot(223);imshow(RMFAFSF);title('AverageM');
subplot(224);imshow(RMFGFSF);title('GaussianM');
figure('NumberTitle', 'off', 'Name', 'SobelFilter Result');
subplot(221);imshow(F);title('Original Image')
subplot(222);imshow(TRMFSF);title('Edges w/o Filtering');
subplot(223);imshow(TRMFAFSF);title('Edges Average');
subplot(224);imshow(TRMFGFSF);title('Edges Gaussian');
% Prewitt
figure('NumberTitle', 'off', 'Name', 'PrewittFilter Gradients');
subplot(231);imshow(RFPFX);title('OriginalX');
subplot(232);imshow(RFAFPFX);title('AverageX');
subplot(233);imshow(RFGFPFX);title('GaussianX');
subplot(234);imshow(RFPFY);title('OriginalY');
subplot(235);imshow(RFAFPFY);title('AverageY');
subplot(236);imshow(RFGFPFY);title('GaussianY');
figure('NumberTitle', 'off', 'Name', 'PrewittFilter Magnitude');
subplot(221);imshow(F);title('Original Image')
subplot(222);imshow(RMFPF);title('OriginalM');
subplot(223);imshow(RMFAFPF);title('AverageM');
subplot(224);imshow(RMFGFPF);title('GaussianM');
figure('NumberTitle', 'off', 'Name', 'PrewittFilter Result');
subplot(221);imshow(F);title('Original Image')
subplot(222);imshow(TRMFPF);title('Edges w/o Filtering');
subplot(223);imshow(TRMFAFPF);title('Edges Average');
subplot(224);imshow(TRMFGFPF);title('Edges Gaussian');
% Roberts
figure('NumberTitle', 'off', 'Name', 'RobertsFilter Gradients');
subplot(231);imshow(RFRFX);title('OriginalX');
subplot(232);imshow(RFAFRFX);title('AverageX');
subplot(233);imshow(RFGFRFX);title('GaussianX');
subplot(234);imshow(RFRFY);title('OriginalY');
subplot(235);imshow(RFAFRFY);title('AverageY');
subplot(236);imshow(RFGFRFY);title('GaussianY');
figure('NumberTitle', 'off', 'Name', 'RobertsFilter Magnitude');
subplot(221);imshow(F);title('Original Image')
subplot(222);imshow(RMTRF);title('OriginalM');
subplot(223);imshow(RMTAFRF);title('AverageM');
subplot(224);imshow(RMTGFRF);title('GaussianM');
figure('NumberTitle', 'off', 'Name', 'RobertsFilter Result');
subplot(221);imshow(F);title('Original Image')
subplot(222);imshow(TRMFRF);title('Edges w/o Filtering');
subplot(223);imshow(TRMFAFRF);title('Edges Average');
subplot(224);imshow(TRMFGFRF);title('Edges Gaussian');

% -------------- Laplacian of Gaussian
SLOGF = 3;
LOGF05 = getfilter('log', [], 0.5, SLOGF);
FLOGF05 = convolution(F, LOGF05, true);
RFLOGF05 = rescale(FLOGF05,'InputMin',0,'InputMax',255);

% -------------- Better LoG model
LOGF03 = getfilter('log', [], 0.3, SLOGF);
FLOGF03 = convolution(F, LOGF03, true);
LOGF04 = getfilter('log', [], 0.4, SLOGF);
FLOGF04 = convolution(F, LOGF04, true);
LOGF06 = getfilter('log', [], 0.6, SLOGF);
FLOGF06 = convolution(F, LOGF06, true);
LOGF08 = getfilter('log', [], 0.8, SLOGF);
FLOGF08 = convolution(F, LOGF08, true);
RFLOGF03 = rescale(FLOGF03,'InputMin',0,'InputMax',255);
RFLOGF04 = rescale(FLOGF04,'InputMin',0,'InputMax',255);
RFLOGF06 = rescale(FLOGF06,'InputMin',0,'InputMax',255);
RFLOGF08 = rescale(FLOGF08,'InputMin',0,'InputMax',255);
FLOGFALL = cat(3, FLOGF05, FLOGF03, FLOGF04, FLOGF06, FLOGF08);
[IR, IC] = size(F);
FLOGFAVG = zeros(IR, IC);
for i=1:IR
    for j=1:IC
        FLOGFAVG(i, j) = mean(FLOGFALL(i, j, :));
    end
end
RFLOGFAVG = rescale(FLOGFAVG,'InputMin',0,'InputMax',255);

% -------------- Extra Plot
figure('NumberTitle', 'off', 'Name', 'Comparison between LoG filters');
subplot(231);imshow(RFLOGF03);title('LoG $$\sigma = 0.3$$', 'interpreter','latex');
subplot(232);imshow(RFLOGF04);title('LoG $$\sigma = 0.4$$', 'interpreter','latex');
subplot(233);imshow(RFLOGF05);title('LoG $$\sigma = 0.5$$', 'interpreter','latex');
subplot(234);imshow(RFLOGF06);title('LoG $$\sigma = 0.6$$', 'interpreter','latex');
subplot(235);imshow(RFLOGF08);title('LoG $$\sigma = 0.8$$', 'interpreter','latex');
figure('NumberTitle', 'off', 'Name', 'Comparison between filters');
subplot(231);imshow(F);title('Original Image')
subplot(232);imshow(TRMFRF);title('Roberts')
subplot(233);imshow(TRMFPF);title('Prewitt');
subplot(234);imshow(TRMFSF);title('Sobel');
subplot(235);imshow(RFLOGF05);title('LoG - 0.5');
subplot(236);imshow(RFLOGFAVG);title('LoG avg');