% Intelligent Systems for Pattern Recognition AY 2020/2021
% Midterm 1, Assignment 6
% Elia Piccoli 621332
% trees.m

clc;clear;
close all;

% Select randomly from the dataset a tree image
% tree = imread(strcat('./dataset/2_', int2str(randi([1,30])), '_s.bmp'));
tree = imread('./dataset/2_30_s.bmp');

% Convert to gray scale
T = rgb2gray(tree);

% -------------- Filtering

% image * average
AF = getfilter('average');
TAF = convolution(T, AF, true);
RTAF = rescale(TAF,'InputMin',0,'InputMax',255);
% image * gaussian
GF = getfilter('gaussian');
TGF = convolution(T, GF, true);
RTGF = rescale(TGF,'InputMin',0,'InputMax',255);

% -------------- Enhancement

% Sobel
SFX = getfilter('sobel', 'X');
SFY = getfilter('sobel', 'Y');
% image * Sobel
TSFX = convolution(T, SFX, true);
TSFY = convolution(T, SFY, true);
RTSFX = rescale(TSFX,'InputMin',0,'InputMax',255);
RTSFY = rescale(TSFY,'InputMin',0,'InputMax',255);
% (image * average) * Sobel
TAFSFX = convolution(TAF, SFX, true);
TAFSFY = convolution(TAF, SFY, true);
RTAFSFX = rescale(TAFSFX,'InputMin',0,'InputMax',255);
RTAFSFY = rescale(TAFSFY,'InputMin',0,'InputMax',255);
% (image * gaussian) * Sobel
TGFSFX = convolution(TGF, SFX, true);
TGFSFY = convolution(TGF, SFY, true);
RTGFSFX = rescale(TGFSFX,'InputMin',0,'InputMax',255);
RTGFSFY = rescale(TGFSFY,'InputMin',0,'InputMax',255);
% compute magnitude
MTSF = uint8(sqrt(double((TSFX.^2)+(TSFY.^2))));
MTAFSF = uint8(sqrt(double((TAFSFX.^2)+(TAFSFY.^2))));
MTGFSF = uint8(sqrt(double((TGFSFX.^2)+(TGFSFY.^2))));
RMTSF = rescale(MTSF,'InputMin',0,'InputMax',255);
RMTAFSF = rescale(MTAFSF,'InputMin',0,'InputMax',255);
RMTGFSF = rescale(MTGFSF,'InputMin',0,'InputMax',255);

%Prewitt
PFX = getfilter('prewitt', 'X');
PFY = getfilter('prewitt', 'Y');
% image * Prewitt
TPFX = convolution(T, PFX, true);
TPFY = convolution(T, PFY, true);
RTPFX = rescale(TPFX,'InputMin',0,'InputMax',255);
RTPFY = rescale(TPFY,'InputMin',0,'InputMax',255);
% (image * average) * Prewitt
TAFPFX = convolution(TAF, PFX, true);
TAFPFY = convolution(TAF, PFY, true);
RTAFPFX = rescale(TAFPFX,'InputMin',0,'InputMax',255);
RTAFPFY = rescale(TAFPFY,'InputMin',0,'InputMax',255);
% (image * gaussian) * Prewitt
TGFPFX = convolution(TGF, PFX, true);
TGFPFY = convolution(TGF, PFY, true);
RTGFPFX = rescale(TGFPFX,'InputMin',0,'InputMax',255);
RTGFPFY = rescale(TGFPFY,'InputMin',0,'InputMax',255);
% compute magnitude
MTPF = uint8(sqrt(double((TPFX.^2)+(TPFY.^2))));
MTAFPF = uint8(sqrt(double((TAFPFX.^2)+(TAFPFY.^2))));
MTGFPF = uint8(sqrt(double((TGFPFX.^2)+(TGFPFY.^2))));
RMTPF = rescale(MTPF,'InputMin',0,'InputMax',255);
RMTAFPF = rescale(MTAFPF,'InputMin',0,'InputMax',255);
RMTGFPF = rescale(MTGFPF,'InputMin',0,'InputMax',255);

% Roberts
RFX = getfilter('roberts', 'X');
RFY = getfilter('roberts', 'Y');
% image * Roberts
TRFX = convolution(T, RFX, true);
TRFY = convolution(T, RFY, true);
RTRFX = rescale(TRFX,'InputMin',0,'InputMax',255);
RTRFY = rescale(TRFY,'InputMin',0,'InputMax',255);
% (image * average) * Roberts
TAFRFX = convolution(TAF, RFX, true);
TAFRFY = convolution(TAF, RFY, true);
RTAFRFX = rescale(TAFRFX,'InputMin',0,'InputMax',255);
RTAFRFY = rescale(TAFRFY,'InputMin',0,'InputMax',255);
% (image * gaussian) * Roberts
TGFRFX = convolution(TGF, RFX, true);
TGFRFY = convolution(TGF, RFY, true);
RTGFRFX = rescale(TGFRFX,'InputMin',0,'InputMax',255);
RTGFRFY = rescale(TGFRFY,'InputMin',0,'InputMax',255);
% compute magnitude
MTRF = abs(TRFX) + abs(TRFY);
MTAFRF = abs(TAFRFX) + abs(TAFRFY);
MTGFRF = abs(TGFRFX) + abs(TGFRFY);
RMTRF = rescale(MTRF,'InputMin',0,'InputMax',255);
RMTAFRF = rescale(MTAFRF,'InputMin',0,'InputMax',255);
RMTGFRF = rescale(MTGFRF,'InputMin',0,'InputMax',255);

% -------------- Detection

% Sobel
threshold = 180/255;
TRMTSF = RMTSF;
TRMTAFSF = RMTAFSF;
TRMTGFSF = RMTGFSF;
TRMTSF(RMTSF < threshold) = 0;
TRMTAFSF(RMTAFSF < threshold) = 0;
TRMTGFSF(RMTGFSF < threshold) = 0;
% Prewitt
threshold = 170/255;
TRMTPF = RMTPF;
TRMTAFPF = RMTAFPF;
TRMTGFPF = RMTGFPF;
TRMTPF(RMTPF < threshold) = 0;
TRMTAFPF(RMTAFPF < threshold) = 0;
TRMTGFPF(RMTGFPF < threshold) = 0;
% Roberts
threshold = 60/255;
TRMTRF = RMTRF;
TRMTAFRF = RMTAFRF;
TRMTGFRF = RMTGFRF;
TRMTRF(RMTRF < threshold) = 0;
TRMTAFRF(RMTAFRF < threshold) = 0;
TRMTGFRF(RMTGFRF < threshold) = 0;

% -------------- Plot
% figure;
% subplot(131);imshow(T);title('Original');
% subplot(132);imshow(RTAF);title('Average');
% subplot(133);imshow(RTGF);title('Gaussian');
% Sobel
figure('NumberTitle', 'off', 'Name', 'SobelFilter Gradients');
subplot(231);imshow(RTSFX);title('OriginalX');
subplot(232);imshow(RTAFSFX);title('AverageX');
subplot(233);imshow(RTGFSFX);title('GaussianX');
subplot(234);imshow(RTSFY);title('OriginalY');
subplot(235);imshow(RTAFSFY);title('AverageY');
subplot(236);imshow(RTGFSFY);title('GaussianY');
figure('NumberTitle', 'off', 'Name', 'SobelFilter Magnitude');
subplot(221);imshow(T);title('Original Image')
subplot(222);imshow(RMTSF);title('OriginalM');
subplot(223);imshow(RMTAFSF);title('AverageM');
subplot(224);imshow(RMTGFSF);title('GaussianM');
figure('NumberTitle', 'off', 'Name', 'SobelFilter Result');
subplot(221);imshow(T);title('Original Image')
subplot(222);imshow(TRMTSF);title('Edges w/o Filtering');
subplot(223);imshow(TRMTAFSF);title('Edges Average');
subplot(224);imshow(TRMTGFSF);title('Edges Gaussian');
% Prewitt
figure('NumberTitle', 'off', 'Name', 'PrewittFilter Gradients');
subplot(231);imshow(RTPFX);title('OriginalX');
subplot(232);imshow(RTAFPFX);title('AverageX');
subplot(233);imshow(RTGFPFX);title('GaussianX');
subplot(234);imshow(RTPFY);title('OriginalY');
subplot(235);imshow(RTAFPFY);title('AverageY');
subplot(236);imshow(RTGFPFY);title('GaussianY');
figure('NumberTitle', 'off', 'Name', 'PrewittFilter Magnitude');
subplot(221);imshow(T);title('Original Image')
subplot(222);imshow(RMTPF);title('OriginalM');
subplot(223);imshow(RMTAFPF);title('AverageM');
subplot(224);imshow(RMTGFPF);title('GaussianM');
figure('NumberTitle', 'off', 'Name', 'PrewittFilter Result');
subplot(221);imshow(T);title('Original Image')
subplot(222);imshow(TRMTPF);title('Edges w/o Filtering');
subplot(223);imshow(TRMTAFPF);title('Edges Average');
subplot(224);imshow(TRMTGFPF);title('Edges Gaussian');
% Roberts
figure('NumberTitle', 'off', 'Name', 'RobertsFilter Gradients');
subplot(231);imshow(RTRFX);title('OriginalX');
subplot(232);imshow(RTAFRFX);title('AverageX');
subplot(233);imshow(RTGFRFX);title('GaussianX');
subplot(234);imshow(RTRFY);title('OriginalY');
subplot(235);imshow(RTAFRFY);title('AverageY');
subplot(236);imshow(RTGFRFY);title('GaussianY');
figure('NumberTitle', 'off', 'Name', 'RobertsFilter Magnitude');
subplot(221);imshow(T);title('Original Image')
subplot(222);imshow(RMTRF);title('OriginalM');
subplot(223);imshow(RMTAFRF);title('AverageM');
subplot(224);imshow(RMTGFRF);title('GaussianM');
figure('NumberTitle', 'off', 'Name', 'RobertsFilter Result');
subplot(221);imshow(T);title('Original Image')
subplot(222);imshow(TRMTRF);title('Edges w/o Filtering');
subplot(223);imshow(TRMTAFRF);title('Edges Average');
subplot(224);imshow(TRMTGFRF);title('Edges Gaussian');

% -------------- Laplacian of Gaussian
SLOGF = 3;
LOGF05 = getfilter('log', [], 0.5, SLOGF);
TLOGF05 = convolution(T, LOGF05, true);
RTLOGF05 = rescale(TLOGF05,'InputMin',0,'InputMax',255);

% -------------- Better LoG model
LOGF03 = getfilter('log', [], 0.3, SLOGF);
TLOGF03 = convolution(T, LOGF03, true);
LOGF04 = getfilter('log', [], 0.4, SLOGF);
TLOGF04 = convolution(T, LOGF04, true);
LOGF06 = getfilter('log', [], 0.6, SLOGF);
TLOGF06 = convolution(T, LOGF06, true);
LOGF08 = getfilter('log', [], 0.8, SLOGF);
TLOGF08 = convolution(T, LOGF08, true);
RTLOGF03 = rescale(TLOGF03,'InputMin',0,'InputMax',255);
RTLOGF04 = rescale(TLOGF04,'InputMin',0,'InputMax',255);
RTLOGF06 = rescale(TLOGF06,'InputMin',0,'InputMax',255);
RTLOGF08 = rescale(TLOGF08,'InputMin',0,'InputMax',255);
TLOGFALL = cat(3, TLOGF05, TLOGF03, TLOGF04, TLOGF06, TLOGF08);
[IR, IC] = size(T);
TLOGFAVG = zeros(IR, IC);
for i=1:IR
    for j=1:IC
        TLOGFAVG(i, j) = mean(TLOGFALL(i, j, :));
    end
end
RTLOGFAVG = rescale(TLOGFAVG,'InputMin',0,'InputMax',255);

% -------------- Extra Plot
figure('NumberTitle', 'off', 'Name', 'Comparison between LoG filters');
subplot(231);imshow(RTLOGF03);title('LoG $$\sigma = 0.3$$', 'interpreter','latex');
subplot(232);imshow(RTLOGF04);title('LoG $$\sigma = 0.4$$', 'interpreter','latex');
subplot(233);imshow(RTLOGF05);title('LoG $$\sigma = 0.5$$', 'interpreter','latex');
subplot(234);imshow(RTLOGF06);title('LoG $$\sigma = 0.6$$', 'interpreter','latex');
subplot(235);imshow(RTLOGF08);title('LoG $$\sigma = 0.8$$', 'interpreter','latex');
figure('NumberTitle', 'off', 'Name', 'Comparison between filters');
subplot(231);imshow(T);title('Original Image')
subplot(232);imshow(TRMTRF);title('Roberts')
subplot(233);imshow(TRMTPF);title('Prewitt');
subplot(234);imshow(TRMTSF);title('Sobel');
subplot(235);imshow(RTLOGF05);title('LoG - 0.5');
subplot(236);imshow(RTLOGFAVG);title('LoG avg');