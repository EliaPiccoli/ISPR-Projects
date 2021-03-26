% Intelligent Systems for Pattern Recognition AY 2020/2021
% Midterm 1, Assignment 6
% Elia Piccoli 621332
% avgtime.m
%
% Compute comparison between single and multi core convolution computation.
% Then plot the results showing the average time and the 25-75 percentile.

clc; clear;

N = 100;
sizes = [200 400 600 800 1000 1500 2000 2500 3000 3500 4000 4500 5000];
ltest = length(sizes);

timeavgs = zeros(1, ltest);
timeavgm = zeros(1, ltest);
mints = zeros(1, ltest);
maxts = zeros(1, ltest);
mintm = zeros(1, ltest);
maxtm = zeros(1, ltest);

kernel = fspecial('sobel');

for size=1:ltest
    M = randi(255, sizes(size));
    elapsed_time = zeros(N,1);
    % b = waitbar(0, 'Computing SingleCore Convolution');
    for i = 1:N
        % waitbar(i/N, b, sprintf('Computing SingleCore Convolution %d/%d', i, N));
        t = tic;
        convolution(M, kernel);
        t = toc(t);
        elapsed_time(i) = t;
    end
    mints(size) = prctile(elapsed_time, 25);
    maxts(size) = prctile(elapsed_time, 75);
    timeavgs(size) = mean(elapsed_time) ;
    fprintf('Matrix size: %d - SingleCore Time: %f\n', sizes(size), timeavgs(size));
    % delete(b);
    
    % b = waitbar(0, 'Computing MultiCore Convolution');
    elapsed_time = zeros(N,1);
    for i = 1:N
        % waitbar(i/N, b, sprintf('Computing MultiCore Convolution %d/%d', i, N));
        t = tic;
        convolution(M, kernel, true);
        t = toc(t);
        elapsed_time(i) = t;
    end
    mintm(size) = prctile(elapsed_time, 25);
    maxtm(size) = prctile(elapsed_time, 75);
    timeavgm(size) = mean(elapsed_time) ;
    fprintf('Matrix size: %d - MultiCore Time: %f\n', sizes(size), timeavgm(size));
    % delete(b);
end

f = figure;
title('Single vs Multi core convolution');
s = plot(sizes, timeavgs, 'r', 'lineWidth', 1);
hold on;
m = plot(sizes, timeavgm, 'b--', 'lineWidth', 1);
xinter = linspace(sizes(1), sizes(end), 3000);
minsint = interp1(sizes, mints, xinter, 'spline');
maxsint = interp1(sizes, maxts, xinter, 'spline');
sc = fill([xinter,fliplr(xinter)],[maxsint,fliplr(minsint)],[1,0,0]);
set(sc, 'facealpha', 0.5);
minmint = interp1(sizes, mintm, xinter, 'spline');
maxmint = interp1(sizes, maxtm, xinter, 'spline');
mc = fill([xinter,fliplr(xinter)],[maxmint,fliplr(minmint)],[0,1,0]);
set(mc, 'facealpha', 0.5);
delete(s);delete(m);
savgtime = interp1(sizes, timeavgs, xinter, 'spline');
plot(xinter, savgtime, 'r', 'lineWidth', 1);
mavgtime = interp1(sizes, timeavgm, xinter, 'spline');
plot(xinter, mavgtime, 'b--', 'lineWidth', 1);
legend('Single', 'Multi', 'Location', 'northwest');
xlabel('Matrix size [NxN]');
ylabel('Time [s]');
saveas(f, './plots/corecomparison.png');