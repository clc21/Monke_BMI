function plotWaveform(M,fs,s,name)
%UNTITLED2 Summary of this function goes here
%   M x N for M (row) signals and N (col) timesteps
arguments
    M 
    fs double
    s double = 0.7  % vertical separation 
    name char = 'Give plot a name'
end

% th = 0.0048;
time = 0:1/fs:(size(M,2)-1)/fs;

vline = 1:size(M,1);
centres = 1-vline*s;
n_pos = 1:size(M,1);

% plot one figure
figure; hold on;
for i=n_pos
    data = M(i,:) - i*s;
    plot(time,data,'k','Linewidth',0.7);
end
grid on
grid minor
title(name);
yticks(flip(centres));
yticklabels(string(flip(n_pos')));
axis tight;
xlabel('time (s)');
ylabel('Principle components');
hold off

title(name)

end