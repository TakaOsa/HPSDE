clear
clc
close all
dbstop if error
%initialize_dropbox_path(1, 0 , 1);

demoOr = [linspace(0,0,100) linspace(1,1,100)];

if 1
    t = linspace(0, 1.25*2*pi, 200);
    demoOr = -cos(t)+1;
    demoOr = [demoOr   demoOr(end)*ones(1,50)];
end

demoOr = interp1(linspace(0,1,numel(demoOr)), demoOr, linspace(0,1,200));


figure; grid on; hold on;
%set_fig_position([0.579 0.359 0.297 0.468]);
plot( demoOr)

param.xf = 20;

h = Dmp(demoOr, 'original');
yh = h.generalize(param);
tic
e = Dmp(demoOr, 'hoffmann');
ye = e.generalize(param);
toc

plot( yh, 'r');
plot( ye, 'k');








