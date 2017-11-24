close all;
clear variables;
% demonstration

%box1
x1 = 2; y1 = -2; w1=2; h1 = 1;
x2 = 2; y2 = 1; w2=2; h2 = 1;
x3 = 2; y3 = 4; w3=2; h3 = 1;
x4 = 2; y4 = -30; w4=2; h4 = 26;
x5 = 2; y5 = 7; w5=2; h5 = 27;

t = linspace(0, pi, 160);
%x = linspace(0, 6, 160  );
%y = linspace(0, 0, 160  );
x = 3 - 3 * cos(t);
%s = 6;
%s = 11 * rand(1, 1) - 5;
s = 13 * rand(1, 1) - 5;
%y = 5 * sin(t) + linspace(0,s, 160) + 0.3 * rand(1, 160); 
y = generateRandomTraj( 0, s, 160, 2 );
y = y';

x = [x, 6*ones( 1, 40 )];
y = [y, s*ones( 1, 40 )];

dmp_x = Dmp(x, 'original');
dmp_y = Dmp(y, 'original');
action = dmp_y.w';
%action = 2 * rand( 1,  5);
%dmp_y.w = 0.2*10^5 * (1 - action' );

param_x.xf = 6; param_y.xf = s; 
x_new = dmp_x.generalize(param_x);
y_new = dmp_y.generalize(param_y);

T = 100;
Nx = size(x_new, 2);
ind = 0:T-1;
ind = round(ind * Nx/(T-1));
ind(1) = 1;
x_norm = x_new(ind);

Ny = size(y_new, 2);
ind = 0:T-1;
ind = round(ind * Ny/(T-1));
ind(1) = 1;
y_norm = y_new(ind);

%check length
xi = [x_norm; y_norm];
len = 0;
for i =2:T
    dx = norm( xi( :, i ) - xi(:, i-1) );
    len = len + dx;
end
len

%check collision
ex = ( x1 < x_norm ) .* ( x_norm < x1 + w1 );
ey = ( y1 < y_norm ) .* ( y_norm < y1 + h1 ) + ( y2 < y_norm ) .* ( y_norm < y2 + h2 ) + ( y3 < y_norm ) .* ( y_norm < y3+ h3 ) + ( y4 < y_norm ) .* ( y_norm < y4+ h4 ) + ( y5 < y_norm ) .* ( y_norm < y5+ h5 );
c = sum( ex .* ey )

ReachingTaskReward( param_y.xf, action );

% Draw the environment
f = figure;
xmin = -2; xmax = 8; ymin = -6; ymax = 8;
plot (x, y); 
axis( [ xmin xmax ymin ymax] ); 
hold on;
grid on;
scatter( 0, 0 , 'bo' );
scatter( 6, s, 'ro');
%set(gca,'XTickLabel',[]);
%set(gca,'YTickLabel',[]);

rectangle('Position',[x1 y1 w1 h1]);
rectangle('Position',[x2 y2 w2 h2]);
rectangle('Position',[x3 y3 w3 h3]);
rectangle('Position',[x4 y4 w4 h4]);
rectangle('Position',[x5 y5 w5 h5]);

plot( x_norm, y_norm );

% z = 1:size(f_x, 1);
% g = ( mod( z, 15 ) == 0 );
% 
% x_plot = [x(1), x( : , g) ];
% y_plot = [y(1), y( : , g) ];
% fx_plot = [f_x(1); f_x(g, :) ];
% fy_plot = [f_y(1); f_y(g, :)];
