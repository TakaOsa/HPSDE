function [ r ] = ReachingTaskReward( s, a )
%REACHINGTASKREWARD 

%box1
x1 = 1.9; y1 = -2.4; w1=2.2; h1 = 1.6;
x2 = 1.9; y2 = 0.8; w2=2.2; h2 = 1.4;
x3 = 1.9; y3 = 3.8; w3=2.2; h3 = 1.6;
x4 = 1.9; y4 = -30.2; w4=2.2; h4 = 26.4;
x5 = 1.9; y5 = 6.8; w5=2.2; h5 = 27;


t = linspace(0, pi, 160);
x = 3 - 3 * cos(t);
y = 8 * sin(t); 
%x = linspace(0, 3*pi, 160  );
%x = pi - (t + pi) .* cos(t);
%y = 0.5 *(t +2* pi ) .* sin(t); 
x = [x, 6*ones( 1, 40 )];
y = [y, zeros( 1, 40 )];

dmp_x = Dmp(x, 'original');
dmp_y = Dmp(y, 'original');

%dmp_y.w = 0.1*10^5 * (1 - a' );
dmp_y.w = a' * 1.0e4;


param_x.xf = 6; param_y.xf = s; 
x_new = dmp_x.generalize(param_x);
y_new = dmp_y.generalize(param_y);
% plot( x_new, y_new );

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
%len

%check collision
ex = ( x1 < x_norm ) .* ( x_norm < x1 + w1 );
ey = ( y1 < y_norm ) .* ( y_norm < y1 + h1 ) + ( y2 < y_norm ) .* ( y_norm < y2 + h2 ) + ( y3 < y_norm ) .* ( y_norm < y3+ h3 ) + ( y4 < y_norm ) .* ( y_norm < y4+ h4 ) + ( y5 < y_norm ) .* ( y_norm < y5+ h5 );
c = sum( ex .* ey );
%c

r = -len - 50 * c ; %500 for puddle2 %50 for puddle 

end

