%box parameters
x1 = 2; y1 = -2; w1=2; h1 = 0.9;
x2 = 2; y2 = 1.1; w2=2; h2 = 0.9;
x3 = 2; y3 = 4.1; w3=2; h3 = 1;
x4 = 2; y4 = -30; w4=2; h4 = 26;
x5 = 2; y5 = 7; w5=2; h5 = 27;

% Draw the environment
f = figure('Position',[1 200 750 300]);
xmin = -2; xmax = 8; ymin = -6; ymax = 8;

axis( [ xmin xmax ymin ymax] ); 
hold on;
grid on;

rectangle('Position',[x1 y1 w1 h1], 'FaceColor', [1.0 1 0.8]);
rectangle('Position',[x2 y2 w2 h2], 'FaceColor', [1.0 1 0.8]);
rectangle('Position',[x3 y3 w3 h3], 'FaceColor', [1.0 1 0.8]);
rectangle('Position',[x4 y4 w4 h4], 'FaceColor', [1.0 1 0.8]);
rectangle('Position',[x5 y5 w5 h5], 'FaceColor', [1.0 1 0.8]);

hold on;

t = linspace(0, pi, 160);
%x = linspace(0, 6, 160  );
%y = linspace(0, 0, 160  );
x = 3 - 3 * cos(t);
s = 0;
y = 8 * sin(t) + linspace(0,s, 160) + 0.3 * rand(1, 160); 
x = [x, 6*ones( 1, 40 )];
y = [y, s*ones( 1, 40 )];

dmp_x = Dmp(x, 'original');
dmp_y = Dmp(y, 'original');

%s = 10 *rand - 4;
%test_context = s;

testNum = 40;
s = linspace( -3.9, 6.7, testNum ); %-3.9,5.7
param_x.xf = 6; 
x_new = dmp_x.generalize(param_x);
T = 100;
Nx = size(x_new, 2);
ind = 0:T-1;
ind = round(ind * Nx/(T-1));
ind(1) = 1;
x_norm = x_new(ind);

line( [6 6],[ -6 8 ], 'Color', 'k', 'LineStyle', '--');

for i = 1:testNum
    test_context = s(i);
    [action_new, option] = hrl.GenerateGreedyAction( test_context  );
    dmp_y.w = action_new' * 1.0e4;

    param_y.xf = test_context; 
    y_new = dmp_y.generalize(param_y);

    Ny = size(y_new, 2);
    ind = 0:T-1;
    ind = round(ind * Ny/(T-1));
    ind(1) = 1;
    y_norm = y_new(ind);


    if option == 1
        plot( x_norm, y_norm, 'Color', [1.0, 0.5, 0  ] );
    elseif option == 2
        plot( x_norm, y_norm, 'm' );
    elseif option == 3
        plot( x_norm, y_norm, 'b' );
    elseif option == 4
        plot( x_norm, y_norm, 'g' );
    elseif option == 5
        plot( x_norm, y_norm,  'Color', [0.8, 0.8, 0.  ] );
    else
        plot( x_norm, y_norm );
    end
    hold on;
    drawnow;
end
plot( 0, 0.8, 'ro' );
set(gca,'FontName','Times New Roman');
set(gca,'FontSize',18);
    %set(gca,'XTickLabel',[]);
    %set(gca,'YTickLabel',[]);
xlabel('x', 'FontSize',20);
ylabel('y', 'FontSize',20);