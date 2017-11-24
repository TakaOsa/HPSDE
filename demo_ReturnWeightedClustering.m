close all;
clear variables;

addpath('featureFunc');
addpath('rewardFunc');

N = 1000;
m = 10;

RewardType = '_RewardDouble';
%RewardType = '_RewardTripleSplit';

if strcmp( RewardType, '_RewardDouble')
    RewardFunc = @RewardDouble;
    Amax = 2;
elseif  strcmp( RewardType, '_RewardTripleSplit')
    RewardFunc = @RewardTripleSplit;
    Amax = 4;
end


range = [ 0; 2; 0; Amax  ];
gridsize = 0.05;
SAset = rand(N, 2);
SAset(: , 1) = 2*SAset(: , 1);
SAset(: , 2) = Amax * SAset(: , 2);

Rset = zeros(N, 1);

for i=1:N
    Rset(i, 1) = RewardFunc( SAset( i, 1 ), SAset( i, 2 ) );
end

D = SAset;
D_laplace = LaplacianEigenMapping(D, 15, 5)';
x = D';
 
z_ini = mod( randperm(N), m ) + 1;

% %===== without reward ==========
% weights = ones( 1, N );
% z_noweight = IWVBEMGMM( D_laplace, m, z_ini, weights, N  );
% 
%  
% 
%  VisualizeReward( RewardFunc, range, gridsize ); axis( [0 2 0 Amax] ); 
%  hold on;
%  g = ( z_noweight == 1 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height, 'ro' );
%  g = ( z_noweight== 2 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'bo' );
%  g = ( z_noweight == 3 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'go' );
%  g = ( z_noweight == 4 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'yo' );
%  g = ( z_noweight == 5 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'co' );
%  

%===== with reward ==========
Rmax = max( Rset );
Rmin = min( Rset);
weights = exp( 10* (Rset - Rmax)/ (Rmax -Rmin) )' ;
weights = N * weights/sum(weights);


 z_rewardweight = IWVBEMGMM( D_laplace, m, z_ini, weights, N  );
 VisualizeReward( RewardFunc, range, gridsize ); axis( [0 2 0 Amax] ); 
 hold on;
 
 g = ( z_rewardweight == 1 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height, 'ro' );
 g = ( z_rewardweight == 2 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'bo' );
 g = ( z_rewardweight == 3 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'go' );
 g = ( z_rewardweight == 4 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'yo' );
 g = ( z_rewardweight == 5 );  X = x( : , g ); height = 2 * ones( sum(g), 1 ); scatter3( X(1, :), X( 2, : ), height,'co' );
 
 
