%=====================================
% script for testing a contextual policy search method.
%=====================================

close all;
clear variables;

addpath('minConf/minConf');
addpath('minConf/minFunc');
addpath('minConf');

addpath('featureFunc');
addpath('rewardFunc');

RewardFunc = @RewardSingle;
%RewardFunc = @RewardDouble;

%cps = ContextualRWR();
cps = ContextualREPS();

cps.SelectFeatureType( 0 );

n_ini = 50;
updateNum = 20;
sampleNum = 50;

ContextIni = 2 * rand(n_ini, 1);
ActionIni = normrnd( 1.0, 0.8, [n_ini 1] );

RewardIni = zeros( n_ini, 1 );

for i = 1:n_ini
    RewardIni( i, 1 ) = RewardFunc( ContextIni( i, 1 ), ActionIni( i, 1 ) );
end


cps.StoreSamples( ContextIni, ActionIni, RewardIni );
cps.SetContextSamples( 3 );

GridRange = [ 0; 2; -0.5; 2  ];
gridsize = 0.005;
VisualizeReward( RewardFunc , GridRange, gridsize );
hold on;
view(2);
scatter3( ContextIni, ActionIni, RewardIni );

x = linspace(0, 2, 41);
Actions = zeros(1, 41 );
 red = [ 1 0 0];
 green = [0 1 0 ];
 blue = [ 0 0 1];
 Result = [];

for i=1:updateNum
    cps.PolicyUpdate();

    for j=1:sampleNum
         test_context = 2 * rand(1, 1);
         
        if i == updateNum
            action_new = cps.GenerateGreedyAction( test_context  );
            reward_new = RewardFunc( test_context, action_new );
            cps.StoreSamples( test_context, action_new, reward_new );
            scatter3( test_context, action_new, reward_new + 1.0, 'ro','filled' );
        else
           
            action_new = cps.GenerateAction( test_context  );
            reward_new = RewardFunc( test_context, action_new );
            cps.StoreSamples( test_context, action_new, reward_new );
            Result = [ Result; reward_new ];
            scatter3( test_context, action_new, reward_new + 1.0, 'bo');
        end
    end
    
    if i == 1 || i == updateNum
        for k = 1:41
            Actions(1, k ) = cps.GenerateGreedyAction( x( k )  );
        end
        if i == 1
            plot( x, Actions(1, :), 'g' );
        else
            plot( x, Actions(1, :), 'r' );
        end
    end
    
    axis( GridRange );
    
end

f = figure;
N = size(cps.RewardSet, 1);
plot( Result)
