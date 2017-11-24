%====================================
% script for performing HPSDE in the puddle world task.
%====================================
close all;
clear variables;

%Set paths to libraries
addpath('gpml-matlab');
startup;  % set a path to gpml-matlab
addpath('minConf/minConf');
addpath('minConf/minFunc');
addpath('minConf');
addpath('featureFunc');
addpath('rewardFunc');
addpath('DMPtask');

RewardFunc = @ReachingTaskReward;

%== comment out one of the option policy update types ==
%PolicyUpdate = 'RWR';
PolicyUpdate = 'REPS';

%== comment out one of the gating policy types ==
GatePolicyType = 'gp';
%GatePolicyType = 'soft';

ActionDim = 10;
ContextDim = 1;
ClusterSeedsNum = 10;

if strcmp( GatePolicyType, 'gp')
    hrl = HPSDEGP(ClusterSeedsNum, ContextDim, ActionDim, PolicyUpdate);
else
    hrl = HPSDEsoft(ClusterSeedsNum, ContextDim, ActionDim, PolicyUpdate);
end

hrl.SelectFeatureType( 1 );

n_ini = 600;
clusteringNum = 3;
updateNum = 6;
sampleNum = 50;

ContextIni = 11 * rand(n_ini, ContextDim) - 5;

ActionIni = zeros( n_ini, ActionDim );
t = linspace(0, pi, 160);
for i = 1:n_ini
    s = ContextIni( i, 1 );
    y = (12 * rand(1, 1) - 6) * sin(t) + linspace( 0, s, 160 ) + + 0.3 * rand(1, 160);
    y = [y,  s *ones( 1, 40 )];
    y(1,1) = 0;
    dmp_y = Dmp(y, 'original');
    ActionIni( i, : ) = dmp_y.w' *1.0e-4;
end

RewardIni = zeros( n_ini, 1 );

for i = 1:n_ini
    RewardIni( i, 1 ) = RewardFunc( ContextIni( i, : ), ActionIni( i, : ) );
end

hrl.StoreSamples( ContextIni, ActionIni, RewardIni );
p_ini = unifpdf( 0, -5, 6 ) * ones( n_ini, 1 );
hrl.StoreSampleProb( p_ini );

selectedOption = [];
OptionNum = [];
begin_t = cputime;

for i = 1:clusteringNum
    message = [ 'iteration num: ', sprintf( '%d', i)  ];
    disp( message );
    
    hrl.ClusterSamples();
    
    for k = 1:updateNum
        hrl.UpdatePolicySet();
        message = [ 'iteration num: ', sprintf( '%d', i), ' update ite: ', sprintf( '%d', k)  ];
        disp( message );

        for j = 1:sampleNum
            test_context = 11 * rand - 5;
            [action_new, option, p] = hrl.GenerateAction( test_context  );
            selectedOption = [selectedOption, option];
            reward_new = RewardFunc( test_context, action_new );
            hrl.StoreSamples( test_context, action_new, reward_new);
            hrl.StoreSampleProb( p );
            hrl.PolicySet( option, 1 ).StoreSamples( test_context, action_new, reward_new );

            policyNum = hrl.CheckOptionNum();
            OptionNum = [ OptionNum;  policyNum];
        end % end of sampling
    end % end of policy update
end % end of clustering

hrl.UpdatePolicySet();

end_t = cputime;
mess = [ 'total learning time: ', sprintf( '%d', end_t - begin_t ), ' [s]' ];
disp(mess);

f0 = figure;
plot( hrl.RewardSet );

f2 = figure;
plot(selectedOption, 'o');

figure;
plot(OptionNum);

% Comment out this line when you want to overwrite the policy file
%save( 'hrl_dmp.mat', 'hrl' );

draw_solution_dmp;
