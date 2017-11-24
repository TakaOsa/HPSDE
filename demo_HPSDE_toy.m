%script for testing policy search with clustering
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


%== comment out one of the reward funtion types ==
%RewardType = '_RewardDouble';
%RewardType = '_DoubleCourve';
RewardType = '_RewardTripleSplit';

%== comment out one of the option policy update types ==
PolicyUpdate = 'RWR';
%PolicyUpdate = 'REPS';

%== comment out one of the gating policy types ==
%GatePolicyType = 'gp';
GatePolicyType = 'soft';

%== set the feature type ==
%0: linear feature, 1: exponential feature. 
%exponential feature is recommended for "DoubleCurve"
feature = 0;  

% == =set up parameters===
red = [ 1 0 0];
green = [0 1 0 ];
blue = [ 0 0 1];


if strcmp( RewardType, '_RewardDouble')
    RewardFunc = @RewardDouble;
    Amax = 2;
    n_ini = 600;
    ClusterSeedsNum = 10;
elseif strcmp( RewardType, '_DoubleCourve')
    RewardFunc = @RewardDoubleCurve;
    Amax = 2;
    n_ini = 600;
    ClusterSeedsNum = 10;
elseif strcmp( RewardType, '_RewardTripleSplit')
    RewardFunc = @RewardTripleSplit;
    Amax = 4;
    n_ini = 600;
    ClusterSeedsNum = 10;
elseif strcmp( RewardType, '_RewardSingle')
    RewardFunc = @RewardSingle;
    Amax = 2;
    n_ini = 300;
    ClusterSeedsNum = 5;
end

GridRange = [ 0; 2; 0; Amax  ];
gridsize = 0.05;

ActionDim = 1;
ContextDim = 1;

if strcmp( GatePolicyType, 'gp')
    hrl = HPSDEGP(ClusterSeedsNum, ContextDim, ActionDim, PolicyUpdate);
else
    hrl = HPSDEsoft(ClusterSeedsNum, ContextDim, ActionDim, PolicyUpdate);
end

hrl.SelectFeatureType( feature ); 

VisualizeReward( RewardFunc, GridRange, gridsize ); 
 axis( [0 2 0 Amax] );  view(2);

clusteringNum = 3; %
updateNum = 3; %3
sampleNum = 50;
% == =end of setting up parameters===

ContextIni = 2 * rand(n_ini, 1);
ActionIni = Amax * rand(n_ini, 1);
RewardIni = zeros( n_ini, 1 );

for i = 1:n_ini
    RewardIni( i, 1 ) = RewardFunc( ContextIni( i, 1 ), ActionIni( i, 1 ) );
end

hrl.StoreSamples( ContextIni, ActionIni, RewardIni );
p_ini = unifpdf( Amax*0.5, 0, Amax ) * ones( n_ini, 1 );
hrl.StoreSampleProb( p_ini );

selectedOption = [];
OptionNum = [];
begin_t = cputime;

%====Main part of learning a hierarchical policy with HPSDE====
for i = 1:clusteringNum
    message = [ 'iteration num: ', sprintf( '%d', i)  ];
    disp( message );
    
    hrl.ClusterSamples();
    
    for k = 1:updateNum
        hrl.UpdatePolicySet();

        for j = 1:sampleNum
            test_context = 2 * rand(1, 1);

            [action_new, option, p] = hrl.GenerateAction( test_context  );
            selectedOption = [selectedOption, option];
            reward_new = RewardFunc( test_context, action_new );
            hrl.StoreSamples( test_context, action_new, reward_new);
            hrl.StoreSampleProb( p );
            hrl.PolicySet( option, 1 ).StoreSamples( test_context, action_new, reward_new );

            policyNum = hrl.CheckOptionNum();
            OptionNum = [ OptionNum;  policyNum];
            
        end % end of sampling
        
            %Visualize the samples 
            if ( i == clusteringNum && k == updateNum ) || ( i ==1 && k == 1)
                VisualizeReward( RewardFunc, GridRange, gridsize ); 
                hold on;
                axis( [0 2 0 Amax] );  view(2);
                axis off;
                a = linspace(0, 1,  ClusterSeedsNum);
                for o = 1:ClusterSeedsNum
                    height = 2 * ones( size(hrl.PolicySet( o, 1 ).ActionSet) );
                    color = red * 0.5 * ( sin( (o-1)/(ClusterSeedsNum - 1) *pi ) + 1 ) + green * 0.5 * ( sin( (o-1)/(ClusterSeedsNum - 1) *pi + 2/3*pi ) + 1)  +   blue * 0.5  * ( sin( (o-1)/(ClusterSeedsNum - 1) *pi + 4/3*pi ) + 1 );
                    if o == 1
                         plot3( hrl.PolicySet( o, 1 ).ContextSet(:, 1), hrl.PolicySet( o, 1 ).ActionSet(:, 1), height, 'Marker', 'o',  'LineStyle', 'none',  'Color',  'r');
                    elseif o == 2
                        plot3( hrl.PolicySet( o, 1 ).ContextSet(:, 1), hrl.PolicySet( o, 1 ).ActionSet(:, 1), height, 'Marker', 'o',  'LineStyle', 'none',  'Color',  'b');
                    elseif o == 3
                        plot3( hrl.PolicySet( o, 1 ).ContextSet(:, 1), hrl.PolicySet( o, 1 ).ActionSet(:, 1), height, 'Marker', 'o',  'LineStyle', 'none',  'Color',  'g');
                    elseif o == 4
                         plot3( hrl.PolicySet( o, 1 ).ContextSet(:, 1), hrl.PolicySet( o, 1 ).ActionSet(:, 1), height, 'Marker', 'o',  'LineStyle', 'none',  'Color',  'c');
                    elseif o == 5
                         plot3( hrl.PolicySet( o, 1 ).ContextSet(:, 1), hrl.PolicySet( o, 1 ).ActionSet(:, 1), height, 'Marker', 'o',  'LineStyle', 'none',  'Color',  'm');
                    else  
                        plot3( hrl.PolicySet( o, 1 ).ContextSet(:, 1), hrl.PolicySet( o, 1 ).ActionSet(:, 1), height, 'Marker', 'o',  'LineStyle', 'none',  'Color',  color);
                    end
                end
                hold on;
            end
        
    end % end of policy update
end % end of clustering

end_t = cputime;
mess = [ 'total learning time: ', sprintf( '%d', end_t - begin_t ), ' [s]' ];
disp(mess);

f0 = figure;
plot( hrl.RewardSet );

set(gca,'FontName','Times New Roman');
set(gca,'FontSize',18);
xlabel('Number of rollout', 'FontSize',18);
ylabel('Return', 'FontSize',18);

f2 = figure;
plot(selectedOption, 'o');
set(gca,'FontName','Times New Roman');
set(gca,'FontSize',18);
xlabel('Number of rollout', 'FontSize',18);
ylabel('Selected options', 'FontSize',18);


%==== Visualize the learned option policies ====
VisualizeReward( RewardFunc, GridRange, gridsize ); 
hold on;
axis( [0 2 0 Amax] );  view(2);
axis off;
 
x = linspace(0, 2, 41);
policyNum = hrl.CheckOptionNum();
Actions = zeros( policyNum, 41 );
 
 for i = 1:policyNum
     if ~isempty( hrl.PolicySet( i, 1).ActionSet)
         for j = 1:41
            Actions(i, j ) = hrl.PolicySet( i, 1).GenerateGreedyAction( x( j )  );
         end
         color = red * 0.5 * ( sin( (i-1)/(ClusterSeedsNum - 1) *pi ) + 1 ) + green * 0.5 * ( sin( (i-1)/(ClusterSeedsNum - 1) *pi + 2/3*pi ) + 1)  +   blue * 0.5  * ( sin( (i-1)/(ClusterSeedsNum - 1) *pi + 4/3*pi ) + 1 );
         if i == 1 
            plot( x, Actions(i, :), 'Color', 'r', 'LineWidth', 1 );
         elseif i == 2
             plot( x, Actions(i, :), 'Color', 'b', 'LineWidth', 1 );
         elseif i == 3
             plot( x, Actions(i, :), 'Color', 'g', 'LineWidth', 1 );
         elseif i == 4
             plot( x, Actions(i, :), 'Color', 'c', 'LineWidth', 1 );
         elseif i == 5
             plot( x, Actions(i, :), 'Color', 'm', 'LineWidth', 1 );
         else    
            plot( x, Actions(i, :), 'Color', color );
         end
         
     end
 end
 axis off
 
figure;
plot(OptionNum);

set(gca,'FontName','Times New Roman');
set(gca,'FontSize',18);
xlabel('Number of rollout', 'FontSize',18);
ylabel('Number of learned options', 'FontSize',18);