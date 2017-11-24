function [ R ] = RewardSingle( s, a )
%REWARDSINGLE outputs the reward based on a single linear function

%a = 0.5 * s + 0.2; 
d = abs(  0.5 * s - a + 0.2    ) / sqrt(  0.5^2 + 1.0^2  );

R = exp( - d / 10 );

end

