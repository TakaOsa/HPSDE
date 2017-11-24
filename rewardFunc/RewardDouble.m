function [ R ] = RewardDouble( s, a )
%REWARDDOUBLE outputs the reward based on TWO optimal linear lines

%a = 0.2 * s + 1.2; 
d1 = abs(  0.2 * s - a + 1.2    ) / sqrt(  0.2^2 + 1.0^2  );

%a = - 0.3 * s + 0.9; 
d2 = abs(  - 0.3 * s - a + 0.9    ) / sqrt(  0.3^2 + 1.0^2  );

d = d1;
if d1 > d2 
    d = d2;
end

R = exp( - d /10 );

end

