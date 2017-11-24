function [ R ] = RewardTripleSplit( s, a )
%REWARDDOUBLE outputs the reward based on TWO optimal linear lines

%a = 0.3 * s + 2.4; 
d1 = abs(  -0.3 * s - a + 2.4    ) / sqrt(  0.3^2 + 1.0^2  );

%a = - 0.2 * s + 0.8; 
d2 = abs(  0.2 * s - a + 0.8    ) / sqrt(  0.2^2 + 1.0^2  );

%a = - 0.05 * s + 2.7; 
d3 = abs(  - 0.2 * s - a + 3.5   ) / sqrt(  0.2^2 + 1.0^2  );

%d = min( [d1, d2, d3] );

if s < 0.7
    d = d1;
elseif s < 1.4
    d = d2;
else
    d = d3;
end


R = 10^2 * exp( - d /100 );

if R < 99.5
    R = 99.5;
% else
%     R = 100;
end


end

