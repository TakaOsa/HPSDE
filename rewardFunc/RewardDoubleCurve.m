function [ R ] = RewardDoubleCurve( s, a )
%REWARDDOUBLECURVE 

theta1 = pi * 0.3;
s1 = 1.0;
r1 = s1 / sin(theta1); 
a1 = - cos( theta1 );

d1 = abs( (a - a1)*(a - a1) + (s - s1)*(s - s1) - r1 );

theta2 = pi * 0.15;
s2 = 1.0;
r2 = s1 / sin(theta2); 
a2 = - cos( theta2 ) + 1.0;

d2 = abs( (a - a2)*(a - a2) + (s - s2)*(s - s2) - r2 );  

d = d1;
if d1 > d2 
    d = d2;
end

R = exp( - d /10 );

end

