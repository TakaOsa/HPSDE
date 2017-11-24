function [ f ] = LinearFeature( s )
%LINEARFEATURE

    [n, ~] = size(s);
    f = [ s, ones( n, 1 ) ];

end

