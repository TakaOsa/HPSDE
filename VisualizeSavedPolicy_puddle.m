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
addpath('DMPtask');

load('hrl_puddle.mat' );

draw_solution_puddle;
