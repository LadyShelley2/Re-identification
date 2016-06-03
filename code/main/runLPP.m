%% МгдиЪ§Он
clear;clc;close all;
load('../datasets/yale/Yale_32x32.mat');
addpath(genpath('../helpers'));
addpath(genpath('../reductionAlgos'));

%% LPP
options.PCARatio = 0.9;
[eigvector, eigvalue, Y] = LPP(fea, gnd,options);
lpp_correct_rate = correct_rate(Y',gnd',Y',gnd',3);