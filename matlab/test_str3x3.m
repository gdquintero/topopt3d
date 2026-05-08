function [xstar,u,itotal,itotalint,f0val,time] = test_str3x3

load str3x3;
str = str3x3;
x0 = 0.4*ones(9,1);
xmin = 0.001;
delta = 0.1;
frmax = 0.4;
prnt = 2000;
opfil = 1;
rmin = 1.5;
[xstar,u,itotal,itotalint,f0val,time] = test_structures(str,x0,xmin,delta,frmax,prnt,opfil,rmin);
xstar



