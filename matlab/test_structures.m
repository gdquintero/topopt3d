function [xstar,u,itotal,itotalint,f0val,time] = test_structures(str,x0,xmin,delta,frmax,prnt,opfil,rmin)

itotal = 0;
itotalint = 0;
xstar = x0;
beta = 0.2;
tt = 0.0;
pmax = 3.0;
p = pmax;
deltap = 1.0;
tol = 1e-3;

if (opfil == 1 || opfil == 2)
   while p <= pmax
         if (p < pmax) 
            maxit = 50;
         else
            maxit = 100;
         end
         [xstar,u,iter,itint,f0val,beta,time] = linseq_struct(str,xstar,xmin,p,delta,frmax,tol,maxit,prnt,opfil,rmin,beta);
         p = p + deltap;
         itotal = itotal + iter;
         itotalint = itotalint + itint;
         tt = tt + time;
   end
end



time = tt;
