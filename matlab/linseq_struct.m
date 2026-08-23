function [xstar,u,iter,itint,f0val,beta,time] = linseq_struct(str,x,xmin,p,delta,frmax,tol,maxit,prnt,opfil,rmin,beta)
%***********************************************************************************************************************
%[xstar,u,iter,itint,f0val,beta,time] = linseq_struct(str,x,xmin,p,delta,frmax,tol,maxit,prnt,opfil,rmin,beta)
%
%This function solves the topology optimization problem for structures, using a globally convergent version of 
%the Sequential Linear Programming (SLP) algorithm*, proposed by Gomes and Senne.
%
%*Gomes, F. A. M.; Senne, T. A.; An SLP algorithm and its application to
%topology optimization. Computational and Applied Mathematics, Vol. 30, N. 1, pp. 53-89, 2011.
%
%
%INPUT PARAMETERS:
%-----------------
%str: Structure with the problem data.
%x: Initial point for the SLP algorithm (initial element densities).
%xmin: Small positive value that the element densities can assume, to avoid numerical instabilities.
%p: Penalty parameter of the SIMP model for the element densities,
%   or penalty parameter of the sinh filter for the volume constraint. 
%delta: Initial trust region radius for the SLP algorithm.
%frmax: Volume fraction of the rectangular domain that the optimal
%       structure can contain. 
%tol: Minimum value for the step norm obtained by the SLP algorithm (stopping criterion).
%maxit: Maximum number of iterations that the algorithm can perform (stopping criterion).
%prnt: Number of times that the outer iteration data and the topology will be
%      printed on the screen. 
%opfil: Parameter that indicates what filter is used.
%       opfil = 0 (no filter)
%       opfil = 1 (mean density filter)
%       opfil = 2 (dilation filter)
%       opfil = 3 (erosion filter)
%       opfil = 4 (sinh filter)
%rmin: Filter radius.
%beta: Initial value of the parameter used in the dilation and erosion
%      filters. If another filter is used, set beta = 0. 
%
%OUTPUT PARAMETERS:
%------------------
%xstar: Approximated optimal solution of the topology optimization problem.
%iter: Total number of outer iterations performed by the SLP algorithm to find
%      the approximated optimal solution of the topology optimization
%      problem.
%itint: Total number of inner iterations performed by the SLP algorithm to find
%       the approximated optimal solution of the topology optimization
%       problem.
%f0val: Approximated optimal value of the objective function.
%beta: Parameter used in the dilation and erosion filters. 
%time: Total time taken by the SLP algorithm to find the approximated
%      optimal solution of the topology optimization problem. 
%***********************************************************************************************************************

%Starting to count the time:
tic; 

%Computing some constants:
n = str.nelem;
sinhp = sinh(p);
spinv = 1/sinhp;
nonempty = 1:str.nelem;
V = str.b*str.h*str.e;
velem = (V/str.nelem)*ones(str.nelem,1);
Vdom = sum(velem);
Vmax = frmax*Vdom;
csnhp = spinv/Vmax;
lpoptions = optimset('Display','off');
%lpoptions2 = optimset('Display','off','LargeScale','off','Simplex','on');
lpoptions2 = lpoptions;
snorm = 1e20;
gpnorm = 1e20;
iter = 0;
itint = 0;
thold1 = 1;
thold2 = 1;
thmax = 1;   
count = 0;
tolcount = 5;

%Computing the neighborhood of each finite element:
if opfil ~= 0
   [N,numviz,weigh,wi,gradxnew] = get_neighborhood(str,rmin,[],nonempty,n,opfil);
else
   N = [];
   numviz = [];
   weigh = [];
   wi = [];
   gradxnew = [];
end

%Applying the filter:
[xfil,gradxdil,gradxer] = apply_filter(str,x,N,weigh,wi,beta,opfil,[],nonempty,n,numviz,xmin);
xdens = xfil;
%[xfil,xdens] = apply_filter(str,x,N,weigh,wi,beta,opfil,[],nonempty,n,numviz,xmin);
%xfil = x;

%Computing the nodal equivalent forces vector of the structure:
f = vetfneq(str);

%Computing the element stiffness matrix:
kel = elem_stiff(str);
kelc = kel(:);

%Computing the matrix that contains the indexes of the degrees of freedom associated to
%each element:
lin = linel(str);

%Choosing the appropriate value of the penalty parameter to compute
%the objective function value. If the sinh filter is used, this penalty
%parameter is always equal to 1. 
if opfil == 4
   pp = 1;
else
   pp = p;
end

%Computing the global stiffness matrix, and applying
%the boundary conditions of the structure:
[row,col,inic] = get_rows_cols(str);
pos = get_pos(str,inic);
K = global_stiff(str,xfil,pp,row,col,inic,kelc,pos);
%K2 = glob_stiff_python(str,xfil,pp,kel,lin);
linap = put_supports(str);
ap = speye(2*str.nnos);
ap1 = ap(linap,:);
ap2 = ap(:,linap);
K(linap,:) = ap1;
K(:,linap) = ap2;
%K2 = glob_stiff_python(str,xfil,pp,kel,lin,linap);
perm = symamd(K);
f2 = f(perm);

%Computing the initial value of the objective function and of the volume
%constraint:
[f0val,b,xnew,xvol,u] = obj_func_str(n,str,xfil,p,f,f2,K,perm,velem,Vmax,opfil,N,weigh,wi,numviz,nonempty,sinhp);

%Computing the gradient vectors of the objective function and of the volume
%constraint:
if opfil == 0 
   A = velem'/Vmax;
   df0dx = grad_func_str(str,xfil,xvol,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,opfil,csnhp,lin);
elseif opfil == 1
       A = grad_volume_mean_density(str,velem,N,weigh,wi,nonempty,n,Vmax,numviz)';
       %%%%%df0dx = grad_func_str(str,xfil,xvol,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,opfil,csnhp,lin);
       df0dx = grad_func_str(str,xfil,xdens,beta,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,weigh,wi,opfil,csnhp,lin);
       %df0dx = grad_func_str(str,xfil,xdens,beta,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,weigh,wi,opfil,csnhp,lin)
else
   [df0dx,A] = grad_func_str(str,xfil,xvol,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,opfil,csnhp,lin);
end

%Normalizing the objective function, the volume constraint and its
%respectives gradient vectors
cfobj = 1.0/norm(df0dx,inf);
cvol = 1.0/norm(A,inf);
%cfobj = 1.0;
%cvol = 1.0;
f0val = cfobj*f0val;
f0valold = f0val;
b = cvol*b;
df0dx = cfobj*df0dx;
A = cvol*A;

%Computing the lower and upper bounds for the LP problem:
sL = max(-delta, xmin-x);
sU = min(delta, 1-x);

%Showing the initial topology of the structure:
if prnt > 0
   figure;
   get_picture(xnew,str.nely,str.nelx);
   iprn = 0;
end

%Solving the topology optimization problem while the stopping criterion is not satisfied:
while ( (gpnorm > tol) && (count < tolcount) && (iter <= maxit) )
      
      if (abs(b) > 1e-10)
         %Normal step:
         [A2,c,s0] = vol_constr_normal(A,b,n);
         sL2 = [max(-0.8*delta, xmin-x); 0.0];
         sU2 = [min(0.8*delta, 1-x); inf];
         [sn,fval,flagn,~,lambda] = linprog(c,[],[],A2,-b,sL2,sU2,s0,lpoptions2);
         sn = sn(1:n);
         if (abs(fval) < 1e-10)
            %Tangent step:
            %[s,fval,flagtg,~,lambda] = linprog(df0dx,[],[],A,-b,sL,sU,sn,lpoptions);
            [s,fval,flagtg,~,lambda] = linprog(df0dx,[],[],A,0,sL,sU,sn,lpoptions);
            if (flagtg ~= 1)
               s = sn;
               flagtg = 0;
               fval = df0dx'*s;
            end
         else
            s = sn;
            flagtg = 0;
            fval = df0dx'*s;
         end    
      else
         %Tangent step:
         sn = zeros(n,1);
         %[s,fval,flagtg,~,lambda] = linprog(df0dx,[],[],A,-b,sL,sU,sn,lpoptions);    
         [s,fval,flagtg,~,lambda] = linprog(df0dx,[],[],A,0,sL,sU,lpoptions);    
         if (flagtg == 1)
            flagn = 0;
         else
            %Normal step:
            [A2,c,s0] = vol_constr_normal(A,b,n);
            sL2 = [max(-0.8*delta, xmin-x); 0.0];
            sU2 = [min(0.8*delta, 1-x); inf];
            [sn,~,flagn,~,lambda] = linprog(c,[],[],A2,-b,sL2,sU2,s0,lpoptions2);
            s = sn(1:n);
            flagtg = 0;
            fval = df0dx'*s;
         end
      end   
      snorm = norm(s,inf);
      xold = x; 
      x = x + s;
      deltaold = delta;
      grfold = df0dx;
      Aold = A;
      bold = b;
     
      %Applying the filter:
      [xfil,gradxdil,gradxer] = apply_filter(str,x,N,weigh,wi,beta,opfil,[],nonempty,n,numviz,xmin);
      xdens = xfil;
      
      %Computing the global stiffness matrix, and applying
      %the boundary conditions of the structure:
      K = global_stiff(str,xfil,pp,row,col,inic,kelc,pos);
      K(linap,:) = ap1;
      K(:,linap) = ap2;
      
      %Computing the objective function and the volume
      %constraint at the current point:
      [f0val,b,xnew,xvol,u] = obj_func_str(n,str,xfil,p,f,f2,K,perm,velem,Vmax,opfil,N,weigh,wi,numviz,nonempty,sinhp);
      f0val = cfobj*f0val;
      b = cvol*b;
      vol = velem'*xnew;
      
      %Computing the predicted reduction of the objective function (predopt) and the 
      %predicted reduction of the infeasibility (predfsb):
      predopt = -fval;
      predfsb = abs(bold) - abs(Aold*s + bold);
      
      %Updating the penalty parameter associated to the merit function:
      thkmin = min(thold1,thold2);
      thklarge = (1 + (1e6/((iter+1)^(1.1))))*thkmin;
      if predopt > 0.5*predfsb
         thksup = 1;
      else
         thksup = (0.5*predfsb)/(predfsb-predopt);
      end
      theta = min(thksup,thklarge);
      theta = min(theta,thmax);
      thold2 = thold1;
      thold1 = theta;
      
      %Computing the predicted reduction of the merit function:
      pred = theta*predopt + (1-theta)*predfsb;
     
      %Computing the actual reduction of the merit function:
      aredopt = f0valold - f0val;
      aredfsb = abs(bold) - abs(b);
      ared = theta*aredopt + (1-theta)*aredfsb;
      
      %Computing the ratio ared/pred:
      apred = ared/pred;
      
      %Testing if the new point will be accepted or rejected:
      if (apred < 0.1)
        
         delta = max(0.25*snorm,0.1*deltaold);
         thmax = theta;
         x = xold;
         f0val = f0valold;
         df0dx = grfold;
         A = Aold;
         b = bold;
         
         %Updating the lower and the upper bounds for the next LP problem:
         sL = max(-delta, xmin-x);
         sU = min(delta, 1-x);
         
         %Counting the number of inner iterations:
         itint = itint + 1;
 
      else
          
         %Updating the trust region radius
         if (apred >= 0.5) 
            delta = min(1.5*deltaold,1.0-xmin);
         elseif ((apred >= 0.2) && (apred < 0.5))
            delta = deltaold;
         else
            delta = 0.25*deltaold;
         end
         thmax = 1.0;
         
         %Updating the lower and the upper bounds for the next LP problem:
         sL = max(-delta, xmin-x);
         sU = min(delta, 1-x);
         
         %Counting the number of inner iterations:
         itint = itint + 1;
         
         %Counting the number of outer iterations:
         iter = iter + 1;
         
         %Updating the value of the objective function:
         f0valold = f0val;
            
         %Updating the gradient vectors of the objective function and of
         %the volume constraint:
         if opfil >= 2
            [df0dx,A] = grad_func_str(str,xfil,xvol,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,opfil,csnhp,lin);
            A = cvol*A;
         else
            %df0dx = grad_func_str(str,xfil,xvol,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,opfil,csnhp,lin);
            df0dx = grad_func_str(str,xfil,xdens,beta,p,u,kel,velem,Vmax,gradxnew,gradxdil,gradxer,numviz,nonempty,n,N,weigh,wi,opfil,csnhp,lin);
         end
         df0dx = cfobj*df0dx;     
         
      end
      
      %Norm of the continuous projected gradient of the Lagrangian onto the
      %box (stopping criterion)
      gradproj = min(1.0,max(xmin,x-df0dx-lambda.eqlin*A'))-x;
      gpnorm = norm(gradproj,inf);
  
      %Counting the number of times that the norm step associated to the
      %SLP algorithm reaches the minimum value given by the user:
      if (snorm <= tol), 
         count = count+1;
      else
         count = 0;
      end
      
      %Showing the data about the inner iteration and the topology of the
      %structure:
      if (prnt>0)
         if (mod(iter,prnt)==0)
            get_picture(xnew,str.nely,str.nelx);
            pause(1e-6);
         end
         if (iprn==0)
            disp([]);
            disp(' iter       ||s||          gpnorm         delta           f0val           ared            pred            fracvol        volume    flagn  flagtg');
            disp('----- --------------- --------------- --------------- --------------- --------------- --------------- --------------- ------------ -----  ------');
         end
         iprn = mod(+1,20);
         fprintf('%5u %+15.7E %+15.7E %+15.7E %+15.7E %+15.7E %15.7E %15.7E %5u %5u',iter,snorm,gpnorm,deltaold,f0val/cfobj,ared,pred,vol/Vdom,vol,flagn,flagtg);
         fprintf('\n');
      end
  
end

%Approximated optimal solution for the topology optimization problem:
xstar = xnew;

%Original value of the objective function:
f0val = f0val/cfobj;

%Showing the optimal topology of the structure and some data:
if (prnt>=0)
   if (prnt==0)
      disp([]);
      disp(' iter       ||s||          gpnorm         delta           f0val           ared            pred            fracvol        volume    flagn  flagtg');
      disp('----- --------------- --------------- --------------- --------------- --------------- --------------- --------------- ------------ -----  ------');
      fprintf('%5u %+15.7E %+15.7E %+15.7E %+15.7E %+15.7E %15.7E %15.7E %5u %5u',iter,snorm,gpnorm,deltaold,f0val/cfobj,ared,pred,vol/Vdom,vol,flagn,flagtg);
      fprintf('\n');
      figure;
   end      
   get_picture(xstar,str.nely,str.nelx);
end

%Total time taken by the SLP algorithm to find the approximated optimal
%solution of the topology optimization problem:
time = toc;


