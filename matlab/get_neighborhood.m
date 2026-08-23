function [N,numviz,weigh,wi,gradxnew] = get_neighborhood(str,rmin,empty,nonempty,kmax,opfil)
%************************************************************************************************************
%[N,numviz,weigh,wi,gradxnew] = get_neighborhood(str,rmin,empty,nonempty,kmax,opfil)
%
%This function obtains the neighborhood of each rectangular finite element.
%
%INPUT PARAMETERS:
%-----------------
%str: Structure with the problem data.
%rmin: Filter radius.
%empty: Vector with the indexes of the finite elements that are always empty.
%nonempty: Vector with the indexes of the finite elements that are nonempty.
%kmax: Number of nonempty elements.
%opfil: Parameter that indicates what filter is used.
%       opfil = 1 (mean density filter)
%       opfil = 2 (dilation filter)
%       opfil = 3 (erosion filter)
%       opfil = 4 (sinh filter)
%    
%OUTPUT PARAMETERS:
%------------------
%N: Matrix where each column contains the neighbors of each finite element.
%numviz: Vector that contains the number of neighbors of each finite element.
%weigh: Matrix where each row contains the weights of each finite element,
%       associated to the mean density filter or to the sinh filter.
%       If the dilation or the erosion filter are used, "weigh" is an empty
%       matrix.
%wi: Vector where each element is the sum of the finite element weights,
%    associated to the mean density filter or to the sinh filter.
%    If the dilation or erosion filter are used, "wi" is an empty
%    vector.
%gradxnew: Matrix where each line contains the coefficients associated to
%          the chain rule for the objective function and volume constraint derivatives,
%          when the mean density filter is used. 
%          If the dilation or the erosion filter are used, "gradxnew" is an empty
%          matrix, because, in this cases, this matrix is obtained in the
%          routine where the filter is applied.
%************************************************************************************************************

ncol = str.nelem;
frmin = floor(rmin);
kmax2 = kmax;
if (rmin == frmin) 
   nrow = (2*rmin + 1)^2;
else
   nrow = (2*frmin + 1)^2;
end
N = zeros(nrow,ncol);
numviz = zeros(str.nelem,1);

if (opfil == 1 || opfil == 2) %Mean density filter (opfil == 1), Sinh filter (opfil == 4)
   distviz = zeros(nrow,ncol);
   for i = 1:str.nelx
       for j = 1:str.nely
           m1 = (i-1)*str.nely + j;
           if isempty(find(empty == m1,1))
              ind = 0;
              kmin = max(i-frmin,1);
              kmax = min(i+frmin,str.nelx);
              lmin = max(j-frmin,1);
              lmax = min(j+frmin,str.nely);
              for k = kmin:kmax
                  for l = lmin:lmax
                      m2 = (k-1)*str.nely + l;
                      if isempty(find(empty == m2,1))
                         ind = ind + 1;
                         N(ind,m1) = m2;
                         distviz(ind,m1) = sqrt((i-k)^2+(j-l)^2);
                         disp(distviz(ind, m1))
                      end
                  end
              end
              numviz(m1) = ind;
           end
       end
   end
   disp(distviz)
   [weigh,wi] = weight(str,rmin,nonempty,kmax2,numviz,distviz);
   gradxnew = graddens(str,N,weigh,wi,nonempty,kmax2,numviz);
end

% if (opfil == 2 || opfil == 3) %Dilation filter (opfil == 2), erosion filter (opfil == 3)
%    for i = 1:str.nelx
%        for j = 1:str.nely
%            m1 = (i-1)*str.nely + j;
%            if isempty(find(empty == m1,1))
%               ind = 0;
%               kmin = max(i-frmin,1);
%               kmax = min(i+frmin,str.nelx);
%               lmin = max(j-frmin,1);
%               lmax = min(j+frmin,str.nely);
%               for k = kmin:kmax
%                   for l = lmin:lmax
%                       m2 = (k-1)*str.nely + l;
%                       if isempty(find(empty == m2,1)) 
%                          ind = ind + 1;
%                          N(ind,m1) = m2;
%                       end
%                   end
%               end
%               numviz(m1) = ind;
%            end
%        end
%    end 
%    weigh = [];
%    wi = [];
%    gradxnew = [];
% end


