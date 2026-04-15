function K = global_stiff(str,x,p,row,col,inic,kelc,pos)
%**************************************************************************
%K = global_stiff(str,x,p,row,col,inic,kelc,pos)
%
%This function returns the global stiffness matrix.
%
%INPUT PARAMETERS:
%-----------------
%str = Structure with the problem data.
%x = Element densities vector.
%p = Penalty parameter of the SIMP model for the element densities.
%row = Vector that contains the indexes of the rows associated to 
%      nonzero elements of the global stiffness matrix.
%col = Vector that contains the indexes of the columns associated to 
%      nonzero elements of the global stiffness matrix.
%inic = Vector where each element represents the number minus 1 of the  
%       first nonzero element of each column of the  global stiffness 
%       matrix.
%kelc = Element stiffness matrix, converted in a vector.
%pos = matrix of order 64 x nelem that contains the positions of the
%      nonzero elements of the global stiffness matrix. 
%
%OUTPUT PARAMETER:
%-----------------
%K = Global stiffness matrix, stored as a sparse matrix.
%**************************************************************************

%**************************************************************************
%Defining some constants:
%**************************************************************************

ny = str.ny;
nx = str.nx;
nely = str.ny-1;
nelem = (nx-1)*(ny-1);
nndes = nx*ny;
kk = zeros(inic(2*nndes+1),1);

%**************************************************************************
%Obtaining the global stiffness matrix columns associated to the
%first column of elements of a rectangular domain:
%**************************************************************************

kk(pos(:,1)) = kk(pos(:,1)) + (x(1)^p)*kelc;

% K1 = sparse(row,col,kk);


for i = 2:nely-1
    kk(pos(:,i)) = kk(pos(:,i)) + (x(i)^p)*kelc;
end

kk(pos(:,nely)) = kk(pos(:,nely)) + (x(nely)^p)*kelc;

% K2 = sparse(row,col,kk);

%**************************************************************************
%Obtaining the global stiffness matrix columns associated to the
%intermediate columns of elements of a rectangular domain:
%**************************************************************************

for j = 2:(nx-2)

    kk(pos(:,(j-1)*nely+1)) = kk(pos(:,(j-1)*nely+1)) + (x((j-1)*nely+1)^p)*kelc;
  
    for i = 2:nely-1
        kk(pos(:,(j-1)*nely+i)) = kk(pos(:,(j-1)*nely+i)) + (x((j-1)*nely+i)^p)*kelc;
    end
  
    kk(pos(:,j*nely)) = kk(pos(:,j*nely)) + (x(j*nely)^p)*kelc;

end

%**************************************************************************
%Obtaining the global stiffness matrix columns associated to the
%last column of elements of a rectangular domain:
%**************************************************************************

kk(pos(:,nelem-nely+1)) = kk(pos(:,nelem-nely+1)) + (x(nelem-nely+1)^p)*kelc;

for i = 2:nely-1
    kk(pos(:,nelem-nely+i)) = kk(pos(:,nelem-nely+i)) + (x(nelem-nely+i)^p)*kelc;
end

kk(pos(:,nelem)) = kk(pos(:,nelem)) + (x(nelem)^p)*kelc;

%**************************************************************************
%Generating the global stiffness matrix:
%**************************************************************************

K = sparse(row,col,kk);

