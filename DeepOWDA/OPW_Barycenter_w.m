function [c,T] = OPW_Barycenter_w(sequences,bcnum,seqweight,options) 
  

  options.support_size = bcnum;
  nIterSink = 50;
  the = 0.8;
  
  %seqnum = size(sequences,2); 
  seqnum = length(sequences);
  if isempty(seqweight)
      seqweight = ones(seqnum,1)./seqnum;
  end
  
  stride = zeros(1,seqnum);
  supp = [];
  w = [];
  supp_sw = [];
  for i = 1:seqnum
      stride(i) = size(sequences{i},1);
      supp = [supp sequences{i}'];
      temp_w = ones(1,stride(i))/stride(i);
      w = [w temp_w];
      supp_sw = [supp_sw seqweight(i)*sequences{i}'];
  end
  
  
  
  d = size(supp,1);
  n = length(stride);
  m = length(w);
  posvec=[1,cumsum(stride)+1];

%   if isempty(c0)
%     c=centroid_init(stride, supp, w, options);
%   else
%     c=c0;
%   end
  c = centroid_init_avealign(sequences, options.support_size);
  support_size=length(c.w);
  clear sequences;
  
  Pcont = [];
  Scont = [];
  for t_count = 1:seqnum
      Ptemp = zeros(support_size,stride(t_count));
      Stemp = Ptemp;
      mid_para = sqrt((1/(support_size^2) + 1/(stride(t_count)^2)));
      for i = 1:support_size
          for j = 1:stride(t_count)
              temp_dis = abs(i/support_size - j/stride(t_count))/mid_para;
              Ptemp(i,j) = exp(-temp_dis^2/(2*options.delta^2))/(options.delta*sqrt(2*pi));
              Stemp(i,j) = options.lambda1/((i/support_size-j/stride(t_count))^2+1);
          end
      end
      Pcont = [Pcont Ptemp];
      Scont = [Scont Stemp];
  end
  %Pcont
  %Scont
  

  %save cstart.mat
  %load(['cstart' num2str(n) '-' num2str(avg_stride) '.mat']);
  %return;
  

  spIDX_rows = zeros(support_size * m,1);
  spIDX_cols = zeros(support_size * m,1);
  for i=1:n
      [xx, yy] = meshgrid((i-1)*support_size + (1:support_size), posvec(i):posvec(i+1)-1);
      ii = support_size*(posvec(i)-1) + (1:(support_size*stride(i)));
      spIDX_rows(ii) = xx';
      spIDX_cols(ii) = yy';
  end
  spIDX = repmat(speye(support_size), [1, n]);
  
  % initialization
  Dcont = pdist2(c.supp', supp', 'sqeuclidean');
  %Dcont(1:10)
  Dcont = Dcont/10;
  %Dcont = Dcont/max(max(Dcont));
  
  nIter = 20000;
  if isfield(options, 'ibp_max_iters')
      nIter = options.ibp_max_iters;
  end
  
%   if isfield(options, 'ibp_vareps')
%       rho = options.ibp_vareps * median(median(pdist2(c.supp', supp', 'sqeuclidean')));
%   else
%       rho = 0.01 * median(median(pdist2(c.supp', supp', 'sqeuclidean')));
%   end
  
  if isfield(options, 'ibp_tol')
      ibp_tol = options.ibp_tol;
  else
      ibp_tol = 1E-4; % no updates of support
  end
  
  %rho = 1;  
  xi = Pcont.*exp((Scont - Dcont)./options.lambda2);
  %xi=exp(-Dcont / rho);  
  xi(xi<1e-200)=1e-200; % add trick to avoid program breaking down
  xi=sparse(spIDX_rows, spIDX_cols, xi(:), support_size * n, m);
  %Dcont
  %full(xi)
  v = ones(m, 1);
  w1=w';
  %fprintf('\n');
  obj=Inf;
  tol=Inf;
  
  for iterin = 1:nIterSink
    w0=repmat(c.w', n, 1);
    u=w0 ./ full(xi*v);
    v=w1 ./ full(xi'*u);
    temp_w_mat = reshape(u .* full(xi * v), support_size, n);
    c.w = ones(1,support_size);
    for temp_count = 1:n
        c.w = c.w.*(temp_w_mat(:,temp_count)').^seqweight(temp_count);
    end
    %c.w = geomean(reshape(u .* full(xi * v), support_size, n), 2)';
    c.w = c.w/sum(c.w);
    %iter
    %sum(c.w)

    if (mod(iterin, 10) == 0)
        tol = norm(reshape(full(spdiags(u, 0, n*support_size, n*support_size) * xi * spdiags(v, 0, m, m) * ones(m,1)) ...
            - w0, support_size, n), Inf); 
        if tol < ibp_tol
            %fprintf('Terminated at iter %d\n',iter);
            break;
        end
    end       
  end
  
  for iter = 1:nIter
    
    if ~isfield(options, 'support_points') %tol < ibp_tol && 
        c_back = c;
        X=full(spIDX * spdiags(u, 0, support_size*n, support_size*n) * xi * spdiags(v, 0, m, m));
        %X
        %size(supp)
        %size(X)
        %oldsupp = c.supp;
        %c.supp = supp * X' ./ repmat(sum(X,2)', [d, 1]);
        %c.supp = (1-the)*c.supp + the* (supp * X' ./ repmat(sum(X,2)', [d, 1]));
        c.supp = (1-the)*c.supp + the* (supp_sw * X' ./ repmat(sum(X,2)', [d, 1]));
        %sum(X,1)
        %sum(X,2)'
        Dcont = pdist2(c.supp', supp', 'sqeuclidean');
        Dcont = Dcont/10;
        %Dcont = Dcont/max(max(Dcont));
        %xi=exp(-Dcont / rho);
        xi = Pcont.*exp((Scont - Dcont)./options.lambda2);
        xi(xi<1e-200)=1e-200; % add trick to avoid program breaking down
        xi=sparse(spIDX_rows, spIDX_cols, xi(:), support_size * n, m);
        v = ones(m, 1);
        last_obj=obj;
        obj=sum(Dcont(:).*X(:))/n;
        if isnan(obj)
            disp('NaN');
        end
        fprintf('\t %d %f\n', iter, obj);   
        if (obj>last_obj)
            c = c_back;
            fprintf('terminate!\n');
            break;
        end
        tol=Inf;
        %sum(c.w)
    end
    
    if (tol < ibp_tol && isfield(options, 'support_points'))
        fprintf('iter = %d\n', iter);
        break;
    end
    
    for iterin = 1:nIterSink
        w0=repmat(c.w', n, 1);
        u=w0 ./ full(xi*v);
        v=w1 ./ full(xi'*u);
        c.w = ones(1,support_size);
        for temp_count = 1:n
            c.w = c.w.*(temp_w_mat(:,temp_count)').^seqweight(temp_count);
        end
        %c.w = geomean(reshape(u .* full(xi * v), support_size, n), 2)';
        c.w = c.w/sum(c.w);
        %iter
        %sum(c.w)

        if (mod(iterin, 10) == 0)
            tol = norm(reshape(full(spdiags(u, 0, n*support_size, n*support_size) * xi * spdiags(v, 0, m, m) * ones(m,1)) ...
                - w0, support_size, n), Inf); 
            if tol < ibp_tol
                fprintf('Terminated at iter %d\n',iter);
                break;
            end
        end       
      end
  end
  
  %posvec
  T = cell(1,n);
  for i=1:n
      T{i} = X(:,posvec(i):posvec(i+1)-1);
  end
end