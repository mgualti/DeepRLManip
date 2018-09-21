% clsPtsNormals extends clsPts slightly by including normals. 
%
% One weird thing about this class is that there is *not method for 
% computing the normals*! This is because normals get computed in 
% clsPtsHood and placed in <this>.normals. But, we encode them here because 
% we want a lightweight representation of the points and normals.
%
classdef clsPtsNormals < clsPts
    
    properties
        normals;
    end
    
    methods
        
        function pout = clsPtsNormals(pin)
            pout@clsPts(pin);
            if nargin > 0
                if isa(pin,'clsPtsNormals')
                    pout.normals = pin.normals;
                end
            end
        end
        
        function p = concatenate(p,newp)
            p = p.concatenate@clsPts(newp);
            p.normals = [p.normals newp.normals];
        end

        function pp = transform(pp,R,p)
            pp = pp.transform@clsPts(R,p);
            pp.normals = R*pp.normals;
        end
        
        % prune elts
        % input: elts2keep -> indices of elts to keep (prune rest)
        function pout = prune(pin,elts2keep)
            pout = pin.prune@clsPts(elts2keep);
            if size(pin.normals,2) == size(pin.pts,2)
                pout.normals = pin.normals(:,elts2keep);
            end
        end
        
        % plot surface normals
        % input: num2plot -> num normals to plot. Subsample to this number. -1 denotes to plot all
        %                    normals. Default=-1
        function plotNormals(q,num2plot)
            if nargin < 2
                num2plot = -1;
            end
            
            %q.plot();
            if size(q.normals,2) > 0
                set2plot = 1:q.num();
                if num2plot > 0
                    set2plot = randperm(q.num(), num2plot());
                end
                for j=1:size(set2plot,2)
                    i = set2plot(j);
                    plot3([q.pts(1,i);q.pts(1,i)+0.02*q.normals(1,i)], ...
                          [q.pts(2,i);q.pts(2,i)+0.02*q.normals(2,i)], ...
                          [q.pts(3,i);q.pts(3,i)+0.02*q.normals(3,i)], ...
                          'b','LineWidth',1);
                end
            end
        end
        
        
    end
    
end
