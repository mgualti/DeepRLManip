classdef clsPtsFrenet < clsPtsHood
    
    properties
        F; % cell array of Frenet frames. [normal cross(axis,normal) axis];
    end
    
    methods
        
        % Constructor
        % input: pin -> clsPts/clsPtsHood/clsHoodQuads used to initialize this
        % instance
        function pout = clsPtsFrenet(pin)
            pout@clsPtsHood(pin);
            if isa(pin,'clsPtsFrenet')
                for i=1:size(pin.quads,2)
                    pout.F{i} = pin.F{i};
                end
            end
        end
        
        function p = concatenate(p,newp)
            p = p.concatenate@clsPtsHood(newp);
            p.F = [p.F newp.F];
        end
        
        % Evaluate Frenet frame for all points.
        % PREREQUISITE: Must run qq.calCloudNormals and qq.evalBall prior to calling
        % this function.
        function qq = evalFrenetFrame(qq)
            
            if isempty(qq.cloud)
                exit('clsPtsFrenet::evalFrenetFrame error -> must populate this.cloud in order to use this function.');
            end
            
            F{qq.num()} = [];
            qq.normals = zeros(3,qq.num());

%            parfor i=1:qq.num()
            for i=1:qq.num()
                
                hoodPts = qq.cloud.pts(:,qq.nhIdx{i});
                hoodNormals = qq.cloud.normals(:,qq.nhIdx{i});

                F{i} = qq.calcFrame(hoodNormals);                
                normals(:,i) = F{i}(:,1);
                
            end
            qq.F = F;
            qq.normals = normals;
            
        end       
        
        
        function F = calcFrame(q,normals)
            
            M = normals * normals';
            
            try
                [v e] = eig(M);
            catch
                exit('clsPtsFrenet::calcFrame -> some kind of error running eig. This shouldnt happen...');                
            end
            
            esorted = sort(diag(e));
%             normalsratio = esorted(2)/esorted(3);
            [~, imin] = min(diag(e));
            axis = v(:,imin);

            [~, imax] = max(diag(e));
            normal = v(:,imax);
            
            % Ensure that new normal is pointing in direction of
            % constituent normals.
            avgnormal = sum(normals,2);
            avgnormal = avgnormal / norm(avgnormal);
            if avgnormal'*normal < 0
                normal = -normal;
            end
            
            F = [normal cross(axis,normal) axis];
        end
        
        
        % plot quadrics
        function plotFrenets(qq)
            for i=1:qq.num()
                
                normal = qq.F{i}(:,1);
                axis = qq.F{i}(:,3);
                sample = qq.pts(:,1); % pick an arbitrary point from which to draw the axes
                
                plot3([sample(1);sample(1)+0.02*axis(1)], [sample(2);sample(2)+0.02*axis(2)], [sample(3);sample(3)+0.02*axis(3)],'r','LineWidth',2);
                plot3([sample(1);sample(1)+0.02*normal(1)], [sample(2);sample(2)+0.02*normal(2)], [sample(3);sample(3)+0.02*normal(3)],'b','LineWidth',2);
                
            end
        end
        
        % prune elts
        % input: elts2keep -> indices of elts to keep (prune rest)
        function nhout = prune(nh,elts2keep)
            nhout = nh.prune@clsPtsHood(elts2keep);
            nhout.F = cell(1,size(elts2keep,2));
            for i=1:size(elts2keep,2)
                if size(nh.F,2) >= elts2keep(i)
                    nhout.F{i} = nh.F{elts2keep(i)};
                end
            end
        end
        
    end    
    
end
