% clsPts is the fundamental class for encoding a set of points in a point
% cloud. This class is inherited by clsPtsNormals.
% 
% This class assumes that you know the viewpoints from which each point was
% seen. This may sound a bit weird, but it helps later on when we figure
% out which parts of the space are occluded from the camera. <camPos>
% denotes the total set of viewpoints. <ptsCamSource> associates each point
% with one or more of these sources.
classdef clsPts
    
    properties
        
        pts; % 3xn matrix of pts
        ptsCamSource; % m x n binary matrix. A one in position (i,j) denotes that point j can be seen from cam i.
        camPos; % 3 x m matrix of numCams camera positions
        colors; % 3xn matrix of RGB colors (UINT8)
        
    end
    
    methods
        
        function p = clsPts(pin)
            p.pts = [];
            p.ptsCamSource = [];
            p.camPos = [];
            
            if nargin > 0
                p = p.addCamPos(pin.camPos);
                p = p.addPts(pin.pts,pin.ptsCamSource,pin.colors);
            end
        end
        

        function p = concatenate(p,newp)
            if ~isequal(p.camPos,newp.camPos)
                exit('clsPts.concatenate: camPos must be identical for two classes being concatenated.');
            end
            p.pts = [p.pts newp.pts];
            p.ptsCamSource = [p.ptsCamSource newp.ptsCamSource];
            p.colors = [p.colors newp.colors];
        end
        
        % Plot the points.
        % input: colr -> letter specifying color (optional; defaults to
        % blue). Enter 'orig' to use original colors.
        %        dontshowsource -> binary flag that denotes not to show
        %        source. defauts to 1.
        function plot(p,colr,dontshowsource)
            
            if nargin < 3
                dontshowsource = 1;
            end
            
            if nargin < 2
                colr = 'orig';
            end
            
            if strcmp(colr,'orig')
                if (p.num == size(p.colors,2))
                    pcshow(p.pts',p.colors','MarkerSize',100);
                else
                    colr = 'b';
                end
            else
%                 pcshow(p.pts',colr,'MarkerSize',200);
                pcshow(p.pts',colr,'MarkerSize',4);
            end

            xlabel('x');
            ylabel('y');
            zlabel('z');
            hold on;
            if ~dontshowsource
                for i=1:size(p.camPos,2)
                    plot3(p.camPos(1,i), p.camPos(2,i), p.camPos(3,i), 'rx', 'MarkerSize', 5);
                end
            end
            axis equal;
        end

        % Get number of points in this class
        function n = num(p)
            n = size(p.pts,2);
        end
        
        % Get number of cams in this class
        function m = numCams(p)
            m = size(p.camPos,2);
        end
        
        % Get points from the cloud that are nearest center points of faces of convex hull
        function p = getHull(p)
            
            % get the convext hull
            K = convhull(double(p.pts)');
            hullMeans = [mean([p.pts(1,K(:,1)); p.pts(1,K(:,2)); p.pts(1,K(:,3))],1); mean([p.pts(2,K(:,1)); p.pts(2,K(:,2)); p.pts(2,K(:,3))],1); mean([p.pts(3,K(:,1)); p.pts(3,K(:,2)); p.pts(3,K(:,3))],1)];
            
            % get nearest points in cloud
            idx = knnsearch(p.pts', hullMeans');
            
            % prune the pts that are not at the center of the hull
            p = p.prune(idx);
            
        end
        
        % Randomly subsample cloud
        function p = subSample(p,numsamples)
            if p.num() > 0
                if numsamples >= p.num()
                    eltstokeep = 1:p.num();
                else
                    eltstokeep = randperm(p.num(), numsamples);
                end
                p = p.prune(eltstokeep);
            end
        end
        
        % Randomly subsample cloud within a radius
        function p = subSampleRadius(p,numsamples,center,radius)
            if p.num() == 0, return; end
            
            radius2 = radius.^2;
            dist2 = sum((p.pts - repmat(center, [1 p.num()])).^2, 1);
            radiusIdx = find(dist2 <= radius2);
            
            if numsamples >= length(radiusIdx)
                eltstokeep = radiusIdx;
            else
                sampleIdx = randperm(length(radiusIdx), numsamples);
                eltstokeep = radiusIdx(sampleIdx);
            end
            
            p = p.prune(eltstokeep);
        end
        
        % Randomly subsample cloud above a table
        function p = subSampleTable(p,numsamples,tablePosition,tableAxis)
            if p.num() == 0, return; end
            
            up = abs(tableAxis);
            if tableAxis > 0
                aboveTableIdx = find(p.pts(up,:) >= tablePosition(up));
            else
                aboveTableIdx = find(p.pts(up,:) <= tablePosition(up));
            end
            
            if numsamples >= length(aboveTableIdx)
                eltstokeep = aboveTableIdx;
            else
                sampleIdx = randperm(length(aboveTableIdx), numsamples);
                eltstokeep = aboveTableIdx(sampleIdx);
            end
            
            p = p.prune(eltstokeep);
        end
        
        % Load a set of points directly into this class
        % input: pts -> 3xn matrix of pts
        %        camsource -> mxn binary matrix indicating which cam(s) each
        %                       point belongs to.
        % All cam sources used by these points must be set before adding points.
        function p = addPts(p,pts,camsource,colors)
            if (size(pts,2) ~= size(camsource,2)) || (size(camsource,1) ~= p.numCams())
                error('Error in clsPts::addPts: input dimensions not correct.');
            end
            p.pts = [p.pts pts];
            p.ptsCamSource = [p.ptsCamSource sparse(camsource)];
            if nargin > 3
                p.colors = [p.colors colors];
            end
            
        end
            
        % Add one new cam position to the current set.
        % input: camPos -> 3x1 position of a new cam pos.
        function p = addCamPos(p,newCamPos)
            p.camPos = [p.camPos newCamPos];
            
            % add zero padding to bottom ptsCamSource for new cam position
            if p.num() > 0
                p.ptsCamSource = [p.ptsCamSource; zeros(1,p.num())];
            end
        end
        
        
        % Transform pts, camPos by <T>.
        % input: T -> 4x4 homogeneous transform
        function p = transformPts(p,T)
            p.pts = T * [p.pts; ones(1,size(p.pts,2))];
            p.pts = p.pts(1:3,:);
            camPosTemp = T*[p.camPos;1];
            p.camPos = camPosTemp(1:3,1);
        end

        % Transform pts, camPos by R and p.
        function pp = transform(pp, R, p)
            pp.pts = R*(pp.pts - repmat(p,1,pp.num()));
            pp.camPos = R*(pp.camPos - repmat(p,1,pp.numCams()));
        end
        
%         % Transform pts, camPos by <T>.
%         % input: T -> 4x4 homogeneous transform
%         function p = transformPts(p,T)
%             p.pts = T * [p.pts; ones(1,size(p.pts,2))];
%             p.pts = p.pts(1:3,:);
%             camPosTemp = T*[p.camPos;1];
%             p.camPos = camPosTemp(1:3,1);
%         end
        
        % Add pts from file to this class.
        % input: filename -> filename
        %        camnum -> cam number to which these pts belong (scalar)
        %        forceunorganized -> (optional) set to 1 to force handling
        %                               as an unorganized point cloud
        function p = loadFromFile(p,file,camnum,forceunorganized)
            
            if nargin < 4
                forceunorganized = 0;
            end
            
            if nargin < 3
                camnum = 1;
            end
            
            pts = loadpcd(file, forceunorganized);            
            if size(pts,1) > 3
                colors = pts(4:6,:);
            else
                colors = zeros(3,size(pts,2));
            end
            pts = pts(1:3,:);
            
            camSource = zeros(p.numCams(),size(pts,2));
            camSource(camnum,:) = 1;
            p = p.addPts(pts,camSource,colors);
        end
        
        % Center points by subtracting off mean
        % input: 
        % output: 
        function p = centerPts(p)
            p.pts = p.pts - repmat(mean(p.pts,2),1,size(p.pts,2));
        end

        
        % Voxelize pts. Updates ptsCamSource as well. If a cell contains
        % points that come from multiple cam sources, ptsCamSource reflects
        % that as multiple sources.
        % input: cellsize -> size of cells
        % output: this.pts, this.ptsCamSource
        function p = voxelize(p,cellsize)

            % voxelize pts associated with each source individually
            for i=1:p.numCams()
                
                % points seen by this cam
                pp = p.pts(:,find(p.ptsCamSource(i,:)));

                % find the cell that each point falls into
                binx = floor((pp(1,:) - min(p.pts(1,:))) / cellsize);
                biny = floor((pp(2,:) - min(p.pts(2,:))) / cellsize);
                binz = floor((pp(3,:) - min(p.pts(3,:))) / cellsize);

                % remove repeated cells
                bins{i} = unique([binx;biny;binz]','rows')';
                
            end
            
            % Calculate voxels accross all cams.
            unionbins = unique(cell2mat(bins)','rows')';
            p.pts = [unionbins(1,:) * cellsize + min(p.pts(1,:)); unionbins(2,:) * cellsize + min(p.pts(2,:)); unionbins(3,:) * cellsize + min(p.pts(3,:))];
            
            % Set ptsCamSource appropriately.
            p.ptsCamSource = zeros(p.numCams(),p.num()); % init w/ all zeros
            for i=1:p.numCams()
                [~,ia,~] = intersect(unionbins',bins{i}','rows'); % indices where points in bins{i} intersect points in unionbins
                p.ptsCamSource(i,ia) = 1; % these indices were seen from this cam
            end
            
            p.colors = []; % voxelize operation does not preserve colors
            
        end
        
        % Calculate indices of pts within <radius> of a line extending from cam to
        % sample.
        % input: source -> position of camera source
        %        sample -> position of a point that refines tube relative
        %                   to camera source
        %        radius -> radius of tube
        %        thresdist -> OPTIONAL argument: back off
        %                       line by <thresdist> from point.
        function indices = getTubeIndices(p,source,sample,radius,thresdist)
            prel = p.pts - repmat(source,1,p.num()); % points relative to camera source
            ax = sample - source; % sample relative to camera source
            unitax = ax/norm(ax);
            distalongax = unitax'*prel;
            distorthax2 = sum((prel - unitax*distalongax).^2,1);
            indices = find((distorthax2 < radius^2) & (distalongax < norm(ax) - thresdist));
        end

        
        % get indices of elts inside ws
        function idx = workspaceFilterIdx(h, ws)
            idx = find((h.pts(1,:) > ws(1)) & (h.pts(1,:) < ws(2)) ...
                & (h.pts(2,:) > ws(3)) & (h.pts(2,:) < ws(4)) ...
                & (h.pts(3,:) > ws(5)) & (h.pts(3,:) < ws(6)));
        end
        
        function p = workspaceFilter(p, ws)
            elts2keep = p.workspaceFilterIdx(ws);
            p = p.prune(elts2keep);
        end
        

        % Delete all points that are not "generated" by a cam source in
        % <camSet> (I call this "ablation").
        % input: camSet -> 1xn vector of cam sources
        function p = ablate(p,camSet)
%             idx = find(camSet);
            pts2Keep = find(sum(p.ptsCamSource(camSet,:),1) > 0);
            p = p.prune(pts2Keep);
        end
        
        % input: indices2keep -> prune all pts except for these
        function p = prune(p,indices2keep)
            if size(p.colors,2) == size(p.pts,2)
                p.colors = p.colors(:,indices2keep);
            end
            p.pts = p.pts(:,indices2keep);
            p.ptsCamSource = p.ptsCamSource(:,indices2keep);
        end
        
    end
    
end
