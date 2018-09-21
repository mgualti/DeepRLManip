classdef clsPtsHood < clsPtsNormals
    
    properties
        
        cloud; % pts in total cloud from which neighborhoods are constructed
        nhIdx; % indices of pts in neighborhoods 
        
    end
    
    methods
        
        % constructor
        % input: pin -> clsPts or clsPtsHood used to initialize this
        % instance
        function pout = clsPtsHood(pin)
            pout@clsPtsNormals(pin);
            if nargin > 0
                if isa(pin,'clsPtsHood')
                    pout = pout.setCloud(pin.cloud);
                    for i=1:size(pin.nhIdx,2)
                        pout.nhIdx{i} = pin.nhIdx{i};
                    end
                end
            end
        end
        
        function p = concatenate(p,newp)
            if isobject(p.cloud) || isobject(newp.cloud) || (size(p.nhIdx,2) > 0) || (size(newp.nhIdx,2) > 0)
                exit('clsPtsHood.concatenate: you need to call clearCloud on both elts prior to calling concatenate.');
            end
            p = p.concatenate@clsPtsNormals(newp);
        end
        
        function p = clearCloud(p)
            p.cloud = [];
            p.nhIdx = [];
        end
        
        function p = setCloud(p,cloud)
            p.cloud = cloud;
            p.nhIdx = []; % resetting the cloud automatically resets neighborhoods
        end
        
        % Calculate indices of points within <radius> of each point. This
        % function uses the matlab KD-tree.
        % input: radius
        % output: <this>.nhIdx -> indices of pts in neighborhood
        function pq = evalBall(pq,radius)
            pq.nhIdx = rangesearch(pq.cloud.pts',pq.pts',radius)';
        end
        
        % Calculate indices of points within <radius> tube of each point
        % relative to <source>
        % input: radius -> tube radius
        %        thresdist -> amount in front of point that is "allowed"
        %                       before it gets included.
        %        source -> position of camsource.
        % output: <this>.nhIdx -> indices of pts in cone
        function p = evalTube(p,radius,thresdist,source)
            nhIdx{p.num()}=[];
            parfor i=1:p.num()
                nhIdx{i} = p.cloud.getTubeIndices(source,p.pts(:,i),radius,thresdist);
            end
            p.nhIdx = nhIdx;
        end

        % Calculate indices of points with neighborhoods (nhIdx) >
        % <hoodThres>.
        % input: hoodThres -> min num of elts in neighborhood required in
        %                       order for this elt to be considered "large".
        function idxLarge = getLargeNeighborhoods(p,hoodThres)
            idx = zeros(1,p.num());
            for i=1:p.num()
                if size(p.nhIdx{i},2) > hoodThres
                    idx(i) = 1;
                end
            end
            idxLarge = find(idx);
        end
        
        
        
        % Convert a set of cams into a set of stereo cams. The effect is to
        % double the number of cams. Each one cam is replaced w/ two
        % stereo cams in sequential order. After you run this function, you
        % must also run updateCamVisibility again to update ptsCamSource.
        % input: baseline -> 1/2 dist between cams
        function pp = makeCamsStereo(p,baseline)
            
            if baseline <= 0
                exist('clsPtsHood:makeCamsStereo: baseline must be positive.');
            end
            
            camPos = zeros(3,size(p.camPos,2)*2);
            for i=1:p.numCams()
                
                % Get a random unit vector orthogonal to the direction to
                % the center of the object.
                n = p.camPos(:,i);
                randInPlane = 0;
                while (norm(randInPlane) == 0)
                    randInPlane = (eye(3) - ((n*n') / (n'*n))) * randn(3,1);
                end
                randInPlane = baseline * randInPlane / norm(randInPlane);
                
                camPos(:,(i-1)*2 + 1) = p.camPos(:,i) + randInPlane;
                camPos(:,(i-1)*2 + 2) = p.camPos(:,i) - randInPlane;
                
%                 figure;
%                 plot3([0;p.camPos(1,i)],[0;p.camPos(2,i)],[0;p.camPos(3,i)],'b');
%                 hold on;
%                 plot3([p.camPos(1,i);p.camPos(1,i) + randInPlane(1)],[p.camPos(2,i);p.camPos(2,i) + randInPlane(2)],[p.camPos(3,i);p.camPos(3,i) + randInPlane(3)],'r');
%                 plot3([p.camPos(1,i);p.camPos(1,i) - randInPlane(1)],[p.camPos(2,i);p.camPos(2,i) - randInPlane(2)],[p.camPos(3,i);p.camPos(3,i) - randInPlane(3)],'r');
                
            end
            pp = p;
            pp.camPos = camPos;
            pp.ptsCamSource = sparse(zeros(size(pp.camPos,2),size(pp.pts,2)));
        end

        
        % Get points not occluded wrt to their cam source.
        % input: tuberadius -> radius of tube between sample and point that
        %        thresdist -> amount in front of point that is "allowed"
        %                       before it gets included.       
        %        thresnum -> num points required in front before pruning
        %                       occurs. (Defines the aggressiveness of
        %                       pruning.)
        % output: elts2keep -> points that are not occluded
        function p = updateCamVisibility(p,tuberadius,thresdist)
            for i=1:p.numCams()
                pTemp = evalTube(p,tuberadius,thresdist,p.camPos(:,i));
                notOccludedIdx = cellfun('isempty',pTemp.nhIdx);
                p.ptsCamSource(i,:) = notOccludedIdx;
            end
        end

        
        % Evaluate quadrics and normals for <this>.
        % input: radius -> (optional) radius to be used for <evalBall>
        %        calcQuads -> (binary) 0 just calculate normals. 1 calculate
        %                       normals and axis.
        % output: same as evalQuadrics()
        function p = calcNormals(p)
            
            % calculate normals for each point in <this>
            ptsInNeighborhood = 30;
            pc = pointCloud(p.pts');
            normals = pcnormals(pc, ptsInNeighborhood)';
            p.normals = normals;
            
			% reverse direction of normals that are not pointing toward at least one camera
			normals2Reverse = ones(1,size(normals,2)); % this will be binary
            for i=1:p.numCams() % iterate through all cams
            
				% get pts and normals associated with this cam
				idxThisCam = find(p.ptsCamSource(i,:));
				ptsThisCam = p.pts(:,idxThisCam);
				normalsThisCam = normals(:,idxThisCam);

				% get vectors from pts back to cam
				posThisCam = p.camPos(:,i);
				numThisCam = sum(p.ptsCamSource(i,:));
                pts2Cam = repmat(posThisCam,1,numThisCam) - ptsThisCam;
				
				% figure out which normals are pointing toward cam
				dotPtsNormals = dot(pts2Cam, normalsThisCam);
 				pointing2ThisCam = dotPtsNormals > 0;
				
				% don't reverse those normals
				normals2Reverse(1,idxThisCam(1,pointing2ThisCam)) = 0;
			end
			
			p.normals = normals .* repmat((1-2*normals2Reverse),3,1);
        end
        
        
        % Calculate normals for <this>. When done, clear everything besides
        % the normal information.
        function p = calcNormalsClear(p)
            p = p.calcNormals();
            p.cloud = [];
            p.nhIdx = [];
        end
        

        % Calculate surface normals for all pts in this.cloud by
        % converting it from clsPtsHood to clsHoodQuads and evaluating all
        % quadrics. This is a computationally expensive operation for large
        % clouds.
        function p = calcCloudNormals(p)
            p.cloud = clsPtsHood(p.cloud); % convert to clsPtsHood to do normals calculation
            p.cloud = p.cloud.calcNormalsClear();
            p.cloud = clsPtsNormals(p.cloud); % reduce to clsPtsNormals to save memory
        end


        function p = cloud2Pts(p)
            p.cloud = clsPts(p.cloud);
        end

        function p = voxelizeCloud(p,gridsize)
            p = cloud2Pts(p);
            p.cloud = p.cloud.voxelize(gridsize);
        end
        
        % Delete all points that are not "generated" by a cam source in
        % <camSet> (I call this "ablation"). Do the same for this.cloud.
        % IMPORTANT: Must call evalBall() or similar AFTER calling this
        % function to get this.nhIdx correct.
        % input: camSet -> 1xn vector of cam sources
        function pout = ablate(p,camSet)
            pout = p.ablate@clsPtsNormals(camSet);
%             pout.cloud = pout.cloud.ablate@clsPtsNormals(camSet);
            pout.cloud = pout.cloud.ablate(camSet);
        end
        
        % prune elts
        % input: elts2keep -> indices of elts to keep (prune rest)
        function nhout = prune(nh,elts2keep)
            nhout = nh.prune@clsPtsNormals(elts2keep);
            nhout.nhIdx = cell(1,size(elts2keep,2));
            for i=1:size(elts2keep,2)
                if size(nh.nhIdx,2) >= elts2keep(i)
                    nhout.nhIdx{i} = nh.nhIdx{elts2keep(i)};
                end
            end
        end
        
    end
    
end
