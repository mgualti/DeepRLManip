classdef clsImageSet < clsHandSet
    
    properties
        
        % properties of image
        imageparams;
        
        % Occluded regions relative to pts in hood. Calculated in calculateOcclusion. 
        occlusionPts;
        
        % image for each elt in handset
        images;
        
    end
    
    methods
        
        function is = clsImageSet(hs,imageparams,DEBUG)
            is@clsHandSet(hs.handparams);
            is = hs.copyFromHandSet(is);
        end


        % Copy all properties from clsHandSet to this class.
        is = copyFromHandSet(hs,is)
            mco = ?clsHandSet;
            propList = mco.PropertyList;
            for i=1:size(propList,1)
                propName = propList(i).Name;
                is.(propName) = hs.(propName);
            end
        end
    

        % Calculate regions in local neighborhood that are occluded from
        % all cam sources. Populates .occlusionPts
        % PREREQUISITE: must have run .populateHands prior to calling this
        % function.
        % output: populates .occlusionPts
        function is = calculateOcclusion(is)

            % This function works as follows. For each cam, we calculate
            % the <shadowVec> that points in the direction of the cam
            % relative to the center of hood.pts. For each pt in hood.pts,
            % we create <numShadowPts> equally spaced along that vec and
            % offset those pts relative to the pt. These will become the
            % occlusionPts. We intersect occlusion pts for each cam and
            % find the pts that are occluded from all cams.

            voxelGridSize = 0.003; % voxel size for pts that fill occluded region.

            % These two parameters are calculated automatically.
            shadowLength = max([is.imageparams.imageOD is.imageparams.imageDepth is.imageparams.imageHeight/2]); % max length of shadow required to fill image window
            numShadowPts = floor(shadowLength/voxelGridSize); % number of points in each shadow line
            
            camSet = find(sum(is.ptsCamSource,2))'; % set of cams that see these points
            ptCenter = mean(is.hood.pts,2); % center of pts in this neighborhood
            
            % Calculate occluded region for each camPos
            for j=1:size(camSet,2)

                % shadowVec is a unit vector pointing from camPos to center
                % of hood.pts
                camPos = is.hood.camPos(:,camSet(j));
                shadowVec = ptCenter - camPos;
                shadowVec = shadowLength * shadowVec / norm(shadowVec);
                
                % Calculate a set of points that cover the occluded region
                shadow = repmat(is.hood.pts,1,1,numShadowPts);
                shadow = permute(shadow,[1 3 2]);
                shadowPts = permute(repmat(rand(numShadowPts,size(shadow,3)),1,1,3),[3 1 2]) .* repmat(shadowVec,1,numShadowPts,size(shadow,3));
                shadow = shadow + shadowPts;
                shadow = reshape(shadow,3,size(shadow,2)*size(shadow,3));

                % Voxelize
                binx = int64(floor(shadow(1,:) / voxelGridSize));
                biny = int64(floor(shadow(2,:) / voxelGridSize));
                binz = int64(floor(shadow(3,:) / voxelGridSize));
                bins = unique([binx;biny;binz]','rows')'; % remove repeated cells
                binsSingleView{j} = bins;
                
                if ismember('clsHandSet.calculateOcclusion',is.DEBUG)
                    shadowPtsVoxelized = [double(bins(1,:)) * voxelGridSize; double(bins(2,:)) * voxelGridSize; double(bins(3,:)) * voxelGridSize];            
                    shadowPtsVoxelized = shadowPtsVoxelized + randn(size(shadowPtsVoxelized)) * voxelGridSize * 0.3;
                    figure;
                    is.hood.plot();
                    hold on;
                    plot3(shadowPtsVoxelized(1,:),shadowPtsVoxelized(2,:),shadowPtsVoxelized(3,:),'m.');
                end
                
            end

            % Intersect set of all occluded regions, i.e. find regions that
            % are occluded from all cam sources.
            bins = binsSingleView{1};
            for j=2:size(camSet,2)
                bins = intersect(bins',binsSingleView{j}','rows')';
            end

            % Convert voxels back into points
            shadowPtsVoxelized = [double(bins(1,:)) * voxelGridSize; double(bins(2,:)) * voxelGridSize; double(bins(3,:)) * voxelGridSize];            
            shadowPtsVoxelized = shadowPtsVoxelized + randn(size(shadowPtsVoxelized)) * voxelGridSize * 0.3;

            if ismember('clsHandSet.calculateOcclusion',is.DEBUG)
                figure;
                is.hood.plot();
                hold on;
                plot3(shadowPtsVoxelized(1,:),shadowPtsVoxelized(2,:),shadowPtsVoxelized(3,:),'m.');
            end
            
        end
        


        % Calculate images
        function calculateImage()
            
        end
        
    end
    
end
