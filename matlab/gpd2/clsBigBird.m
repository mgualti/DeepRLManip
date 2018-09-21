classdef clsBigBird
    
    properties
        
        bbroot; % path to bigbird directory
        categoryroot; % path to a directory containing category-level subdirectories.
        allCamSources; % set of cam sources to extract from bigbird datset
        voxresolution; % resolution at which to voxelized imported clouds
        
    end
    
    methods
        
%         voxresolution = 0.002;
%         bbroot = '~/projects/object_datasets/bigbird/';        
%         categoryroot = '~/projects/object_datasets/bb_onesource/'; 
%         allCamSources = [0:3:357];
        function bb = clsBigBird(bbroot, categoryroot, allCamSources, voxresolution)
            bb.bbroot = bbroot;
            bb.categoryroot = categoryroot;
            bb.allCamSources = allCamSources;
            bb.voxresolution = voxresolution;
        end
        
        % input: baseline -> 1/2 dist between stereo cam sources; 
        %                    baseline = 0.04
        % input: desiredWidthRange -> 1x2 vector of lower and upper bound
        %                             on object width. The width of a new
        %                             object will be set by uniformly
        %                             randomly selecting from this range.
        function extractCloudsObjectSet3DNet(bb, folderset, baseline, desiredWidthRange)

            % iterate through all categories in <matroot>
            for j=1:size(folderset,2)
        
                folder = [bb.categoryroot folderset{j} '/'];

                % open category list inside each category directory
                fid = fopen([folder 'category_object_list.txt'],'r');
                tempdata = textscan(fid,'%s','Delimiter','\n');
                objectList = tempdata{1};
                fclose(fid);

                % iterate through all objects in a category
                for i=1:size(objectList,1)

                    objUID = objectList{i};

                    plyFile = [bb.bbroot folderset{j} '/' objectList{i} '.ply'];
                    matFileAllClouds = [bb.categoryroot folderset{j} '/' objUID '_allclouds.mat'];
                    matFileClouds4Learning = [bb.categoryroot folderset{j} '/' objUID '_clouds4Learning.mat'];
                    
                    flagObjectUpdated = 0;
                    if ~exist(matFileClouds4Learning, 'file')
                        
                        numCamSources = 20;
                        desiredWidth = rand * (desiredWidthRange(2) - desiredWidthRange(1)) + desiredWidthRange(1);
                        pGroundTruth = getPtsFromPLY(plyFile, numCamSources, desiredWidth, baseline);
                        
                        dualOffsetDeg = 90; % not sure if this is still used
                        pViews = pGroundTruth;
                        save(matFileClouds4Learning,'pViews', 'pGroundTruth', 'dualOffsetDeg', '-v7.3');
                    end


                end

            end

        end
        
        
        % Extract point clouds from big bird dataset. For each object, get
        % point cloud derived from mesh and get set of all raw clouds. Raw
        % point clouds are stored in *_allclouds.MAT. The set of views is
        % subsampled, massaged, and saved to *_clouds4Learning.MAT for the
        % purposes of learning. All files are placed in the corresponding
        % subdirectory of <categoryroot>.        
        % input: folderset -> set of folders under <categoryroot> for which
        %                     this function should extract point clouds.
        %                     Each folder must contain a file called
        %                     "category_object_list.txt" containing a list
        %                     of all objects within that category.
        % output: a set of *_allclouds.MAT files containing point clouds for
        %         all objects and categories. set of *_clouds4Learning.MAT
        %         clouds to be used to generate learning data.
        function extractCloudsObjectSet(bb, folderset, dualOffsetDeg)

            % iterate through all categories in <matroot>
            for j=1:size(folderset,2)
        %     for j=1:1
        
                folder = [bb.categoryroot folderset{j} '/'];

                % open category list inside each category directory
                fid = fopen([folder 'category_object_list.txt'],'r');
                tempdata = textscan(fid,'%s','Delimiter','\n');
                objectList = tempdata{1};
                fclose(fid);

                % iterate through all objects in a category
                for i=1:size(objectList,1)

                    objUID = objectList{i};

                    matFileAllClouds = [bb.categoryroot folderset{j} '/' objUID '_allclouds.mat'];
                    matFileClouds4Learning = [bb.categoryroot folderset{j} '/' objUID '_clouds4Learning.mat'];
                    
                    flagObjectUpdated = 0;
                    if ~exist(matFileAllClouds)
                        
                        % Extract mesh and raw clouds from BB dataset.
                        % Align raw clouds to the mesh.
% %                         [pGroundTruth, ptsView] = bb.extractCloudsSingleObject(objUID,2);
                        
%                         [pGroundTruth, ptsView] = bb.extractCloudsSingleObject(objUID,1);
                        [pGroundTruth, ptsView] = bb.extractCloudsSingleObject(objUID,2);
%                         [pGroundTruth, ptsView] = bb.extractCloudsSingleObject(objUID,3);
%                         [pGroundTruth, ptsView] = bb.extractCloudsSingleObject(objUID,4);
%                         [pGroundTruth, ptsView] = bb.extractCloudsSingleObject(objUID,5);

                        save(matFileAllClouds,'pGroundTruth', 'ptsView', '-v7.3');
                        flagObjectUpdated = 1;
                    else
                        data = load(matFileAllClouds);
                        pGroundTruth = data.pGroundTruth;
                        ptsView = data.ptsView;
                    end
                    
                    if ~exist(matFileClouds4Learning) || flagObjectUpdated
                        numCamSources = 20;
%                         dualOffsetDeg = 53;
                        pViews = bb.getClsPts4Learning(ptsView, numCamSources, dualOffsetDeg);
                        save(matFileClouds4Learning,'pViews', 'pGroundTruth', 'dualOffsetDeg', '-v7.3');
                    end


                end

            end

        end

        % Create a single clsPtsHood cloud containing the views that are to
        % be used to create clsLearning. We are assuming that each view
        % is actually a stereo cloud comprised of two sources. Therefore,
        % there will be 2*numCamSources total camPos in output. 
        % input: ptsView -> 1x120 cell array of clsPts structures taken
        %                   from different views. Each view is 3deg apart
        %                   from the last.
        %        numCamSources -> num of views to create in <pViews>
        %                   output. 
        %        dualOffsetDeg -> amount of offset between dual views in
        %                         stereo cloud. dualOffsetDeg=0 indicates
        %                         that there is only one cam source. A
        %                         positive value indicates the offset for
        %                         dual cameras.
        function pViews = getClsPts4Learning(bb, ptsView, numCamSources, dualOffsetDeg)
            

            pViews = clsPtsHood(clsPts());
            
            % Get <numCamSources> linspaced cams from set of all BB cam
            % poses.
            camSources4LearningIdx = floor(linspace(1,size(bb.allCamSources,2),numCamSources));
            for k=1:size(camSources4LearningIdx,2)

                % Get all cam offsets for this view. This might be a single
                % offset or it might be multiple offsets in the case of a
                % multiview cloud.
                primaryOffset = camSources4LearningIdx(k); % offset of main view into ptsView
                offsetsThisView = primaryOffset;
                if dualOffsetDeg > 0
                    dualOffset = primaryOffset + floor(dualOffsetDeg/3); % offset of second view into ptsView; divide by 3 because views are every 3 deg in BB dataset.
                    if dualOffset > 120
                        dualOffset = dualOffset - 120;
                    end
                    offsetsThisView = [primaryOffset dualOffset];
                end
                
                % All cams associated with this view are placed
                % sequentially in <pViews>. Not so good: you need to remember how many
                % in sequence when using this info.
                for i=1:size(offsetsThisView,2)
                    pViews = pViews.addCamPos(ptsView{offsetsThisView(i)}.camPos);
                    pViews = pViews.addPts(ptsView{offsetsThisView(i)}.pts,repmat([zeros(pViews.numCams()-1,1);1],1,ptsView{offsetsThisView(i)}.num()),ptsView{offsetsThisView(i)}.colors);
                end
                
            end
            
            % Eliminate spurious pts by cropping outside of 1m
            ptsInBoundsIdx = find(sqrt(sum(pViews.pts.^2,1)) < 0.4);
            pViews = pViews.prune(ptsInBoundsIdx);
            
            % Set cloud
            pViews = pViews.setCloud(clsPts(pViews)); % update pViews so that its cloud has the right camsources
        end
        
        
        
        % Get mesh and raw point clouds for a specified object in BB dataset. Raw
        % clouds are aligned with each other and with the mesh using ICP.
        % Obtains raw clouds for each orientation in <bb.orientationList>
        % input: object -> name of object directory
        %        BBcam -> an integer between 1 and 5 indicating which BB cam to
        %                 use. 1 is from the side; 5 is from the top.
        % output: ptsPoisson -> mesh cloud for this object. Each point is
        %                       assigned to one of 20 cams positioned
        %                       around the object.
        %         ptsView -> 1xtotalBBCams cell array aligned to mesh. One
        %                    cloud returned for each view in BB dataset.
        function [ptsPoisson, ptsView] = extractCloudsSingleObject(bb,object,BBcam)

            % Calculate raw cloud mean.
            ptsAll = bb.getAllRawClouds(object);
            ptsAllMean = mean(ptsAll.pts,2);

            % Get "complete" mesh for this object.
            ptsPoisson = getPtsFromPLY([bb.bbroot object '/meshes/poisson.ply'], 20, -1, 0);
%             ptsPoisson = getPtsFromPLY([bb.bbroot object '/textured_meshes/optimized_tsdf_textured_mesh.ply'], 20, -1, 0);

            orientationList = bb.allCamSources;
            
            % load raw clouds into elts of ptsView. Clouds are approximately
            % aligned with mesh in <ptsPoisson>
            orientationListNonZero = zeros(1,size(orientationList,2));
            for i=1:size(orientationList,2)
                ptsView{i} = bb.getSingleRawCloud(object, BBcam, orientationList(i));
                if ptsView{i}.num() > 0
                    ptsView{i}.pts = ptsView{i}.pts - repmat(ptsAllMean,1,size(ptsView{i}.pts,2));
                    orientationListNonZero(i) = 1;
                end
            end

            % Use ICP to get exact transform of <ptsView> to <ptsPoisson>.
            % Transform each cloud in ptsView accordingly.
            icpPerm = randperm(size(orientationList,2),4); % last param is the num of ICPs to run and average in order to get overage offset
            offsetList = [];
            for i=1:size(icpPerm,2)
                if orientationListNonZero(i)
                    [R,T,err] = icp(ptsPoisson.pts,ptsView{icpPerm(i)}.pts,10);
                    offsetList = [offsetList T];
                end
            end
            offset = mean(offsetList,2);
            for i=1:size(orientationList,2)
                if orientationListNonZero(i)
                    ptsView{i}.pts = ptsView{i}.pts + repmat(offset,1,length(ptsView{i}.pts));

                    % Plot first raw cloud just for the heck of it.
                    if i==1
                        figure;
                        ptsPoisson.plot();
                        hold on;
                        ptsView{i}.plot('r');
                    end
                end
            end

        end

        % Return the raw point cloud from BB dataset for cam <NP> and orientation
        % <orientation>. Cloud is rotated into a common reference frame.
        % input: NP -> an int between 1 and 5 indicating BB cam to use
        %        orientation -> an int indicating orientation in degrees. Will fail
        %                       if BB does not have that orientation.
        %        bbroot -> path to bigbird root (~/projects/object_datasets/bigbird/)
        %        object -> string denoting object name
        % output: clsPts structure containing the desired raw cloud.
        function p = getSingleRawCloud(bb, object, NP, orientation)

            % get paths to big bird
            objectroot = [bb.bbroot object];

            % get transforms between depth cameras
            calibrationfilename = [objectroot '/calibration.h5'];
        %     h5disp(calibrationfilename);
            T_NP5_NP{1} = h5read(calibrationfilename,'/H_NP1_from_NP5')';
            T_NP5_NP{2} = h5read(calibrationfilename,'/H_NP2_from_NP5')';
            T_NP5_NP{3} = h5read(calibrationfilename,'/H_NP3_from_NP5')';
            T_NP5_NP{4} = h5read(calibrationfilename,'/H_NP4_from_NP5')';
            T_NP5_NP{5} = eye(4);

            % iterate through cams specified in this list
            cams = [{'NP1'} {'NP2'} {'NP3'} {'NP4'} {'NP5'}];    

            % get paths to PCD and pose files
            pcdfilename = [objectroot '/clouds/' cams{NP} '_' int2str(orientation) '.pcd'];
            posefilename = [objectroot '/poses/NP5_' int2str(orientation) '_pose.h5'];

            % get table tranform for this object
        %     h5disp(posefilename);
            T_NP5_table = h5read(posefilename,'/H_table_from_reference_camera')';

            % init temporary clsPts container
            p = clsPts;
            p = p.addCamPos([0;0;0.0]);

            try % this try is needed in case a file is missing
                p = p.loadFromFile(pcdfilename);

                % transform pts and place into accumulator clsPts
                p = p.transformPts(T_NP5_table*inv(T_NP5_NP{NP}));
            end
        end

        % Create single point cloud from the set of all raw point clouds for a
        % single object instance in the BB dataset. Constituent raw point clouds
        % are transformed into a common reference frame and concatenated together.
        % Cloud output is workspace filtered at +-0.4m in all dimensions.
        % input: bbroot -> path to bigbird root (~/projects/object_datasets/bigbird/)
        %        object -> string denoting object name
        % output: p-> clsPts structure containing points. camPos is set to zero.
        %             ptsCamSource is set to 1 for all points.
        % function pp = getAllBBPts(bbroot, object, voxresolution)
        function pp = getAllRawClouds(bb, object)

            if nargin < 2
                exit('getBBPts error: must supply <bbroot> and <object> as arguments');
            end

            % get paths to big bird
        %     bbroot = '~/projects/object_datasets/bigbird/';
            objectroot = [bb.bbroot object];

            % get transforms between depth cameras
            calibrationfilename = [objectroot '/calibration.h5'];
        %     h5disp(calibrationfilename);
            T_NP5_NP{1} = h5read(calibrationfilename,'/H_NP1_from_NP5')';
            T_NP5_NP{2} = h5read(calibrationfilename,'/H_NP2_from_NP5')';
            T_NP5_NP{3} = h5read(calibrationfilename,'/H_NP3_from_NP5')';
            T_NP5_NP{4} = h5read(calibrationfilename,'/H_NP4_from_NP5')';
            T_NP5_NP{5} = eye(4);

            % initialize clsPts for point accumulation
            pp = clsPts;

            % iterate through cams specified in this list
            cams = [{'NP1'} {'NP2'} {'NP3'} {'NP4'} {'NP5'}];
            for j=1:size(cams,2)

                % get list of files associated with this cam
                cloudfiles = dir([objectroot '/clouds/' cams{j} '*']);
                cloudfileoffset = 2;

                % iterate through PCD files for this cam
                for i=1:size(cloudfiles,1)-cloudfileoffset

                    % get paths to PCD and pose files
                    prefix = cloudfiles(cloudfileoffset+i).name(1:end-4);
                    pcdfilename = [objectroot '/clouds/' prefix '.pcd'];
                    posefilename = [objectroot '/poses/NP5_' prefix(5:end) '_pose.h5'];

                    % get table tranform for this object
            %         h5disp(posefilename);
                    T_NP5_table = h5read(posefilename,'/H_table_from_reference_camera')';

                    % init temporary clsPts container
                    p = clsPts;
                    p = p.addCamPos([0;0;0]);
                    p = p.loadFromFile(pcdfilename);

                    % transform pts and place into accumulator clsPts
                    p = p.transformPts(T_NP5_table*inv(T_NP5_NP{j}));
                    
                    pp = pp.addCamPos(p.camPos);

                    ptsCamSource = zeros(pp.numCams(),p.num());
                    ptsCamSource(pp.numCams(),:) = 1;
                    pp = pp.addPts(p.pts,ptsCamSource);
                    

                end

            end

            % filter output cloud
            pp = pp.workspaceFilter([-0.4 0.4 -0.4 0.4 -0.4 0.4]);
            pp = pp.voxelize(bb.voxresolution);

        end
        
        
        % ********************** OLD *****************************
        
        % This function creates the clsPts needed to do ablation and to
        % create training/testing data. Given a set of raw views as input,
        % this function contactentates <numCamSources> views into a single
        % clsPtsHood class where each view is associated with a different
        % ptsCam.
        % input: ptsView -> a cell array of clsPts structures where there
        %                   is one structure for each elt of bb.allCamSources
        %        numCamSources -> num cam sources to be concatenated into
        %                         <pViews>. recommend: numCamSources = 20;
        %        dualOffsetDeg -> degree offset between two views in dual
        %                         cloud. recommend: dualOffsetDeg = 53; Set
        %                         to 0 if a single cloud is to be used.
        % output: pViews -> a single clsPtsHood structure containing
        %                   <numCamSources> orientations concatentated
        %                   together. Each orientation is associated with a
        %                   different camSource.
        %                   
        function pViews = getClsPts4LearningOld(bb, ptsView, numCamSources, dualOffsetDeg)
            

            pts = [];
            ptsColors = [];
            ptsCam = [];
            camPos = [];
            
            % Get <numCamSources> linspaced cams from set of all BB cam
            % poses.
            camSources4LearningIdx = floor(linspace(1,size(bb.allCamSources,2),numCamSources));
            for k=1:size(camSources4LearningIdx,2)

                % get offset of this view
                primaryOffset = camSources4LearningIdx(k); % offset of main view into ptsView
                
                % if needed, construct dual cloud
                if dualOffsetDeg > 0
                    dualOffset = primaryOffset + floor(dualOffsetDeg/3); % offset of second view into ptsView; divide by 3 because views are every 3 deg in BB dataset.
                    if dualOffset > 120
                        dualOffset = dualOffset - 120;
                    end
                    temppts = [ptsView{primaryOffset}.pts ptsView{dualOffset}.pts]; % dual view
                    tempptscolors = [ptsView{primaryOffset}.colors ptsView{dualOffset}.colors]; % dual view
                else
                    temppts = ptsView{primaryOffset}.pts; % single view
                    tempptscolors = ptsView{primaryOffset}.colors; % single view
                end

                % Eliminate spurious pts by cropping outside of 1m
                tempptsidx = find(sqrt(sum(temppts.^2,1)) < 0.4);
                temppts = temppts(:,tempptsidx);
                tempptscolors = tempptscolors(:,tempptsidx);

                % Pts in <temppts> are concatenated into <pts> and
                % associated with a single view.
                thisPtsCam = zeros(numCamSources,size(temppts,2));
                thisPtsCam(k,:) = ones(1,size(temppts,2));
                
                % concatenate into pts, ptsCam
                pts = [pts temppts];
                ptsColors = [ptsColors tempptscolors];
                ptsCam = [ptsCam thisPtsCam];
                camPos = [camPos ptsView{primaryOffset}.camPos];
            end

            pViews = clsPtsHood(clsPts());
%             pViews.camPos = zeros(3,numCamSources); % set cam pos manually because we're setting them all at once
            pViews.camPos = camPos;
            pViews = pViews.addPts(pts,ptsCam,ptsColors);
            pViews = pViews.setCloud(clsPts(pViews)); % update pViews so that its cloud has the right camsources

        end
            
        

        
        
    end
    
    
    
end
