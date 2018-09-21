classdef clsTrainCaffe
    
    properties
        
        % All object .MAT files get saved to a path off <categoryroot>
        categoryroot; 
        
        % *** <folderset> is a set of directories off <categoryroot>. Each
        % directory must contain a txt file called category_object_list.txt
        % that contains the name of each object for which to create images.
        % It is intended that each directory contains a set of objects that
        % belong to a different category.
        folderset; 
        
        DEBUG;
        useParfor; % indicates whether to use a parfor (parallel) loops when appropriate or just regular serial for loops.
    end
    
    methods
        
        function tc = clsTrainCaffe(categoryroot,folderset,useParfor,DEBUG)
            tc.categoryroot = categoryroot;
            tc.folderset = folderset;
            tc.useParfor = useParfor;
            tc.DEBUG = DEBUG
        end
        
        % input: maxNumPerObject -> max number of exemplars loaded per
        % object. You want to limit this so that the total number of
        % exemplars (numperobject x num objects) <= 250k.
        function l = loadClsLearning(tc,l,maxNumPerObject)

            % Iterate through categories
            for i=1:size(tc.folderset,2)

                folder = [tc.categoryroot tc.folderset{i} '/'];
                objectList = tc.getCategoryObjects(folder);

                % Iterate through all objects within a category
                for j=1:size(objectList,1)
                    objUID = objectList{j};

                    filenameImages = [folder objUID '_Images.mat'];
                    if exist(filenameImages, 'file')

                        data = load(filenameImages);
                        handsThisObject = data.handsThisObject;

                        ltemp = clsLearning();
                        for k=1:size(handsThisObject,2)
                            ltemp = ltemp.importFromHands(handsThisObject{k});
                        end
                        
                        % Cap number of exemplars from each object at
                        % <maxNumPerObject>
                        if ltemp.num > maxNumPerObject
                            ltemp = ltemp.subSample(maxNumPerObject);
                        end
                        l = l.importFromClsLearning(ltemp);
                        
                    end
                end
            end
        end
        
        
        % Get a random object. 
        function [objUID, folder] = getRandomObject(tc)
            folder = tc.folderset{randi(size(tc.folderset,2))}; % random folder
            folder = [tc.categoryroot folder '/'];
            objectList = tc.getCategoryObjects(folder);
            objUID = objectList{randi(size(objectList,2))}; % random object
        end
        
        % Get contents of _allclouds.mat file for a random object.
        function [ptsView, mesh] = getAllClouds4RandomObject(tc)
            [objUID, folder] = tc.getRandomObject();            
            filenameCloud = [folder objUID '_allclouds.mat'];
            data = load(filenameCloud);
            ptsView = data.ptsView;
            mesh = clsPts(data.pGroundTruth);
        end
        
        % Get a random view of an object in one of the categories in
        % <folderset>. Assumes a stereo view. Can also request the
        % corresponding mesh as a second parameter. 
        % input: singleDual -> 0 means single view; 1 means dual view. In dual view
        % case, paired dual cams are assumed to be sequential. EG: 1 2 go into a
        % single view; 3 4 go into the next view.        
        function [p, mesh] = getRandomCloud(tc,singleDual)
            
            % get random object
            [objUID, folder] = tc.getRandomObject();
            
            % get clouds
            filenameCloud = [folder objUID '_clouds4Learning.mat'];
            data = load(filenameCloud);
            ptsViews = clsPts(data.pViews);
            mesh = clsPts(data.pGroundTruth);
            
            % get random pair of clouds
            if singleDual
                camSet = 1:2:ptsViews.numCams();
                i = randi(floor(ptsViews.numCams()/2));
                p = ptsViews.ablate([camSet(i) camSet(i)+1]);
            else
                camSet = 1:ptsViews.numCams();
                i = randi(floor(ptsViews.numCams()));
                p = ptsViews.ablate(camSet(i));
            end
        end
        
        
        % Extract clsLearning structure from objects in the categories contained in
        % <categoryroot>.
        % input: singleDual -> 0 means single view; 1 means dual view. In dual view
        % case, paired dual cams are assumed to be sequential. EG: 1 2 go into a
        % single view; 3 4 go into the next view.
        function calcLabeledGrasps(tc, handparams, numCams, numSamples, singleDual, curvAxisProb)

            maxPerCategory = 100;

            % Iterate through categories
            for i=1:size(tc.folderset,2)

                folder = [tc.categoryroot tc.folderset{i} '/'];

                objectList = tc.getCategoryObjects(folder);

                % Iterate through all objects within a category
                for j=1:size(objectList,1)
                    objUID = objectList{j};

                    filenameCloud = [folder objUID '_clouds4Learning.mat'];
                    filenameLabeledCandidates = [folder objUID '_LabeledGraspCandidates.mat'];
                    if exist(filenameCloud, 'file') && ~exist(filenameLabeledCandidates, 'file')

                        % load point clouds for this object
                        data = load(filenameCloud);
                        ptsViews = clsPts(data.pViews);
                        ptsMesh = clsPts(data.pGroundTruth);

                        % If there aren't enough points in this cloud, go on to the
                        % next one
                        if ptsMesh.num() < 100
                            continue;
                        end

                        % iterate through all views for this object
                        if singleDual
                            camSet = 1:2:ptsViews.numCams(); % dual view data (skip every other cam)
                        else
                            camSet = 1:ptsViews.numCams(); % single view data
                        end
                        cams2Select = randperm(size(camSet,2),numCams);
                        camSet = camSet(:,cams2Select);

                        handsThisObject = cell(size(camSet));
                        if tc.useParfor
                            parfor k=1:size(camSet,2)

                                % calculate cam sources for this source
                                if singleDual camPair = [camSet(k) camSet(k)+1]; else camPair = camSet(k); end

                                % Init object/cloud for <thisView>
                                thisView = ptsViews.ablate(camPair); % ablate cloud to get <thisView>

                                % Get labeled grasps for <thisView>
                                hands = clsPtsHands(thisView,handparams,objUID,camPair,tc.DEBUG);
                                hands = hands.subSample(numSamples);
                                handsThisObject{k} = hands.calcBalancedCandidatesAndLabels(thisView,ptsMesh,curvAxisProb)
                            end
                        else
                            for k=1:size(camSet,2)

                                % calculate cam sources for this source
                                if singleDual camPair = [camSet(k) camSet(k)+1]; else camPair = camSet(k); end

                                % Init object/cloud for <thisView>
                                thisView = ptsViews.ablate(camPair); % ablate cloud to get <thisView>

                                % Get labeled grasps for <thisView>
                                hands = clsPtsHands(thisView,handparams,objUID,camPair,tc.DEBUG);
                                hands = hands.subSample(numSamples);
                                handsThisObject{k} = hands.calcBalancedCandidatesAndLabels(thisView,ptsMesh,curvAxisProb);
                            end                            
                        end
                        
                        save(filenameLabeledCandidates,'handsThisObject','-v7.3');
                    end
                end
            end
        end

        
        function objectList = getCategoryObjects(tc,folder)
            % Open category list inside each category directory
            fid = fopen([folder 'category_object_list.txt'],'r');
            tempdata = textscan(fid,'%s','Delimiter','\n');
            objectList = tempdata{1};
            fclose(fid);
        end


        % input: 
        %        flip -> binary flag indicating whether to add each
        %        candidate as a single image (0), or to add each candidate
        %        twice -- once w/ the regular image and once w/ the flipped
        %        image. Typically, you want to flip duing training, but not
        %        during deployment.
        function calcOcclusionImages(tc,imageparams,flip)

            if nargin < 3
                flip = 0;
            end
            
            % Iterate through categories
            for i=1:size(tc.folderset,2)

                folder = [tc.categoryroot tc.folderset{i} '/'];

                % Open category list inside each category directory
                fid = fopen([folder 'category_object_list.txt'],'r');
                tempdata = textscan(fid,'%s','Delimiter','\n');
                objectList = tempdata{1};
                fclose(fid);

                % Iterate through all objects within a category
                for j=1:size(objectList,1)
                    objUID = objectList{j};

                    filenameCloud = [folder objUID '_clouds4Learning.mat'];
                    filenameLabeledCandidates = [folder objUID '_LabeledGraspCandidates.mat'];
                    filenameImages = [folder objUID '_Images.mat'];
                    if exist(filenameCloud, 'file') && exist(filenameLabeledCandidates, 'file') && ~exist(filenameImages, 'file')

                         % load point clouds for this object
                        data = load(filenameCloud);
                        ptsViews = clsPts(data.pViews);
                        ptsMesh = clsPts(data.pGroundTruth);

                        data = load(filenameLabeledCandidates);
                        handsThisObject = data.handsThisObject;
                        
                        if tc.useParfor
                            parfor k=1:size(handsThisObject,2)
                                if handsThisObject{k}.num() > 0
                                    handsThisObject{k}.DEBUG = tc.DEBUG;
                                    handsThisObject{k} = handsThisObject{k}.setImageParams(imageparams);
                                    thisView = ptsViews.ablate(handsThisObject{k}.camSet);
%                                   handsThisObject{k} = handsThisObject{k}.calculateOcclusion(thisView);
                                    handsThisObject{k} = handsThisObject{k}.calculateImagesClear(thisView);
                                end
                            end
                        else
                            for k=1:size(handsThisObject,2)
                                if handsThisObject{k}.num() > 0
                                    handsThisObject{k}.DEBUG = tc.DEBUG;
                                    handsThisObject{k} = handsThisObject{k}.setImageParams(imageparams);
                                    thisView = ptsViews.ablate(handsThisObject{k}.camSet);
                                    %handsThisObject{k} = handsThisObject{k}.calculateOcclusion(thisView);
                                    handsThisObject{k} = handsThisObject{k}.calculateImagesClear(thisView);
                                end
                            end                            
                        end
                        
                        save(filenameImages,'handsThisObject','-v7.3');

                    end
                end
            end
        end

        
    end
    
end
