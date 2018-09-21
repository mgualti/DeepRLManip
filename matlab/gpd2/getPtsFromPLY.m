% Get points with associated cam sources and cam associations from a given
% PLY file. Points are voxelized into 0.003m cubes.
% input: plyFilename -> filename
%        numCams -> number of cam sources to use. Set this parameter to -1
%                   if no cam association is desired. In this case, all
%                   points are visible from all cams.
%        sizeCAD -> desired width of object. Set to -1 to leave width
%                   unchanged.
%        baseline -> stero baseline. set to zero if monocular
function pp = getPtsFromPLY(plyFilename, numCams, sizeCAD, baseline)

    % Create point set
    pp = clsPtsHood(clsPts());
    
    % randomly sample a bunch of cam positions w/ radius
%     camRadius = 0.4;
    camRadius = 0.6;
    cams = randn(3,numCams); % sample cam positions
    cams = camRadius * cams .* repmat(1./sqrt(sum(cams.^2,1)),3,1);

    % add cams
    if size(cams,2) > 0
        for i=1:size(cams,2)
            pp = pp.addCamPos(cams(:,i));
        end
    else
        pp = pp.addCamPos(zeros(3,1));        
    end
    
%     % read points without densifying
%     [vertex,face] = read_ply(plyFilename);
%     pts = vertex';

    % read points
    [vertex,face,vcolors] = read_ply(plyFilename);
    vertex = centerPts(vertex')';

    % rescale points according to <sizeCAD> parameter
    if sizeCAD ~= -1
        objWidth = getMinWidth(vertex'); % width of object in cad file
        desiredWidth = sizeCAD; % assume we want the object width to be this
        vertex = vertex * desiredWidth / objWidth; % scale pts to desiredWidth
    end
    
%     if size(vertex,1) > 40000
%     if size(vertex,1) > 60000
%         return;
%     end
    
    %  "densify" the cloud by adding pts to each face
    if size(vertex,1) < 60000
        [pts, colors] = densify(vertex,face,25,vcolors);
        pts = pts';
        colors = colors';
    else
        pts = vertex';
        colors = vcolors';
    end
    
%     % bail from ths function if too mainy points
%     size(pts,2)/1e6
    if size(pts,2) > 14e6 % any more points than this, and will risk running out of memory...
%     if size(pts,2) > 10e6 % any more points than this, and will risk running out of memory...
% %     if size(pts,2) > 2e6 % any more points than this, and will risk running out of memory...
% %     if size(pts,2) > 4e6 % any more points than this, and will risk running out of memory...
        return;
    end
    
    % voxelize using new matlab methods in order to preserve colors
    cloud = pointCloud(pts','Color',uint8(colors'));
    cloudvox = pcdownsample(cloud,'gridAverage',0.00225);
    pts = cloudvox.Location';
    colors = cloudvox.Color';
    
    % bail if there are too many points in the cloud
    if size(pts,2) > 40000
        return;
    end
    
%     pp = pp.addPts(pts,ones(pp.numCams(),size(pts,2)));
    pp = pp.addPts(pts,ones(pp.numCams(),size(pts,2)),colors);

% %     pp = pp.voxelize(0.003);
% %     pp = pp.voxelize(0.0015);
%     pp = pp.voxelize(0.00225);
    
    % Copy pts into clsPtsHood and calculate cam visibility. Prune
    % non-visible points.
    pp = pp.setCloud(clsPts(pp));
%     pp = pp.updateCamVisibility(0.003,0.01); % tube radius; dist along tube where we start counting

    % A negative <numCams> denotes that no cam assocation is to be
    % performed.
    if numCams > 0
        if baseline > 0
            pp = pp.makeCamsStereo(baseline);
        end
        pp = pp.updateCamVisibility(0.005,0.01); % tube radius; dist along tube where we start counting
        pp = pp.prune(find(sum(pp.ptsCamSource,1)>0));
        pp = pp.setCloud(clsPts(pp)); % update pp so that its cloud has the right camsources
    end
    
%     ppTemp = pp.prune(find(pp.ptsCamSource(1,:)));  % prune points in cloud that do not belong to source in <cam>
%     figure;
%     pp.plot();
%     figure;
%     ppTemp.plot();
        
end

% Center points by subtracting off mean
% input: pts -> 3xn set of pts
% output: centeredpts -> 3xn centered points
function centeredpts = centerPts(pts)
    centeredpts = pts - repmat(mean(pts,2),1,size(pts,2));
end


% Get min width of object. This is the distance between two parallel
% planes that are as close as possible while still containing the object
% input: pts -> 3xn set of pts
function minWidth = getMinWidth(pts)
    % randomly sample orientations
    for i=1:10000
        
        % get a random normal
        normal = randn(3,1);
        normal = normal ./ norm(normal);
        
        % project all points onto this normal
        ptsProjected = normal'*pts;
        
        % calculate width
        width(i) = max(ptsProjected) - min(ptsProjected);
        
    end
    minWidth = min(width);
end


% Make a surface more dense by randomly interpolating additional points on
% each face. This might be important when the PLY files we are reading
% have sparse vertices on one side of the cloud.
% input: desiredDensity -> desired num of pts per square cm (I suggest 25
% to achieve approx one pt every square 2mms).
% output: pts -> nx3 pt cloud
function [pts, colors] = densify(vertex,face,desiredDensity,colors)

    newpts = [];
    numpts = 0;
    for i=1:size(face,1)
        
        thisface = face(i,:);
        thisTriangle = vertex(thisface,:);
        triangleArea = getTriangleArea(thisTriangle(1,:)', thisTriangle(2,:)', thisTriangle(3,:)');
        numPtsPerFace = max(1,(triangleArea / 0.01^2) * desiredDensity);
        for j=1:numPtsPerFace
            coords = rand(1,3);
            coords = coords / sum(coords,2); % sample a pt on the 3d simplex
            newpt(j,:) = sum(thisTriangle .* [coords' coords' coords'],1);
            if nargin > 3
                [~,iClosestVertex] = max(coords);
                newcolor(j,:) = uint8(colors(thisface(iClosestVertex),:));
            end
        end
        newpts{i} = newpt;
        numpts = numpts + numPtsPerFace;
        if nargin > 3
            newcolors{i} = newcolor;
        end
    end
    pts = cell2mat(newpts');
    if nargin > 3
        newcolors = cell2mat(newcolors');
    else
        newcolors = [];
    end
    
    % add densified points to original points
    pts = [pts; vertex];
    colors = [newcolors; colors];
end

% Calculate area of the triangle defined by the three given points. Uses
% heuron's method.
% input: p1, p2, p3: nx1 vectors
function area = getTriangleArea(p1,p2,p3)
    a = norm(p1-p2);
    b = norm(p2-p3);
    c = norm(p1-p3);
    s = 0.5*(a+b+c);
    area = sqrt(s*(s-a)*(s-b)*(s-c));
end
