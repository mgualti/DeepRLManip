IMPORTANT: when you start matlab, you should use the command below. 
If I don't do this, java sucks up all my memory and eventually 
crashes matlab.

matlab -nodesktop

I created a couple of examples that should enable you to train a grasp 
detector for a single object (works for multiple objects too). When 
caffe is installed properly, you should be able to run:

1. createImages.m -> create images to train on a new object

2. trainCaffe.m -> train caffe model (you must run createImages first)

3. exampleDeployGPD.m -> test caffe model on a novel point cloud 
(you must run trainCaffe and train network first)

4. exampleEvaluateAccuracy.m -> Do grasp detection on a randomly 
extracted cloud and calculate detection accuracy against ground 
truth. This value should roughly match what was obtained against the 
test set during training.

You need to have the following directory structure. Create it if it doesn't exist.

gpd2 -> the code
gpd2/data/CAFFEfiles -> where we do learning in caffe
gpd2/data/MATfiles -> where we store intermediate .mat files

A few other examples you might look at:

exampleFrenet.m -> illustrate calculating surface normals and frenet frames

exampleGraspCandidates.m -> illustrate calculating grasp candidates 
and getting ground truth labels using a mesh

exampleEvaluateAccuracy.m -> Do grasp detection on a randomly 
extracted cloud and calculate detection accuracy against ground 
truth. This value should roughly match what was obtained against the 
test set during training.


Organization:

handsList -> One data structure that I don't think is explained in 
the code well is the handsList. This is a 2xn matrix that indexes 
into a corresponding clsPtsHands structure. The first col is the 
index of the clsHandSet. The second col is the index of the 
orientation within that handset. If you want to get the handsList 
for a given clsPtsHands structure, just call 
clsPtsHands.getAllHands(). Any reference to a "list" of hands is 
typically in this format.


Q/A:

> 1. What exactly does M represent (clsPtsFrenet/calcFrame)? I can't 
find this in any paper that talks about local reference frames. They 
always use a covariance matrix based on a point neighborhood. Is 
this some simplification of a covariance matrix?

This creates the local reference frame that is used in candidate 
generation. We've already generated surface normals using the 
standard methods. This just smooths those normals a bit. M does that 
smoothing. I don't know what you would call M. It's just a matrix 
where the major eigen vector of M is the smoothed normal. The 
minimum eigenvector is the smoothed axis. The remaining one is the 
"binormal."

> 2. If the minimum eigenvalue of M corresponds to the minor 
curvature axis, should not the maximum eigenvalue correspond to the 
major curvature axis? In this case, what's called "normal" in the 
code is not actually the surface normal, but, e.g., in the case of 
an upright cylinder, this would be the minor object axis (the 
lateral axis of the cylinder)? 

No, the largest eigenvector should be the surface normal. I just 
added exampleFrenet.m which you can run to see a visualization.

> 3. Is the ordering of the axes in F arbitrary?

Definitely not arbitrary. F is a 3x3 matrix where the first col is the normal; the third is the axis; the second is the vector orthogonal to both.
