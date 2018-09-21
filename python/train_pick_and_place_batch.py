#!/usr/bin/env python
'''Trains the robot to locate stable placements.'''

# python
import os
# scipy
# self
import train_pick_and_place_bottles
import train_pick_and_place_blocks
import train_pick_and_place_mugs

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  task = "mugs"
  resultsPrefix = "2018-04-05"
  nRealizations = 1
  startRealization = 0

  # RUN TEST =======================================================================================

  for i in xrange(startRealization, nRealizations):
    if task == "bottles":
      train_pick_and_place_bottles.main()
    elif task == "blocks":
      train_pick_and_place_blocks.main()
    elif task == "mugs":
      train_pick_and_place_mugs.main()
    else:
      raise Exception("Unknown task {}.".format(task))
    os.system("mv results.mat {}-{}-results.mat".format(resultsPrefix, i))
    os.system("mkdir -p {}-{}-caffe/grasp".format(resultsPrefix, i))
    os.system("mkdir -p {}-{}-caffe/place".format(resultsPrefix, i))
    os.system("mv caffe/weights-grasp-{}/*.caffemodel ./{}-{}-caffe/grasp/".format(task, resultsPrefix, i))
    os.system("mv caffe/weights-place-{}/*.caffemodel ./{}-{}-caffe/place/".format(task, resultsPrefix, i))

if __name__ == "__main__":
  main()
  exit()
