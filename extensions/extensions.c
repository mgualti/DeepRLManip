#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Sets image to value at coordinates, using average value whenever multiple same coordinates exist. 
 * - Input image: nx by ny image, where ny is number of pixels in the consecutive dimension.
 *   Assumed to be initialized with zeros.
 * - Input nx: Number of pixels in dimension 0.
 * - Input ny: Number of pixels in dimension 1.
 * - Input coordinates: List of paris [(ix, iy)_0, ..., (ix, iy)_nValues] indexing into image.
 * - Input values: Values to set the image to a the respective coordinates.
 * - Input nValues: The number of elements in the array values.
 */
void SetImageToAverageValues(float* image, int nx, int ny, float* coordinates, float* values, int nValues)
{
  int* counts = malloc(nx*ny*sizeof(int));
  memset(counts, 0, nx*ny*sizeof(int));
  for (int i = 0; i < nValues; i++)
  {
    int idx = (int) coordinates[2*i+0];
    int jdx = (int) coordinates[2*i+1];
    int kdx = idx*ny+jdx;
    counts[kdx]++;
    image[kdx] = values[i];
  }
  for (int i = 0; i < nx*ny; i++)
    if (counts[i] > 0)
      image[i] /= (float) counts[i];
  free(counts);
}

/* Sets image to value at coordinates, using max value whenever multiple same coordinates exist. 
 * - Input image: nx by ny image, where ny is number of pixels in the consecutive dimension.
 *   Assumed to be initialized with zeros.
 * - Input nx: Number of pixels in dimension 0.
 * - Input ny: Number of pixels in dimension 1.
 * - Input coordinates: List of paris [(ix, iy)_0, ..., (ix, iy)_nValues] indexing into image.
 * - Input values: Values to set the image to a the respective coordinates.
 * - Input nValues: The number of elements in the array values.
 */
void SetImageToMaxValues(float* image, int nx, int ny, float* coordinates, float* values, int nValues)
{
  for (int i = 0; i < nValues; i++)
  {
    int idx = (int) coordinates[2*i+0];
    int jdx = (int) coordinates[2*i+1];
    int kdx = idx*ny+jdx;
    image[kdx] = fmax(image[kdx], values[i]);
  }
}
