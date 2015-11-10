SimpleImageClassifier

Brian Davis
2015

Our first project for CS601R. Uses lots of OpenCV.

This provides several ways of going about the pipeline:
* Feature point detection: SIFT (OpenCV), dense
* Feature point description: SIFT (OpenCV and personal implementation), HOG (only does dense feature points)
* Pooling: Locality constrained proportional encoding, Spatial pyramid (only allows for a second level of 2x2 bins)
* Classifier: SVM (OpenCV and LIBSVM, though only OpenCV’s implementation was used in the tests)

USAGE:

`./SimpleClassifier train_codebook(_size=<int>) <feature point detection option> <feature point description option> <path:image directory> <path: out codebook file>`

`./SimpleClassifier train_libsvm(_eps=<epsilon float>_C=<int>_AllVs=<int>) <feature point detection option> <feature point description option> <pooling option> <path:image directory> <path:codebook file> <path:out svm file>`

`./SimpleClassifier train_cvsvm(_AllVs=<int>) <feature point detection option> <feature point description option> <pooling option> <path:image directory> <path:codebook file> <path:out svm file>`

`./SimpleClassifier test_(libsvm or cvsvm)(_AllVs=<int>)(_prcurves) <feature point detection option> <feature point description option> <pooling option> <path:image directory> <path:codebook file> <path:out svm file>`

Where:

`<feature point detection option>` can be:
* `SIFT`			Extracts feature points using OpenCV’s SIFT implementation
* `dense(_stride=<int>)`	Extracts feature points densely (defaults to stride of 5)

`<feature point description option>` can be:
* `SIFT`				Describes feature points using OpenCV’s SIFT implementation
* `customSIFT`			Describes feature points using my implementation of SIFT
* `HOG(_stride=<int>)(_size=<int>)(_thresh=<int>)`	Describes feature points densely (ignores detection parameter) using my implementation of HOG.

`<pooling option>` can be:
* `bovw(_LLC=<int>)`		Encodes feature points in a Bag-of-Visual-Words representation, using 
locality constrained proportional encoding (not actually LLC encoding). The integer you pass is the number of nearest neighbors used when encoding.
* `sppy(_LLC=<int>)`		Encodes feature points in a Bag-of-Visual-Words representation, using a spatial pyramid with two layers, the first being 1x1 and the second 2x2. It uses the same encoding as the above option.

The parentheses are showing optional input. Any of the paths can be entered as `defaut`, and the program automatically uses a consistent file name (in the ./save/ directory) for the parameters. For the `AllVs=<int>` parameter, if the integer is positive, a one-vs-all classifier is trained (for the entered class number). If `-1` is entered, a multiclass classifier is trained. If `-2` is entered, ten one-vs-all classifiers are trained individually, and tested together as one classifier (so the P-R curve can be generated). The OpenCV implementation of the SVM is always trained with the `train_auto` function, which optimises its parameters. When a one-vs-all SVM is trained, the positive examples are given a weight of 9.
