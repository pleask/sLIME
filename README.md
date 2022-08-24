# sLIME

sLIME (semantic LIME) provides a generic interface to the [Local Interpretable Model-Agnostic Explanations](https://github.com/marcotcr/lime) package, allowing for construction of arbitrary transformers that remove features / concepts from data instances. For example, with images, the original package only implements superpixels as features; with sLIME it is possible to consider human-level concepts, such as eyes or ears, in the local models.

Whilst this implementation is available, research into how to construct these concepts and transformers is ongoing as part of [Patrick Leask's](mailto:patrickaaleask@gmail.com) master's dissertation research at the University of Liverpool. This research will be published at the end of September.

## Tutorials
The following tutorials are / will be available in the repository.
- [Superpixel segmentation:](https://github.com/pleask/sLIME/blob/main/tutorials/superpixels.ipynb) Recreates the superpixel segmentation tutorial from the [LIME repo](https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html) as a basic introduction to transformers and perturbers.
- Generated datasets (not yet available): How to explain classifications on a generated dataset where the user can create arbitrary in-distribution images through feature perturbation.
- Training transformers from generated datasets (not yet available): How to train transformers on a dataset where the user has access to examples of images with and without features (eg. the same background with and without a foreground object).
- Training transformers from feature detectors (not yet available): How to train transformers on a dataset where the user only has accessed to examples that are labelled as to whether they contain a feature or not.