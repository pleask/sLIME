# sLIME

sLIME (semantic LIME) provides a generic interface to the [Local Interpretable Model-Agnostic Explanations](https://github.com/marcotcr/lime) package, allowing for construction of arbitrary transformers that remove features / concepts from data instances. For example, with images, the original package only implements superpixels as features; with sLIME it is possible to consider human-level concepts, such as eyes or ears, in the local models.

The dissertation associated with this project is [here](https://drive.google.com/file/d/1bzDltWzkp9EWEewCuEkOVhQpUysL8TGz/view?usp=sharing) - I probably won't be writing this into a shorter paper.

## Tutorials
The following tutorials are / will be available in the repository.
- [Superpixel segmentation:](https://github.com/pleask/sLIME/blob/main/tutorials/superpixels.ipynb) Recreates the superpixel segmentation tutorial from the [LIME repo](https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html) as a basic introduction to transformers and perturbers.
- [Generated datasets](https://github.com/pleask/sLIME/blob/main/tutorials/generated_datasets.ipynb): How to explain classifications on a generated dataset where the user can create arbitrary in-distribution images through feature perturbation.
- [Training transformers from generated datasets](https://github.com/pleask/sLIME/blob/main/tutorials/generated_transformers.ipynb): How to train transformers on a dataset where the user has access to examples of images with and without features (eg. the same background with and without a foreground object).
- Training transformers from feature detectors (not yet available): How to train transformers on a dataset where the user only has accessed to examples that are labelled as to whether they contain a feature or not.
