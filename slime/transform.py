'''
Implements classes for perturbing instances.
'''
from abc import ABC, abstractmethod
import numpy as np


class Transformer(ABC):
    '''
    Transformer is an abstract base class implementing methods for applying any
    kind of transformation to data as part of LIME.
    '''
    @abstractmethod
    def transform(self, instance):
        '''
        Transform applies a transformation to an instance and returns the
        transformed data point. For example, when working with images of
        mammals, this transformation could be removing their ears.
        '''

    @property
    def name(self):
        '''
        Returns the name of the class. This is used to identify the
        transformation / feature in LIME.
        '''
        return self.__class__.__name__


class SegmentTransformer(Transformer):
    '''
    An image transformer that masks out part of the image.
    '''
    def __init__(self, mask):
        self._mask = mask

    def transform(self, instance):
        return instance * self._mask


class BinaryPerturber(ABC):
    '''
    Generates perturb instances by turning features on and off (eg. super-pixels)
    in an image.
    '''
    @abstractmethod
    def perturb(self, enabled_features):
        '''
        Takes a list of booleans specifying which features to enable or disable.
        If a boolean value is False, the corresponding transformation is applied
        (eg. the feature is removed).
        '''

    @property
    @abstractmethod
    def feature_count(self):
        '''
        The number of features that can be perturbed.
        '''

    def base(self):
        '''
        The base image, ie. where all the features are removed.
        '''
        return self.perturb([False for _ in range(self.feature_count)])


class TransformerPerturber(BinaryPerturber):
    '''
    Applies transformations to an instance in order to perturb it.
    '''
    def __init__(self, instance, transformers):
        self._instance = instance
        self._transformers = transformers

    def perturb(self, enabled_features):
        instance_copy = np.copy(self._instance)
        for enabled, transformer in zip(enabled_features, self._transformers):
            if not enabled:
                instance_copy = transformer.transform(instance_copy)
        return instance_copy

    @property
    def feature_count(self):
        return len(self._transformers)
