'''
Implements classes for explaining the predictions of a model on an instance.
'''
from collections import Counter

import numpy as np
from lime.lime_base import LimeBase
from tqdm.auto import tqdm
from sklearn.utils import check_random_state


class Explanation:
    """
    Explanation stores the local interpretable models used in explaining a
    prediction on a data instance.
    """

    def __init__(self, perturber):
        self.perturber = perturber
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}
        self.top_labels = None

    def explain(self, label, num_features=5):
        """
        Returns a perturbed copy of the instance using the local model for a
        label to only show the most important features that contribute to an
        explanation.
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')

        positive_features = [x[0] for x in self.local_exp[label] if x[1] >= 0]

        features_to_keep = positive_features[:num_features]

        enabled_features = [False for _ in range(self.perturber.feature_count)]
        for feature in features_to_keep:
            enabled_features[feature] = True
        return self.perturber.perturb(enabled_features)


class Explainer:
    """
    Explainer generates explanations of a model's prediction by applying
    transformations to the image and measuring the change in the prediction.
    """
    def __init__(self, metric, kernel_width=0.25, random_state=None):
        self._metric = metric

        def kernel(distance):
            return np.sqrt(np.exp(-(distance ** 2) / kernel_width ** 2))
        self._base = LimeBase(kernel)
        self._random_state = check_random_state(random_state)

    def explain_instance(self, perturber, predict, top_labels=5, num_samples=10, num_features=100):
        """
        Generates explanations for an instance.
        """
        baseline = predict(np.array([perturber.base()]))
        data, predictions = self._feature_values(
            perturber, predict, num_samples=num_samples)
        distances = np.array([self._metric(baseline, prediction)
                             for prediction in predictions]).ravel()

        ret_exp = Explanation(perturber)
        if top_labels:
            top_predicted_labels = [
                np.argmax(prediction) for prediction in predictions]
            top_label_counter = Counter(
                top_predicted_labels).most_common(top_labels)
            top = [count[0] for count in top_label_counter]
            ret_exp.top_labels = top
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self._base.explain_instance_with_data(
                data, predictions, distances, label, num_features,
                model_regressor=None,
                feature_selection='auto')
        return ret_exp

    def _feature_values(self, perturber, predict, num_samples=10, batch_size=10):
        feature_combinations = self._random_state.randint(
            2, size=(num_samples, perturber.feature_count))
        progress_bar = tqdm(feature_combinations)
        predictions = []
        batch = []
        for combination in progress_bar:
            perturbed = perturber.perturb(combination)
            batch.append(perturbed)
            if len(batch) == batch_size:
                batch_predictions = predict(np.array(batch))
                predictions.extend(batch_predictions)
                batch = []
        return feature_combinations, np.array(predictions)
