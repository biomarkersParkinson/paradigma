import pickle

def read_PPG_quality_classifier(classifier_path: str):
    """
    Read the PPG quality classifier from a file.

    Parameters
    ----------
    classifier_path : str
        The path to the classifier file.

    Returns
    -------
    dict
        The classifier dictionary.
    """
    with open(classifier_path, 'rb') as f:
        clf = pickle.load(f)
    return clf