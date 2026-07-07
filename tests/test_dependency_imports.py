def test_all_models_import():
    import prophet
    import neuralprophet
    import statsmodels.api as sm
    import catboost
    import pymc as pm
    import matplotlib
    import yaml
    # If we reach this line, all imports succeeded
    assert True