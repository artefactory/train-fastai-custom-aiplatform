from fastai.learner import load_learner

learner = load_learner("test_aip.pth")
print(learner.predict("hello guys"))
