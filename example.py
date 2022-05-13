# simple usage
from anyclass import Classifier, load_img

classifier = Classifier(["dog", "cat", "mouse"])

image = load_img("dog.jpg")

prediction = classifier(image)
prediction = prediction.to_string()

print(prediction)