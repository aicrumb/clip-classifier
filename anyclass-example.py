# simple usage
from anyclass import CLIPClassifier, load_img

classifier = CLIPClassifier(["dog", "cat", "mouse"])

image = load_img("dog.jpg")

prediction = classifier(image)
prediction = prediction.to_string()

print(prediction)