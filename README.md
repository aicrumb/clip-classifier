### why

i was so bored of writing the code myself like once a month so i spent way too long creating a little package to help me out

### installation

requires pytorch, tqdm, torchvision, i imagine einops and ftfy because it depends on clip?
if you just use `pip install git+https://github.com/openai/CLIP` it'll download every requirement there

`pip install git+https://github.com/aicrumb/anyclass` for installation of the package itself

### usage
```python
# simple usage
from anyclass import Classifier, load_img

classifier = Classifier(["dog", "cat", "mouse"])

image = load_img("dog.jpg")

prediction = classifier(image)
prediction = prediction.to_string()

print(prediction)
```
that will print:
```
dog    0.3565158545970917%
cat    0.3199838101863861%
mouse    0.32350030541419983%
```


```python

# advanced usage, add trainable projection head and train
from torch import optim
import torch
from anyclass import Classifier, SphericalDistanceLoss, load_img
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = Classifier(["among us", "fortnite", "frito-lay product"], 
						clip_model="ViT-B/32", # ViT-L/14@336px, ViT-L/14, ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16, RN50x64
						projection=True,
						projection_dim=128) # can be anything really
loss_fn = SphericalDistanceLoss()

urls = ["https://play-lh.googleusercontent.com/8ddL1kuoNUB5vUvgDVjYY3_6HwQcrg1K2fd_R8soD-e2QYj8fT9cfhfh3G0hnSruLKec",
		"https://static-assets-prod.epicgames.com/fortnite/static/webpack/8f9484f10eb14f85a189fb6117a57026.jpg",
		"https://mobileimages.lowes.com/product/converted/028400/028400097802.jpg"
		]
labels = [torch.tensor([1, 0, 0]),
		   torch.tensor([0, 1, 0]),
		   torch.tensor([0, 0, 1])
		   ]
dataloader = torch.utils.data.DataLoader(list(zip(urls,labels)), batch_size=1, shuffle=True)

optimizer = optim.Adam(classifier.projection_head.parameters(), 0.01)
for step, batch in enumerate(tqdm(dataloader)): 
	
	batch_urls = batch[0]
	batch_images = [load_img(url) for url in batch_urls]

	optimizer.zero_grad()
	loss = loss_fn(classifier(batch_images).outputs, torch.tensor(batch[1]).to(device))
	print(loss.item())

	classifier.recompute_classes()

print(classifier([load_img(url) for url in urls]).outputs)
```
that will print:
```
tensor([[0.3238, 0.3399, 0.3364],
        [0.3262, 0.3632, 0.3106],
        [0.3183, 0.3289, 0.3527]])
```
that's because it's only training one step, the confidence will go higher if you train for more (just put a for i in range(epochs): before the enumerate step)
