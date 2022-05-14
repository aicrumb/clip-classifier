
# advanced usage, add trainable projection head
from torch import optim
import torch
from anyclass import CLIPClassifier, SphericalDistanceLoss, load_img
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = Classifier(["among us", "fortnite", "frito-lay product"], 
						clip_model="ViT-B/32", # ViT-L/14@336px, ViT-L/14, ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16, RN50x64
						projection=True,
						projection_dim=128)
loss_fn = SphericalDistanceLoss()

urls = ["https://play-lh.googleusercontent.com/8ddL1kuoNUB5vUvgDVjYY3_6HwQcrg1K2fd_R8soD-e2QYj8fT9cfhfh3G0hnSruLKec",
		"https://static-assets-prod.epicgames.com/fortnite/static/webpack/8f9484f10eb14f85a189fb6117a57026.jpg",
		"https://mobileimages.lowes.com/product/converted/028400/028400097802.jpg"
		]
targets = [torch.tensor([1, 0, 0]),
		   torch.tensor([0, 1, 0]),
		   torch.tensor([0, 0, 1])
		   ]
dataloader = torch.utils.data.DataLoader(list(zip(urls,targets)), batch_size=1, shuffle=True)

optimizer = optim.Adam(classifier.projection_head.parameters(), 0.01)
for step, batch in enumerate(tqdm(dataloader)): 
	
	batch_urls = batch[0]
	batch_images = [load_img(url) for url in batch_urls]

	optimizer.zero_grad()
	loss = loss_fn(classifier(batch_images).outputs, torch.tensor(batch[1]).to(device))
	print(loss.item())

	classifier.recompute_classes()

print(classifier([load_img(url) for url in urls]).outputs)