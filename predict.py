from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision import datasets, transforms, models
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image', type=str, help='path to the image to be identified')

    parser.add_argument('checkpoint', type=str, help='path to checkpoint of trained neural network.')

    parser.add_argument('--gpu', help='enables gpu mode', action='store_true')

    parser.add_argument('--topk', help='sets number of results to return from most to least likely', type=int,
                        default=5)

    parser.add_argument('--category_names', help='sets path to dictionary of names to display in output', type=str,
                        default=' ')

    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.features = checkpoint['features']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    im = Image.open(image)
    im = im.resize((256, 256))
    cropped_im = im.crop((0, 0, 249, 249))
    im_to_np = np.array(cropped_im)
    im_to_np = (im_to_np / 45) - np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
    im_to_np = im_to_np.transpose(2, 0, 1)
    return torch.from_numpy(im_to_np)


def predict(image, device, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if topk > len(model.class_to_idx):
        topk = len(model.class_to_idx)
    model.to(device)
    with torch.no_grad():
        model.eval()
        image = image.float().to(device)
        output = model.forward(image)
        prediction = torch.exp(output).data[0].topk(topk)
        return prediction


def main():
    args = get_args()
    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = load_checkpoint(args.checkpoint)

    image = process_image(args.image).unsqueeze_(0)

    probs, classes = predict(image, device, model, args.topk)
    probs, classes = probs.cpu().numpy(), classes.cpu().numpy()

    if args.category_names != ' ':
        indexes = []
        names = []

        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

        for category in classes:
            indexes.append(model.class_to_idx[str(category + 1)])
        for name in indexes:
            names.append(cat_to_name[str(name + 1)])
        for result in range(len(probs)):
            print('rank: {:<3}, name: {}, class: {:<3}, probability: {:.4f}\n'.format(result + 1,
                                                                                      names[result], classes[result],
                                                                                      probs[result]))

    else:
        for result in range(len(probs)):
            print('rank: {:<3}, class: {:<3}, probability: {:.4f}\n'.format(result + 1, classes[result], probs[result]))


if __name__ == "__main__":
    main()

    # python predict.py flowers/test/1/image_06752.jpg checkpoints/checkpoint.pth