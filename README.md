# Image Captioning Preprocessing Project

## Description
This project focuses on preprocessing data for an Image Captioning task.  
The completed steps include:

1. **Image Preprocessing**:
   - Reading images from the `data/Images` folder.
   - Resizing all images.
   - Normalizing pixel values to the range [0, 1].

2. **Captions Preprocessing**:
   - Converting all text to lowercase.
   - Removing unwanted characters.
   - Removing duplicate captions.
   - Tokenizing sentences into words.
   - Building a filtered vocabulary.

## Project Structure

## Installation
1. Create a virtual environment:
```bash
Activate the virtual environment:

python -m venv venv

venv\Scripts\activate

source venv/bin/activate


Usage
To run the project:

Install required packages:

python main.py
