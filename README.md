# Character Extraction

## 1) Application description

### What it does right now
**1)** Character extraction from computer document based on OpenCV and classic OCR

- Extract lines

- then extract words from lines

- then extract characters from words

- store everything in sorted dictionnaries 1 doc dict <== <N line dicts <==M word dicts <== X chars)

  

**2)** Generate CSV dataset from [here](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/). (3rd one, named EnglishFnt.tgz)

- *train_datas.csv* contains training samples <u>sorted by characters</u>
- *labels.csv* contains labels of training samples
- *table.csv* contains conversion table from labels to real character


###  In progress
**3)** Train a CNN with computer font images

**4)** CNN inference to recognize every char

**5)** Generate document in txt format


### In the future
**6)** Replace classical OCR character separation by ML based method



## 2) Instruction
**Run** *"char_extract.py"* for char extraction from a document

**Extract** the archive named *"EnglishFnt.tgz"* in the *./dataset/* directory

**Run** *"generate_dataset.py"* for dataset generation



## 3) Problems

Not effective with all writing policies.
To many problems, but I wanted to try.

Regarding bullet (6) :

Way better to go for a ML method in order to extract characters in an image. 
A good method would be for example to move a frame around the image. 
The content of this frame would then feed a NN which would tell if there's a character or not.
If there's a character, the content of the frame would be stored to be analyzed by another NN to determine which character it is.

