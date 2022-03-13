# Phase splitter
Welcome to Phase splitter, a non-negative matrix factorization to separate the different phases in an in situ "PDF" data.

## Description

The algorithms were used as part of the bachelor thesis "Phase-Splitter: An Automated Tool for phase identification and
characterization of in situ Pair Distribution Function data." 

Example of in situ data with multiple phase shifts to the left and the resultant NMF components from Phase splitter on the right. 
![12](https://user-images.githubusercontent.com/65853425/157948154-62826afd-76ca-4926-bec8-313124d36bc9.png)

The GitHub contains a test dataset called "insitu_sim" which is simulated PDFs of an Cu reduction synthesis from Cu(OH)_2 precursor.
## Getting Started (with colab)
Using Phase splitter on your PDFs is straightforward and does not require anything installed or downloaded to your computer. Follow the instructions in the Colab notebook and try to play around.

[Phase-splitter (Colab)](https://colab.research.google.com/drive/1ypGob83K4NawqdE_1lfQORmObUiLb89c?usp=sharing)

### Installing on own PC
Download the folder "phase-splitter.zip". 

Set 'root_path_results' in funcs.py to the location of 'phase-splitter/Results' to the path on your computer

### Dependencies
The lite version is made to work on python 3.7 as a notebook with the following libraries:

For plots:

matplotlib, seaborn, IPython.display, celluloid

For data:

numpy, pandas

For statistical analysis:

sklearn, scipy


### Executing program

Open the notebook "phase-splitter.ipynb" in your favorite python interpreter and follow the instructions.

## Authors

Contributors' names and contact info

Joakim Lajer (gpw395@alumni.ku.dk)

## Version History

* 0.1
    * Initial Release (lite version)

## License

This project is licensed under the GNU General Public License v3.0, January 2004 - see the [LICENSE](https://github.com/Kabelkim/phase-splitter/blob/main/LICENSE) file for details.

## Acknowledgments

Inspiration, mental support, guidance, etc.
* [Emil Skaaning](https://github.com/EmilSkaaning)
* [Andy S. Anker](https://github.com/AndyNano)
