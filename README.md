# MUSTARD

This package is a ADI sequence processing tool that aim to distangle part of the sequence thats is startics (quasi-static speakels), decorelated signal (fast variating outliers and uncorrected by AO and noise) and the signal that is rotating (the signal of interest : Disk and exoplanets)

## Demo

<img src="./example-data/demo.gif" alt="demo" width="700" text-align="center"/>

## Install package

Clone the project

```bash
  git clone https://github.com/Sand-jrd/mustard
```

Go to the project directory

```bash
  cd mustard
```

Install dependencies

```bash
  py setup.py install
```

## Usage/Exemple

Follow instruction in the demo to test the algorithm with your own datasets.
[Pyhton file](demo.py)

Produce the cool gif presented [above](#Screenshots) with PSD-70 dataset (03/2019, program 1100.C-0481(L), K2-band)
[How to use MUSTARD](doc/demo.html)
[Source jupyter Notebook](demo.ipynb)

Use [documentation](#Documentation) to known more about configration.

## Related

Also check out other package for Exoplanet/disk direct imaging

[GreeDS](https://github.com/Sand-jrd/GreeDS)
I-PCA (Iterative Principal Component Analysis). 
Refactored implementation of the original code from [Pairet et al](https://arxiv.org/pdf/2008.05170.pdf)

[VIP - Vortex Image Processing package](https://github.com/vortex-exoplanet/VIP)
Tools for high-contrast imaging of exoplanets and circumstellar disks.

## Documentation

Auto-generated user guide
[Class description](doc/index.html)
[How to use MUSTARD](doc/demo.html)

Doc about the maths behind the algorithm
[Slides](https://docs.google.com/presentation/d/1aPjWJUztfjROtt8BPi8uh6X-vBD5dc81wQ1MhMGGOas/edit) 

Doc about package structure
[Documentation](doc/UMLdocs.png)


## Feedback/Support

You can contact me by email : sjuillard@uliege.be
