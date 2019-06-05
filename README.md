# IsRealSmile

[![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/MartinKondor/IsRealSmile/)
[![version](https://img.shields.io/badge/version-2019.07-brightgreen.svg)](https://github.com/MartinKondor/IsRealSmile)
[![GitHub Issues](https://img.shields.io/github/issues/MartinKondor/IsRealSmile.svg)](https://github.com/MartinKondor/IsRealSmile/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Real or fake smile recognizer.

## Getting Started

### Prerequisites

* Python 3.6+
* Anaconda 4+ (optional)
* Python modules from the `requirements.txt`

### Deployment

Dowload and install the dependencies with the command:

```
$ python -m pip install -r requirements.txt
```

Then train the model, or download one of our releases and place the contents of the `trained` directory to this project's `trained` directory.

To train the model you should run:

```
$ python train.py
```

Model files will be saved in the `trained` directory and will overwrite the existing files there.

To use the trained model run

```
$ python isrealsmile.py -h
```

and see what commands you can run.

### Example

```
$ python isrealsmile.py --show %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Arnold_Schwarzenegger\Arnold_Schwarzenegger_0006.jpg
$ python isrealsmile.py --show %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Yoko_Ono\Yoko_Ono_0003.jpg
```

## Contributing

This project is open for any kind of contribution from anyone.

### Steps

1. Fork this repository
2. Create a new branch (optional)
3. Clone it
4. Make your changes
5. Upload them
6. Make a pull request here
7. Profit.

## Citation for data

```
Arigbabu, Olasimbo Ayodeji, et al. "Smile detection using hybrid face representation."
Journal of Ambient Intelligence and Humanized Computing (2016): 1-12.

C. Sanderson, B.C. Lovell. Multi-Region Probabilistic Histograms for Robust and
Scalable Identity Inference.
ICB 2009, LNCS 5558, pp. 199-208, 2009

Huang GB, Mattar M, Berg T, Learned-Miller E (2007) Labeled faces in the wild:
a database for studying face recognition in unconstrained environments.
University of Massachusetts, Amherst, Technical Report
```

## Authors

* **[Martin Kondor](https://github.com/MartinKondor)**

# License

See the [LICENSE](LICENSE) file for details.
