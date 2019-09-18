# SmileRecognizer

[![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/MartinKondor/SmileRecognizer/)
[![version](https://img.shields.io/badge/version-2019.07-brightgreen.svg)](https://github.com/MartinKondor/SmileRecognizer)
[![GitHub Issues](https://img.shields.io/github/issues/MartinKondor/SmileRecognizer.svg)](https://github.com/MartinKondor/SmileRecognizer/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Smile recognizer AI.

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
$ python issmile.py -h
```

and see what commands you can run.

### Example

```
$ python issmile.py --show %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Arnold_Schwarzenegger\Arnold_Schwarzenegger_0006.jpg
$ python issmile.py --show %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Yoko_Ono\Yoko_Ono_0003.jpg
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

## Authors

* **[Martin Kondor](https://github.com/MartinKondor)**

<p align="center"><a href="https://www.patreon.com/bePatron?u=17006186" data-patreon-widget-type="become-patron-button"><img width="222" class="img-responsive" alt="Become a Patron!" title="Become a Patron!" src="https://martinkondor.github.io/img/become_a_patron_button.png"></a></p>

# License

See the [LICENSE](LICENSE) file for details.
