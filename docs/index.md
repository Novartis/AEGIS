# Welcome to Aegis 👋

_The NIBR CBT efforts to understand which peptides are being presented by MHCII._

Welcome to the code repository of NIBR efforts to model peptide presentation by the MHCII using natural language processing-inspired modelling techniques.

Modelling the MHCII immunopeptidome is of paramount importance for a plethora of
biological applications, ranging from the humanization of therapeutic
(monoclonal) antibodies to the induction of tolerance against self antigens in
the context of autoimmune disease. Here, we document the code used to build such
a model.

The repository is structured as follows:

+ `docs` contains all of the documentation related to the project, including this page.
+ `src` contains the source code for the code which has been tested.


# How to get started
To get started working with these models:
1. Clone this repository
2. Download the data using this command [TODO: data download command]
3. Install the dependencies with `$ poetry install`
4. Install the `mhciipresentation` package with `$ cd mhciipresentation && pip install -e . && cd ../`
5. Retrain all the model variants with `$ ./scripts/train_models.sh`

<!-- # Reference -->
<!-- ::: src.mhciipresentation.lstm.make_model -->
