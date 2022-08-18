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
+ `archive` contains code and explorations which are not necessarily relevant to the project anymore.
+ `src` contains the source code for the code which has been tested.

    - The `test` subdirectory also contains the testing suite used to validate the functions that are used.


+ `exploration` contains code which has been used to explore the data and try out new models and preprocessing steps
+ `data` contains a file, `data_dictionary.yaml` item IDs which can be used to retrieve the data from the DataStore



# Dependencies
You can either use a `conda` environment or `poetry` to install the dependencies of this project.


<!-- # Reference -->
<!-- ::: src.mhciipresentation.lstm.make_model -->
